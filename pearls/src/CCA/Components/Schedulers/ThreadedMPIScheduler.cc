/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/ThreadedMPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Time.h>

#include <cstring>

#define USE_PACKING

using namespace Uintah;
using namespace SCIRun;

// sync cout/cerr so they are readable when output by multiple threads
extern SCIRun::Mutex coutLock;
extern SCIRun::Mutex cerrLock;

extern DebugStream taskdbg;
extern DebugStream mpidbg;
extern DebugStream waitout;
extern DebugStream execout;

extern std::map<std::string, double> waittimes;
extern std::map<std::string, double> exectimes;

static DebugStream threadedmpi_dbg(                  "ThreadedMPI_DBG",                  false);
static DebugStream threadedmpi_timeout(              "ThreadedMPI_TimingsOut",           false);
static DebugStream threadedmpi_queuelength(          "ThreadedMPI_QueueLength",          false);
static DebugStream threadedmpi_threaddbg(            "ThreadedMPI_ThreadDBG",            false);
static DebugStream threadedmpi_compactaffinity(      "ThreadedMPI_CompactAffinity",      true);
static DebugStream threadedmpi_scatteraffinity(      "ThreadedMPI_ScatterAffinity",      false);
static DebugStream threadedmpi_selectiveaffinity(    "ThreadedMPI_SelectiveAffinity",    false);
static DebugStream threadedmpi_miccompactaffinity2t( "ThreadedMPI_MICCompactAffinity2t", false);
static DebugStream threadedmpi_miccompactaffinity3t( "ThreadedMPI_MICCompactAffinity3t", false);
static DebugStream threadedmpi_miccompactaffinity4t( "ThreadedMPI_MICCompactAffinity4t", false);
static DebugStream threadedmpi_micscatteraffinity(   "ThreadedMPI_MICScatterAffinity",   false);
static DebugStream threadedmpi_micselectiveaffinity( "ThreadedMPI_MICSelectiveAffinity", false);


ThreadedMPIScheduler::ThreadedMPIScheduler( const ProcessorGroup*       myworld,
                                            const Output*               oport,
                                                  ThreadedMPIScheduler* parentScheduler)
  : MPIScheduler(myworld, oport, parentScheduler),
    d_nextsignal("next condition"),
    d_nextmutex("next mutex")
{
  if (threadedmpi_timeout.active()) {
    char filename[64];
    sprintf(filename, "timingStats.%d", d_myworld->myrank());
    timingStats.open(filename);
    if (d_myworld->myrank() == 0) {
      sprintf(filename, "timingStats.%d.max", d_myworld->size());
      maxStats.open(filename);
      sprintf(filename, "timingStats.%d.avg", d_myworld->size());
      avgStats.open(filename);
    }
  }
}

//______________________________________________________________________
//

ThreadedMPIScheduler::~ThreadedMPIScheduler()
{
  for( int i = 0; i < numThreads_; i++ ){
    t_worker[i]->d_runmutex.lock();
    t_worker[i]->quit();
    t_worker[i]->d_runsignal.conditionSignal();
    t_worker[i]->d_runmutex.unlock();
    t_thread[i]->setCleanupFunction( NULL );
    t_thread[i]->join();
  }

  if (threadedmpi_timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      maxStats.close();
      avgStats.close();
    }
  }
}

//______________________________________________________________________
//

void
ThreadedMPIScheduler::problemSetup( const ProblemSpecP&     prob_spec,
                                          SimulationStateP& state)
{
  // Default taskReadyQueueAlg
  taskQueueAlg_ = MostMessages;
  std::string taskQueueAlg = "MostMessages";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
    if (taskQueueAlg == "FCFS") {
      taskQueueAlg_ = FCFS;
    }
    else if (taskQueueAlg == "Random") {
      taskQueueAlg_ = Random;
    }
    else if (taskQueueAlg == "Stack") {
      taskQueueAlg_ = Stack;
    }
    else if (taskQueueAlg == "MostChildren") {
      taskQueueAlg_ = MostChildren;
    }
    else if (taskQueueAlg == "LeastChildren") {
      taskQueueAlg_ = LeastChildren;
    }
    else if (taskQueueAlg == "MostAllChildren") {
      taskQueueAlg_ = MostChildren;
    }
    else if (taskQueueAlg == "LeastAllChildren") {
      taskQueueAlg_ = LeastChildren;
    }
    else if (taskQueueAlg == "MostL2Children") {
      taskQueueAlg_ = MostL2Children;
    }
    else if (taskQueueAlg == "LeastL2Children") {
      taskQueueAlg_ = LeastL2Children;
    }
    else if (taskQueueAlg == "MostMessages") {
      taskQueueAlg_ = MostMessages;
    }
    else if (taskQueueAlg == "LeastMessages") {
      taskQueueAlg_ = LeastMessages;
    }
    else if (taskQueueAlg == "PatchOrder") {
      taskQueueAlg_ = PatchOrder;
    }
    else if (taskQueueAlg == "PatchOrderRandom") {
      taskQueueAlg_ = PatchOrderRandom;
    }
  }

  proc0cout << "   Using \"" << taskQueueAlg << "\" task queue priority algorithm" << std::endl;

  numThreads_ = Uintah::Parallel::getNumThreads() - 1;
  if ((numThreads_ < 1) && Uintah::Parallel::usingMPI()) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: no thread number specified for ThreadedMPIScheduler" << std::endl;
      throw ProblemSetupException("This scheduler requires number of threads to be in the range [2, 64],\n.... please use -nthreads <num>", __FILE__, __LINE__);
      }
    }
  else if (numThreads_ > MAX_THREADS) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: Number of threads too large..." << std::endl;
      throw ProblemSetupException("Too many threads. Reduce MAX_THREADS and recompile.", __FILE__, __LINE__);
    }
  }

  if (d_myworld->myrank() == 0) {
    std::string plural = (numThreads_ == 1) ? " thread" : " threads";
    std::cout << "   WARNING: Component tasks must be thread safe.\n"
              << "   Using 1 thread for scheduling, and " << numThreads_
              << plural + " for task execution." << std::endl;
  }

  // Reset Uintah thread ID (to reflect number of last physical core used)
  //
  // MIC affinity types are hard-coded for a 61-core Xeon Phi where the
  // the lastmost physical core contains logical cores 241, 242, 243, and 0
  //
  // NOTE: Logical core 0 is reserved for the Xeon Phi OS and the offload
  //       daemon also runs on this physical core
  if ( threadedmpi_micselectiveaffinity.active() ) {

    // These cases are hard-coded to support thread counts for 2 and 3
    // threads per physical core
    switch(Uintah::Parallel::getNumThreads()){
      case 120:
        Thread::self()->set_myid(240);
        break;
      case 121:
        Thread::self()->set_myid(241);
        break;
      case 122:
        Thread::self()->set_myid(242);
        break;
      case 180:
        Thread::self()->set_myid(240);
        break;
      case 181:
        Thread::self()->set_myid(241);
        break;
      case 182:
        Thread::self()->set_myid(242);
        break;
      case 183:
        Thread::self()->set_myid(243);
        break;
      default:
        Thread::self()->set_myid(Uintah::Parallel::getNumThreads() % 244);
        break;
    }
  }
  else if ( threadedmpi_miccompactaffinity2t.active() ||
            threadedmpi_miccompactaffinity3t.active() ||
            threadedmpi_miccompactaffinity4t.active() ||
            threadedmpi_micscatteraffinity.active()   ) {

    // These cases are hard-coded to support thread counts for 2 and 3
    // threads per physical core
    switch(Uintah::Parallel::getNumThreads()){
      case 120:
        Thread::self()->set_myid(238);
        break;
      case 121:
        Thread::self()->set_myid(241);
        break;
      case 122:
        Thread::self()->set_myid(242);
        break;
      case 180:
        Thread::self()->set_myid(239);
        break;
      case 181:
        Thread::self()->set_myid(241);
        break;
      case 182:
        Thread::self()->set_myid(242);
        break;
      case 183:
        Thread::self()->set_myid(243);
        break;
      default:
        Thread::self()->set_myid(Uintah::Parallel::getNumThreads() % 244);
        break;
    }
  }
  else {
    Thread::self()->set_myid(numThreads_);
  }

  // Disable CPU compact affinity to eliminate the need to explicitly disable this via SCI_DEBUG
  if ( threadedmpi_scatteraffinity.active()      ||
       threadedmpi_selectiveaffinity.active()    ||
       threadedmpi_miccompactaffinity2t.active() ||
       threadedmpi_miccompactaffinity3t.active() ||
       threadedmpi_miccompactaffinity4t.active() ||
       threadedmpi_micscatteraffinity.active()   ||
       threadedmpi_micselectiveaffinity.active() ) {

    threadedmpi_compactaffinity.setActive(false);
  }

  // CPU-specific binding of the main thread to the last physical core used
  if ( threadedmpi_compactaffinity.active()   ||
       threadedmpi_scatteraffinity.active()   ||
       threadedmpi_selectiveaffinity.active() ) {

    if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {
      threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                            << ") to CPU core " << numThreads_ << "\n";
    }
    Thread::self()->set_affinity(numThreads_); // CPU - bind main thread to last physical core
  }

  // MIC-specific binding of the main thread
  if ( threadedmpi_micselectiveaffinity.active() ) {

    // These cases are hard-coded to support thread counts for 2 and 3
    // threads per physical core
    switch(Uintah::Parallel::getNumThreads()){
      case 120:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 240 << "\n";
        }
        Thread::self()->set_affinity(240); // MIC - bind main thread to logical core 240
        break;
      case 121:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 241 << "\n";
        }
        Thread::self()->set_affinity(241); // MIC - bind main thread to logical core 241
        break;
      case 122:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 242 << "\n";
        }
        Thread::self()->set_affinity(242); // MIC - bind main thread to logical core 242
        break;
      case 180:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 240 << "\n";
        }
        Thread::self()->set_affinity(240); // MIC - bind main thread to logical core 240
        break;
      case 181:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 241 << "\n";
        }
        Thread::self()->set_affinity(241); // MIC - bind main thread to logical core 241
        break;
      case 182:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 242 << "\n";
        }
        Thread::self()->set_affinity(242); // MIC - bind main thread to logical core 242
        break;
      case 183:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 243 << "\n";
        }
        Thread::self()->set_affinity(243); // MIC - bind main thread to logical core 243
        break;
      default:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << Uintah::Parallel::getNumThreads() % 244 << "\n";
        }
        Thread::self()->set_affinity(Uintah::Parallel::getNumThreads() % 244); // MIC - bind main thread to last logical core used
        break;
    }
  }
  else if ( threadedmpi_miccompactaffinity2t.active() ||
            threadedmpi_miccompactaffinity3t.active() ||
            threadedmpi_miccompactaffinity4t.active() ||
            threadedmpi_micscatteraffinity.active()   ) {

    // These cases are hard-coded to support thread counts for 2 and 3
    // threads per physical core
    switch(Uintah::Parallel::getNumThreads()){
      case 120:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 238 << "\n";
        }
        Thread::self()->set_affinity(238); // MIC - bind main thread to logical core 238
        break;
      case 121:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 241 << "\n";
        }
        Thread::self()->set_affinity(241); // MIC - bind main thread to logical core 241
        break;
      case 122:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 242 << "\n";
        }
        Thread::self()->set_affinity(242); // MIC - bind main thread to to logical core 242
        break;
      case 180:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 239 << "\n";
        }
        Thread::self()->set_affinity(239); // MIC - bind main thread to logical core 239
        break;
      case 181:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 241 << "\n";
        }
        Thread::self()->set_affinity(241); // MIC - bind main thread to logical core 241
        break;
      case 182:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 242 << "\n";
        }
        Thread::self()->set_affinity(242); // MIC - bind main thread to logical core 242
        break;
      case 183:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << 243 << "\n";
        }
        Thread::self()->set_affinity(243); // MIC - bind main thread to logical core 243
        break;
      default:
        if ( (threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0) ) {

          threadedmpi_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid()
                                << ") to MIC core " << Uintah::Parallel::getNumThreads() % 244 << "\n";
        }
        Thread::self()->set_affinity(Uintah::Parallel::getNumThreads() % 244); // MIC - bind main thread to last logical core used
        break;
    }
  }

  // Create the TaskWorkers here (pinned to cores in TaskWorker::run())
  char name[1024];

  // MIC-specific TaskWorker indexing
  if (threadedmpi_miccompactaffinity2t.active() ||
      threadedmpi_miccompactaffinity3t.active() ||
      threadedmpi_miccompactaffinity4t.active() ||
      threadedmpi_micscatteraffinity.active()   ||
      threadedmpi_micselectiveaffinity.active() ) {
    for (int i = 0; i < numThreads_; i++) {
          TaskWorker* worker = scinew TaskWorker(this, i + 1); // MIC-specific TaskWorkers are indexed from core 1
          t_worker[i] = worker;
          sprintf(name, "Computing Worker %d-%d", Parallel::getRootProcessorGroup()->myrank(), i + 1);
          Thread* t = scinew Thread(worker, name);
          t_thread[i] = t;
        }
  }
  // CPU-specific TaskWorker indexing
  else {
    for (int i = 0; i < numThreads_; i++) {
      TaskWorker* worker = scinew TaskWorker(this, i); // CPU-specific TaskWorkers are indexed from core 0
      t_worker[i] = worker;
      sprintf(name, "Computing Worker %d-%d", Parallel::getRootProcessorGroup()->myrank(), i);
      Thread* t = scinew Thread(worker, name);
      t_thread[i] = t;
    }
  }

  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}

//______________________________________________________________________
//

SchedulerP
ThreadedMPIScheduler::createSubScheduler()
{
  ThreadedMPIScheduler* subsched = scinew ThreadedMPIScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  subsched->attachPort("load balancer", lbp);
  subsched->d_sharedState = d_sharedState;

  subsched->numThreads_ = Uintah::Parallel::getNumThreads() - 1;

  // Reset Uintah subscheduler thread ID (to reflect number of last physical core used)
  Thread::self()->set_myid(numThreads_);

  if (subsched->numThreads_ > 0) {

    std::string plural = (numThreads_ == 1) ? " thread" : " threads";
    proc0cout << "\n"
              << "   Using multi-threaded sub-scheduler\n"
              << "   WARNING: Component tasks must be thread safe.\n"
              << "   Using 1 thread for scheduling.\n"
              << "   Creating " << subsched->numThreads_ << plural << " for task execution.\n\n" << std::endl;

    // CPU-specific binding of the main thread to the last physical core used
    if ( threadedmpi_compactaffinity.active() ) {
      if ((threadedmpi_threaddbg.active()) && (d_myworld->myrank() == 0)) {
        threadedmpi_threaddbg << "Binding main subscheduler thread (ID "
                              << Thread::self()->myid() << ") to CPU core "
                              << subsched->numThreads_ << "\n";
      }
      Thread::self()->set_affinity(numThreads_);    // CPU - bind main subscheduler thread to last physical core
    }

    // Create TaskWorker threads for the subscheduler
    char name[1024];
    ThreadGroup* subGroup = new ThreadGroup("subscheduler-group", 0);  // 0 is main/parent thread group
    for (int i = 0; i < subsched->numThreads_; i++) {
      TaskWorker* worker = scinew TaskWorker(subsched, i);
      subsched->t_worker[i] = worker;
      sprintf(name, "Task Compute Thread ID: %d", i + subsched->numThreads_);
      Thread* t = scinew Thread(worker, name, subGroup);
      subsched->t_thread[i] = t;
    }
  }

  return subsched;

}

//______________________________________________________________________
//

void
ThreadedMPIScheduler::execute( int tgnum     /* = 0 */,
                               int iteration /* = 0 */ )
{
  // copy data timestep must be single threaded for now
  if (d_sharedState->isCopyDataTimestep()) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  MALLOC_TRACE_TAG_SCOPE("ThreadedMPIScheduler::execute");

  TAU_PROFILE("ThreadedMPIScheduler::execute()", " ", TAU_USER);
  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[ThreadedMPIScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[ThreadedMPIScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[ThreadedMPIScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[ThreadedMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(testsometimer, "Test Some", "[ThreadedMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[ThreadedMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[ThreadedMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[ThreadedMPIScheduler::execute()] ", TAU_USER);

  ASSERTRANGE(tgnum, 0, static_cast<int>(graphs.size()));
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  DetailedTasks* dts = tg->getDetailedTasks();

  if (dts == 0) {
    proc0cout << "ThreadedMPIScheduler skipping execute, no tasks\n";
    return;
  }

  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  if (threadedmpi_timeout.active()) {
    d_labels.clear();
    d_times.clear();
    //emitTime("time since last execute");
  }

  int me = d_myworld->myrank();
  makeTaskGraphDoc(dts, me);

  // TODO - figure out and fix this (APH - 01/12/15)
//  if (timeout.active()) {
//    emitTime("taskGraph output");
//  }

  mpi_info_.totalreduce    = 0;
  mpi_info_.totalsend      = 0;
  mpi_info_.totalrecv      = 0;
  mpi_info_.totaltask      = 0;
  mpi_info_.totalreducempi = 0;
  mpi_info_.totalsendmpi   = 0;
  mpi_info_.totalrecvmpi   = 0;
  mpi_info_.totaltestmpi   = 0;
  mpi_info_.totalwaitmpi   = 0;

  int numTasksDone = 0;
  bool abort = false;
  int abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  TAU_PROFILE_TIMER(doittimer, "Task execution", "[ThreadedMPIScheduler::execute() loop] ", TAU_USER);
  TAU_PROFILE_START(doittimer);

  int currphase = 0;
  int numPhases = tg->getNumTaskPhases();
  std::vector<int> phaseTasks(numPhases);
  std::vector<int> phaseTasksDone(numPhases);
  std::vector<DetailedTask*> phaseSyncTask(numPhases);

  dts->setTaskPriorityAlg(taskQueueAlg_);

  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }

  if (threadedmpi_dbg.active()) {
    cerrLock.lock();
    {
      threadedmpi_dbg << "\n"
                      << "Rank-" << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)\n"
                      << "Total task phases: " << numPhases
                      << "\n";
      for (size_t phase = 0; phase < phaseTasks.size(); ++phase) {
        threadedmpi_dbg << "Phase: " << phase << " has " << phaseTasks[phase] << " total tasks\n";
      }
      threadedmpi_dbg << std::endl;
    }
    cerrLock.unlock();
  }

  static int totaltasks;
  static std::vector<int> histogram;
  std::set<DetailedTask*> pending_tasks;

  if (taskdbg.active()) {
    cerrLock.lock();
    taskdbg << "Rank-" << me << " starting task phase " << currphase << ", total phase " << currphase << " tasks = "
            << phaseTasks[currphase] << std::endl;
    cerrLock.unlock();
  }

  for (int i = 0; i < numThreads_; i++) {
    t_worker[i]->resetWaittime(Time::currentSeconds());
  }

  // The main task loop
  while (numTasksDone < ntasks) {

    if (phaseTasks[currphase] == phaseTasksDone[currphase]) {  // this phase done, goto next phase
      currphase++;
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Rank-" << me << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                << phaseTasks[currphase] << std::endl;
        cerrLock.unlock();
      }
    }
    // if we have an internally-ready task, initiate its recvs
    else if (dts->numInternalReadyTasks() > 0) {
      DetailedTask* task = dts->getNextInternalReadyTask();
      // save the reduction task and once per proc task for later execution
      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {
        phaseSyncTask[task->getTask()->d_phase] = task;
        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << "Rank-" << me << " Task Reduction/OPP ready " << *task << " deps needed: " << task->getExternalDepCount()
                  << std::endl;
          cerrLock.unlock();
        }
      }
      else {
        initiateTask(task, abort, abort_point, iteration);
        task->markInitiated();
        task->checkExternalDepCount();
        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << "Rank-" << me << " Task internal ready " << *task << " deps needed: " << task->getExternalDepCount()
                  << std::endl;
          cerrLock.unlock();
          pending_tasks.insert(task);
        }
      }
    }
    //if it is time to run reduction task
    else if ((phaseSyncTask[currphase] != NULL) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {
      if (threadedmpi_queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }
      DetailedTask* reducetask = phaseSyncTask[currphase];

      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Rank-" << me << " Ready Reduce/OPP task " << reducetask->getTask()->getName() << std::endl;
        cerrLock.unlock();
      }

      if (reducetask->getTask()->getType() == Task::Reduction) {
        if (!abort) {
          if (taskdbg.active()) {
            cerrLock.lock();
            taskdbg << "Rank-" << me << " Running Reduce task " << reducetask->getTask()->getName() << " with communicator "
                    << reducetask->getTask()->d_comm << std::endl;
            cerrLock.unlock();
          }
          assignTask(reducetask, iteration);
        }
      }
      else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->usesMPI());
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();

        ASSERT(reducetask->getExternalDepCount() == 0)
        assignTask(reducetask, iteration);

        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << "Rank-" << me << " Running OPP task: ";
          cerrLock.unlock();
        }

        cerrLock.lock();
        printTask(taskdbg, reducetask);
        cerrLock.unlock();

        if (taskdbg.active()) {
          taskdbg << '\n';
        }
      }
      ASSERT(reducetask->getTask()->d_phase == currphase);
      numTasksDone++;
      phaseTasksDone[reducetask->getTask()->d_phase]++;
    }

    // TODO - need to update these comments (APH - 01/12/15)
    // run a task that has its communication complete
    // tasks get in this queue automatically when their receive count hits 0
    //   in DependencyBatch::received, which is called when a message is delivered.
    else if (dts->numExternalReadyTasks() > 0) {

      if (threadedmpi_queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }

      DetailedTask* task = dts->getNextExternalReadyTask();
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Rank-" << me << " Dispatching task " << *task << "(" << dts->numExternalReadyTasks() << "/"
                << pending_tasks.size() << " tasks in queue)" << std::endl;
        cerrLock.unlock();
        pending_tasks.erase(pending_tasks.find(task));
      }
      ASSERTEQ(task->getExternalDepCount(), 0);
      assignTask(task, iteration);
      numTasksDone++;
      phaseTasksDone[task->getTask()->d_phase]++;
    }
    else { // nothing to do process MPI
      processMPIRecvs(TEST);
    }

  }  // end while( numTasksDone < ntasks )

  TAU_PROFILE_STOP(doittimer);

  // wait for all tasks to finish
  d_nextmutex.lock();
  while (getAviableThreadNum() < numThreads_) {
    // if any thread is busy, conditional wait here
    d_nextsignal.wait(d_nextmutex);
  }
  d_nextmutex.unlock();

  if (threadedmpi_queuelength.active()) {
    float lengthsum = 0;
    totaltasks += ntasks;
    for (unsigned int i = 1; i < histogram.size(); i++) {
      lengthsum = lengthsum + i * histogram[i];
    }
    float queuelength = lengthsum / totaltasks;
    float allqueuelength = 0;

    MPI_Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());

    proc0cout << "average queue length:" << allqueuelength / d_myworld->size() << std::endl;
  }

  if (threadedmpi_timeout.active()) {
    emitTime("MPI send time", mpi_info_.totalsendmpi);
    emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
    emitTime("Total send time", mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time", mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    emitTime("Total reduction time", mpi_info_.totalreduce - mpi_info_.totalreducempi);
    emitTime("Total comm time", mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

    double time = Time::currentSeconds();
    double totalexec = time - d_lasttime;

    d_lasttime = time;

    emitTime("Other excution time",
             totalexec - mpi_info_.totalsend - mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);
  }

  if (d_sharedState != 0) {
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime; // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
    for (int i = 0; i < numThreads_; i++) {
      d_sharedState->taskWaitThreadTime += t_worker[i]->getWaittime();
    }
  }

  //if(timeout.active())
  //emitTime("final wait");
  if (restartable && tgnum == (int)graphs.size() - 1) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size() - 1]->timestepRestarted();
    int netrestart;

    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR, d_myworld->getComm());

    if (netrestart) {
      dws[dws.size() - 1]->restartTimestep();
      if (dws[0]) {
        dws[0]->setRestarted();
      }
    }
  }

  finalizeTimestep();

  log.finishTimestep();

  if (threadedmpi_timeout.active() && !parentScheduler_) {  // only do on toplevel scheduler
    //emitTime("finalize");

    // add number of cells, patches, and particles
    int numCells = 0, numParticles = 0;
    OnDemandDataWarehouseP dw = dws[dws.size() - 1];
    const GridP grid(const_cast<Grid*>(dw->getGrid()));
    const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(grid)->getSubset(d_myworld->myrank());
    for (int p = 0; p < myPatches->size(); p++) {
      const Patch* patch = myPatches->get(p);
      IntVector range = patch->getExtraCellHighIndex() - patch->getExtraCellLowIndex();
      numCells += range.x() * range.y() * range.z();

      // go through all materials since getting an MPMMaterial correctly would depend on MPM
      for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
        if (dw->haveParticleSubset(m, patch)) {
          numParticles += dw->getParticleSubset(m, patch)->numParticles();
        }
      }
    }

    // Used in printing simulation stats
    emitTime("NumPatches", myPatches->size());
    emitTime("NumCells", numCells);
    emitTime("NumParticles", numParticles);
    std::vector<double> d_totaltimes(d_times.size());
    std::vector<double> d_maxtimes(d_times.size());
    std:: vector<double> d_avgtimes(d_times.size());
    double avgTask = -1, maxTask = -1;
    double avgComm = -1, maxComm = -1;
    double avgCell = -1, maxCell = -1;

    // Get the SUM and MAX reduction times for simulation stats
    MPI_Comm comm = d_myworld->getComm();
    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&d_times[0], &d_maxtimes[0],   (int)d_times.size(), MPI_DOUBLE, MPI_MAX, 0, comm);

    double total = 0, avgTotal = 0, maxTotal = 0;
    for (int i = 0; i < static_cast<int>(d_totaltimes.size()); i++) {
      d_avgtimes[i] = d_totaltimes[i] / d_myworld->size();
      if (strcmp(d_labels[i], "Total task time") == 0) {
        avgTask = d_avgtimes[i];
        maxTask = d_maxtimes[i];
      }
      else if (strcmp(d_labels[i], "Total comm time") == 0) {
        avgComm = d_avgtimes[i];
        maxComm = d_maxtimes[i];
      }
      else if (strncmp(d_labels[i], "Num", 3) == 0) {
        if (strcmp(d_labels[i], "NumCells") == 0) {
          avgCell = d_avgtimes[i];
          maxCell = d_maxtimes[i];
        }
        // these are independent stats - not to be summed
        continue;
      }

      total    += d_times[i];
      avgTotal += d_avgtimes[i];
      maxTotal += d_maxtimes[i];
    }

    // to not duplicate the code
    std::vector<std::ofstream*> files;
    std::vector<std::vector<double>*> data;
    files.push_back(&timingStats);
    data.push_back(&d_times);

    if (me == 0) {
      files.push_back(&avgStats);
      files.push_back(&maxStats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (unsigned int file = 0; file < files.size(); file++) {
      std::ofstream& out = *files[file];
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep() << std::endl;
      for (int i = 0; i < static_cast<int>((*data[file]).size()); i++) {
        out << "ThreadedMPIScheduler: " << d_labels[i] << ": ";
        int len = static_cast<int>((strlen(d_labels[i])) + strlen("ThreadedMPIScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++) {
          out << ' ';
        }
        double percent = 0.0;
        if (strncmp(d_labels[i], "Num", 3) == 0) {
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i] / d_totaltimes[i] * 100;
        }
        else {
          percent = (*data[file])[i] / total * 100;
        }
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << std::endl << std::endl;
    }

    if (me == 0) {
      threadedmpi_timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1 - avgTask / maxTask) * 100 << " load imbalance (exec)%\n";
      threadedmpi_timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1 - avgComm / maxComm) * 100 << " load imbalance (comm)%\n";
      threadedmpi_timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1 - avgCell / maxCell) * 100 << " load imbalance (theoretical)%\n\n";
    }

    // TODO - need to clean this up (APH - 01/22/15)
    double time  = Time::currentSeconds();
//    double rtime = time - d_lasttime;
    d_lasttime = time;
//    threadedmpi_timeout << "ThreadedMPIScheduler: TOTAL                                    " << total << '\n';
//    threadedmpi_timeout << "ThreadedMPIScheduler: time sum reduction (one processor only): " << rtime << '\n';
  } // end threadedmpi_timeout.active()

  if (execout.active()) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {
      std::ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);

      for (std::map<std::string, double>::iterator iter = exectimes.begin(); iter != exectimes.end(); iter++) {
        fout << std::fixed << d_myworld->myrank() << ": TaskExecTime: " << iter->second << " Task:" << iter->first << std::endl;
      }
      fout.close();
      exectimes.clear();
    }
  }

  if (waitout.active()) {
    static int count = 0;

    // only output the wait times every 10 timesteps
    if (++count % 10 == 0) {
      if (d_myworld->myrank() == 0 || d_myworld->myrank() == d_myworld->size() / 2
          || d_myworld->myrank() == d_myworld->size() - 1) {

        std::ofstream wout;
        char fname[100];
        sprintf(fname, "waittimes.%d.%d", d_myworld->size(), d_myworld->myrank());
        wout.open(fname);

        for (std::map<std::string, double>::iterator iter = waittimes.begin(); iter != waittimes.end(); iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second << " Task:" << iter->first << std::endl;
        }

        for (std::map<std::string, double>::iterator iter = DependencyBatch::waittimes.begin(); iter != DependencyBatch::waittimes.end();
            iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second << " Task:" << iter->first << std::endl;
        }
        wout.close();
      }

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }

  if (threadedmpi_dbg.active()) {
    coutLock.lock();
    threadedmpi_dbg << "Rank-" << me << " - ThreadedMPIScheduler finished" << std::endl;
    coutLock.unlock();
  }
} // end execute()

//______________________________________________________________________
//

int ThreadedMPIScheduler::getAviableThreadNum()
{
  int num = 0;
  for (int i = 0; i < numThreads_; i++) {
    if (t_worker[i]->d_task == NULL) {
      num++;
    }
  }
  return num;
}

//______________________________________________________________________
//

void ThreadedMPIScheduler::assignTask( DetailedTask* task,
                                       int           iteration )
{
  d_nextmutex.lock();
  if (getAviableThreadNum() == 0) {
    d_nextsignal.wait(d_nextmutex);
  }
  // find an idle thread and assign task
  int targetThread = -1;
  for (int i = 0; i < numThreads_; i++) {
    if (t_worker[i]->d_task == NULL) {
      targetThread = i;
      break;
    }
  }
  d_nextmutex.unlock();

  ASSERT(targetThread >= 0);

  // send task and wake up worker
  t_worker[targetThread]->d_runmutex.lock();
  t_worker[targetThread]->d_task = task;
  t_worker[targetThread]->d_iteration = iteration;
  t_worker[targetThread]->d_runsignal.conditionSignal();
  t_worker[targetThread]->d_runmutex.unlock();
}

//------------------------------------------
// TaskWorker Thread Methods
//------------------------------------------
TaskWorker::TaskWorker( ThreadedMPIScheduler* scheduler,
                        int                   thread_id )
  : d_scheduler( scheduler ),
    d_task( NULL ),
    d_runsignal( "run condition" ),
    d_runmutex("run mutex"),
    d_quit( false ),
    d_thread_id( thread_id ),
    d_rank( scheduler->getProcessorGroup()->myrank() ),
    d_iteration( 0 ),
    d_waittime( 0.0 ),
    d_waitstart( 0.0 )
{
  d_runmutex.lock();
}

//______________________________________________________________________
//

void
TaskWorker::run()
{

  // Set Uintah thread ID
  Thread::self()->set_myid(d_thread_id);

  // CPU-specific compact affinity
  if (threadedmpi_compactaffinity.active()) {
    if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
      cerrLock.lock();
      std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to CPU core " << d_thread_id << "\n";
      cerrLock.unlock();
    }
    Thread::self()->set_affinity(d_thread_id);
  }

  // CPU-specific scatter affinity
  if (threadedmpi_scatteraffinity.active()) {

    // This is hard-coded for a dual socket configuration featuring two
    // Xeon E5 CPUs with 8 physical cores and 16 logical cores each

    int physCoresPerSocket  = 8;
    int socketIndex         = 0;

    // Determine whether the thread will be bound to a logical core on the 0th or 1st socket
    socketIndex = d_thread_id%2;

    // The thread will be bound to a logical core on the 0th socket
    if ( socketIndex == 0 ) {

      // The thread will be bound to the 0th logical core on a given physical core
      if ( d_thread_id < 16 ) {

        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to CPU core " << d_thread_id/2 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinity(d_thread_id/2);

      // The thread will be bound to the 1st logical core on a given physical core
      } else if ( d_thread_id >= 16 ) {

        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to CPU core " << physCoresPerSocket + d_thread_id/2 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinity(physCoresPerSocket + d_thread_id/2);
      }

    // The thread will be bound to a logical core on the 1st socket
    } else if ( socketIndex == 1 ) {

      // The thread will be bound to the 0th logical core on a given physical core
      if ( d_thread_id < 16 ) {

        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to CPU core " << physCoresPerSocket + floor(d_thread_id/2) << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinity(physCoresPerSocket + floor(d_thread_id/2));

      // The thread will be bound to the 1st logical core on a given physical core
      } else if ( d_thread_id >= 16 ) {

        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to CPU core " << 2*physCoresPerSocket + floor(d_thread_id/2) << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinity(2*physCoresPerSocket + floor(d_thread_id/2));
      }
    }
  }

  // CPU-specific selective affinity
  if (threadedmpi_selectiveaffinity.active()) {
    if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
      cerrLock.lock();
      std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any CPU core between 0 and " << Uintah::Parallel::getNumThreads() - 2 << "\n";
      cerrLock.unlock();
    }
    Thread::self()->set_affinityMainOnly(Uintah::Parallel::getNumThreads());
  }

  // MIC-specific compact affinity for 2 threads per core
  if (threadedmpi_miccompactaffinity2t.active()) {

    // Handle threads to be bound to the 1st logical core in a given physical core
    if ( (d_thread_id-1) % 2 == 0 ) {

      int logCoresPerPhysCore = 4;
      int overallIndex = 0;

      overallIndex = ( ((d_thread_id-1)/2) * logCoresPerPhysCore ) + 1;

      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << overallIndex << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(overallIndex);

    // Handle threads to be bound to the 2nd logical core in a given physical core
    } else {

      int logCoresPerPhysCore = 4;
      int overallIndex = 0;

      overallIndex = ( ((d_thread_id-2)/2) * logCoresPerPhysCore ) + 2;

      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << overallIndex << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(overallIndex);
    }
  }

  // MIC-specific compact affinity for 3 threads per core
  if (threadedmpi_miccompactaffinity3t.active()) {

    // Handle threads to be bound to the 1st logical core in a given physical core
    if ( (d_thread_id-1) % 3 == 0 ) {

      int logCoresPerPhysCore = 4;
      int overallIndex = 0;

      overallIndex = ( ((d_thread_id-1)/3) * logCoresPerPhysCore ) + 1;

      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << overallIndex << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(overallIndex);

    // Handle threads to be bound to the 2nd logical core in a given physical core
    } else if ( (d_thread_id-2) % 3 == 0 ) {

      int logCoresPerPhysCore = 4;
      int overallIndex = 0;

      overallIndex = ( ((d_thread_id-2)/3) * logCoresPerPhysCore ) +2;

      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << overallIndex << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(overallIndex);

    // Handle threads to be bound to the 3rd logical core in a given physical core
    } else {

      int logCoresPerPhysCore = 4;
      int overallIndex = 0;

      overallIndex = ( ((d_thread_id-3)/3) * logCoresPerPhysCore ) + 3;

      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << overallIndex << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(overallIndex);
    }
  }

  // MIC-specific compact affinity for 4 threads per core
  if (threadedmpi_miccompactaffinity4t.active()) {
    if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
      cerrLock.lock();
      std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << d_thread_id << "\n";
      cerrLock.unlock();
    }
    Thread::self()->set_affinity(d_thread_id);
  }

  // MIC-specific scatter affinity
  if (threadedmpi_micscatteraffinity.active()) {

    int scatterPhysCores    = 60; // Scatter across 60 cores to experiment with the 61st core
    int logCoresPerPhysCore = 4;
    int logCoreIndex        = 0;
    int physCoreIndex       = 0;
    int overallIndex        = 0;

    // These cases are used to place threads the 61st core
    if (d_thread_id > 240) {
      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << d_thread_id << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(d_thread_id);
    }
    else if ( Uintah::Parallel::getNumThreads() == 122 && d_thread_id == 121) {
      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << 241 << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(241);
    }
    else if ( ( Uintah::Parallel::getNumThreads() == 182 && d_thread_id == 181) ||
              ( Uintah::Parallel::getNumThreads() == 183 && d_thread_id == 181) ||
              ( Uintah::Parallel::getNumThreads() == 183 && d_thread_id == 182) ) {
      if (d_thread_id == 181) {
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << 241 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinity(241);
      }
      else if (d_thread_id == 182) {
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << 242 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinity(242);
      }
    }
    else {

      // Determine whether the thread will be bound to the 1st, 2nd, 3rd, or 4th logical core in a given physical core
      logCoreIndex = floor ( (d_thread_id - 1 ) / scatterPhysCores ) + 1;

      // Determine which physical core the thread will be bound to
      physCoreIndex = ( d_thread_id - (( logCoreIndex - 1 ) * scatterPhysCores ));

      // Determine the specific logical core to bind the thread to
      overallIndex = logCoreIndex + ( physCoreIndex - 1 ) * logCoresPerPhysCore;

      if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
        cerrLock.lock();
        std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
        threadedmpi_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to MIC core " << overallIndex << "\n";
        cerrLock.unlock();
      }
      Thread::self()->set_affinity(overallIndex);
    }
  }

  // MIC-specific selective affinity
  if (threadedmpi_micselectiveaffinity.active()) {

    // These cases are hard-coded to support thread counts for 2 and 3
    // threads per physical core
    switch(Uintah::Parallel::getNumThreads()){
      case 120:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 239 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(240);
        break;
      case 121:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 240 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(241);
        break;
      case 122:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 241 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(242);
        break;
      case 180:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 239 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(240);
        break;
      case 181:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 240 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(241);
        break;
      case 182:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 241 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(242);
        break;
      case 183:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << 242 << "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(243);
        break;
      default:
        if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
          cerrLock.lock();
          std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
          threadedmpi_threaddbg << "Allowing" << threadType << "thread ID " << d_thread_id << " to run on any MIC core between 1 and " << Uintah::Parallel::getNumThreads() - 1<< "\n";
          cerrLock.unlock();
        }
        Thread::self()->set_affinityMICMainOnly(Uintah::Parallel::getNumThreads());
        break;
    }
  }

  while (true) {
    d_runsignal.wait(d_runmutex); // wait for main thread signal
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds() - d_waitstart;

    if (d_quit) {
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Worker " << d_rank << "-" << d_thread_id << " quitting\n";
        cerrLock.unlock();
      }
      return;
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank << "-" << d_thread_id << ": began executing task: " << *d_task << "\n";
      cerrLock.unlock();
    }

    ASSERT(d_task != NULL);
    try {
      if (d_task->getTask()->getType() == Task::Reduction) {
        d_scheduler->initiateReduction(d_task);
      }
      else {
        d_scheduler->runTask(d_task, d_iteration, d_thread_id);
      }
    }
    catch (Exception& e) {
      cerrLock.lock();
      std::cerr << "Worker " << d_rank << "-" << d_thread_id << ": Caught exception: " << e.message() << "\n";
      if (e.stackTrace()) {
        std::cerr << "Stack trace: " << e.stackTrace() << '\n';
      }
      cerrLock.unlock();
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank << "-" << d_thread_id << ": finished executing task: " << *d_task << std::endl;
      cerrLock.unlock();
    }

    // Signal main thread for next task.
    d_scheduler->d_nextmutex.lock();
    d_runmutex.lock();
    d_task = NULL;
    d_iteration = 0;
    d_waitstart = Time::currentSeconds();
    d_scheduler->d_nextsignal.conditionSignal();
    d_scheduler->d_nextmutex.unlock();
  }
}

//______________________________________________________________________
//

double TaskWorker::getWaittime()
{
    return  d_waittime;
}

//______________________________________________________________________
//

void TaskWorker::resetWaittime( double start )
{
    d_waitstart = start;
    d_waittime  = 0.0;
}

