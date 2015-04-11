/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Time.h>

#ifdef HAVE_CUDA
#  include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#  include <Core/Grid/Variables/GPUGridVariable.h>
#  include <Core/Grid/Variables/GPUStencil7.h>
#endif

#include <sci_defs/cuda_defs.h>

#include <cstring>

#define USE_PACKING

using namespace Uintah;

// sync cout/cerr so they are readable when output by multiple threads
extern SCIRun::Mutex coutLock;
extern SCIRun::Mutex cerrLock;

extern DebugStream taskdbg;
extern DebugStream mpidbg;
extern DebugStream waitout;
extern DebugStream execout;
extern DebugStream taskorder;
extern DebugStream taskLevel_dbg;

extern std::map<std::string, double> waittimes;
extern std::map<std::string, double> exectimes;

static double Unified_CurrentWaitTime = 0;

static DebugStream unified_dbg(             "Unified_DBG",             false);
static DebugStream unified_timeout(         "Unified_TimingsOut",      false);
static DebugStream unified_queuelength(     "Unified_QueueLength",     false);
static DebugStream unified_threaddbg(       "Unified_ThreadDBG",       false);
static DebugStream unified_compactaffinity( "Unified_CompactAffinity", true);

#ifdef HAVE_CUDA
  static DebugStream gpu_stats(        "Unified_GPUStats",     false);
         DebugStream use_single_device("Unified_SingleDevice", false);
#endif

//______________________________________________________________________
//

UnifiedScheduler::UnifiedScheduler( const ProcessorGroup*   myworld,
                                    const Output*           oport,
                                          UnifiedScheduler* parentScheduler )
  : MPIScheduler(myworld, oport, parentScheduler),
    d_nextsignal("next condition"),
    d_nextmutex("next mutex"),
    schedulerLock("scheduler lock")
#ifdef HAVE_CUDA
  ,
  idleStreamsLock_("CUDA streams lock"),
  d2hComputesLock_("Device-DB computes copy lock"),
  h2dRequiresLock_("Device-DB requires copy lock")
#endif
{
#ifdef HAVE_CUDA
  if (Uintah::Parallel::usingDevice()) {
    gpuInitialize();

    // we need one of these for each GPU, as each device will have it's own CUDA context
    for (int i = 0; i < numDevices_; i++) {
      idleStreams.push_back(std::queue<cudaStream_t*>());
    }

    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::d_combineMemory = false;
  }
#endif

  if (unified_timeout.active()) {
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

UnifiedScheduler::~UnifiedScheduler()
{
  if (Uintah::Parallel::usingMPI()) {
    for (int i = 0; i < numThreads_; i++) {
      t_worker[i]->d_runmutex.lock();
      t_worker[i]->quit();
      t_worker[i]->d_runsignal.conditionSignal();
      t_worker[i]->d_runmutex.unlock();
      t_thread[i]->setCleanupFunction(NULL);
      t_thread[i]->join();
    }
  }

  if (unified_timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      maxStats.close();
      avgStats.close();
    }
  }
#ifdef HAVE_CUDA
  freeCudaStreams();
#endif
}

//______________________________________________________________________
//

void
UnifiedScheduler::problemSetup( const ProblemSpecP&     prob_spec,
                                      SimulationStateP& state )
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
  if (numThreads_ < 1 && (Uintah::Parallel::usingMPI() || Uintah::Parallel::usingDevice())) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: no thread number specified for Unified Scheduler" << std::endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads to be in the range [2, 64],\n.... please use -nthreads <num>, and -gpu if using GPUs",
          __FILE__,
          __LINE__);
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
    std::cout << "   WARNING: Multi-threaded Unified scheduler is EXPERIMENTAL, not all tasks are thread safe yet.\n"
              << "   Creating " << numThreads_ << " additional "
              << plural + " for task execution (total task execution threads = "
              << numThreads_ + 1 << ")." << "\n";
#ifdef HAVE_CUDA
    if (Uintah::Parallel::usingDevice()) {
      cudaError_t retVal;
      int availableDevices = numDevices_;
      CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&availableDevices));
      std::cout << "   Using " << numDevices_ << "/" << availableDevices << " available GPU(s)" << std::endl;
    }
#endif
  }

  if (unified_compactaffinity.active()) {
    if ( (unified_threaddbg.active()) && (d_myworld->myrank() == 0) ) {
      unified_threaddbg << "   Binding main thread (ID "<<  Thread::self()->myid() << ") to core 0\n";
    }
    Thread::self()->set_affinity(0);  // Bind main thread to core 0
  }

  // Create the UnifiedWorkers here (pinned to cores in UnifiedSchedulerWorker::run())
  char name[1024];
  for (int i = 0; i < numThreads_; i++) {
    UnifiedSchedulerWorker* worker = scinew UnifiedSchedulerWorker(this, i);
    t_worker[i] = worker;
    sprintf(name, "Computing Worker %d-%d", Parallel::getRootProcessorGroup()->myrank(), i);
    Thread* t = scinew Thread(worker, name);
    t_thread[i] = t;
  }

  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}

//______________________________________________________________________
//

SchedulerP
UnifiedScheduler::createSubScheduler()
{
  UnifiedScheduler* subsched = scinew UnifiedScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  subsched->attachPort("load balancer", lbp);
  subsched->d_sharedState = d_sharedState;
  subsched->numThreads_ = Uintah::Parallel::getNumThreads() - 1;

  if (subsched->numThreads_ > 0) {

    proc0cout << "\n"
              << "   Using EXPERIMENTAL multi-threaded sub-scheduler\n"
              << "   WARNING: Component tasks must be thread safe.\n"
              << "   Creating " << subsched->numThreads_ << " subscheduler threads for task execution.\n\n" << std::endl;

    // Bind main execution thread
    if (unified_compactaffinity.active()) {
      if ( (unified_threaddbg.active()) && (d_myworld->myrank() == 0) ) {
        unified_threaddbg << "Binding main subscheduler thread (ID " << Thread::self()->myid() << ") to core 0\n";
      }
      Thread::self()->set_affinity(0);    // bind subscheduler main thread to core 0
    }

    // Create UnifiedWorker threads for the subscheduler
    char name[1024];
    ThreadGroup* subGroup = new ThreadGroup("subscheduler-group", 0);  // 0 is main/parent thread group
    for (int i = 0; i < subsched->numThreads_; i++) {
      UnifiedSchedulerWorker* worker = scinew UnifiedSchedulerWorker(subsched, i);
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
UnifiedScheduler::runTask( DetailedTask*         task,
                           int                   iteration,
                           int                   thread_id /* = 0 */,
                           Task::CallBackEvent   event )
{
  TAU_PROFILE("UnifiedScheduler::runTask()", " ", TAU_USER);

  if (waitout.active()) {
    waittimesLock.lock();
    {
      waittimes[task->getTask()->getName()] += Unified_CurrentWaitTime;
      Unified_CurrentWaitTime = 0;
    }
    waittimesLock.unlock();
  }

  double taskstart = Time::currentSeconds();

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }

  std::vector<DataWarehouseP> plain_old_dws(dws.size());
  for (int i = 0; i < (int)dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
  }

  task->doit(d_myworld, dws, plain_old_dws, event);

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  double dtask = Time::currentSeconds() - taskstart;

  dlbLock.lock();
  {
    if (execout.active()) {
      exectimes[task->getTask()->getName()] += dtask;
    }

    // If I do not have a sub scheduler
    if (!task->getTask()->getHasSubScheduler()) {
      //add my task time to the total time
      mpi_info_.totaltask += dtask;
      if (!d_sharedState->isCopyDataTimestep() && task->getTask()->getType() != Task::Output) {
        getLoadBalancer()->addContribution(task, dtask);
      }
    }
  }
  dlbLock.unlock();

  // For CPU and postGPU task runs, post MPI sends and call task->done;
  if (event == Task::CPU || event == Task::postGPU) {
    if (Uintah::Parallel::usingMPI()) {
      postMPISends(task, iteration, thread_id);
    }
    task->done(dws);  // should this be timed with taskstart? - BJW
    double teststart = Time::currentSeconds();

    if (Uintah::Parallel::usingMPI()) {
      // This is per thread, no lock needed.
      sends_[thread_id].testsome(d_myworld);
    }

    mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;

    // add my timings to the parent scheduler
    if( parentScheduler_ ) {
      parentScheduler_->mpi_info_.totaltask += mpi_info_.totaltask;
      parentScheduler_->mpi_info_.totaltestmpi += mpi_info_.totaltestmpi;
      parentScheduler_->mpi_info_.totalrecv += mpi_info_.totalrecv;
      parentScheduler_->mpi_info_.totalsend += mpi_info_.totalsend;
      parentScheduler_->mpi_info_.totalwaitmpi += mpi_info_.totalwaitmpi;
      parentScheduler_->mpi_info_.totalreduce += mpi_info_.totalreduce;
    }
  }
}  // end runTask()

//______________________________________________________________________
//

void
UnifiedScheduler::execute( int tgnum     /* = 0 */,
                           int iteration /* = 0 */ )
{
  // copy data and restart timesteps must be single threaded for now
  bool isMPICopyDataTS = Uintah::Parallel::usingMPI() && d_sharedState->isCopyDataTimestep();
  bool isRestartTS = d_isInitTimestep || d_isRestartInitTimestep;
  if (isMPICopyDataTS || isRestartTS) {
    MPIScheduler::execute( tgnum, iteration );
    return;
  }

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::execute");

  TAU_PROFILE("UnifiedScheduler::execute()", " ", TAU_USER);
  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[UnifiedScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[UnifiedScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[UnifiedScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[UnifiedScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(testsometimer, "Test Some", "[UnifiedScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[UnifiedScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[UnifiedScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[UnifiedScheduler::execute()] ", TAU_USER);

  ASSERTRANGE(tgnum, 0, (int )graphs.size());
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  dts = tg->getDetailedTasks();

  if (dts == 0) {
    proc0cout << "UnifiedScheduler skipping execute, no tasks\n";
    return;
  }

  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  ntasks = dts->numLocalTasks();
  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  if (unified_timeout.active()) {
    d_labels.clear();
    d_times.clear();
  }

  int me = d_myworld->myrank();
  makeTaskGraphDoc(dts, me);

  // TODO - figure out and fix this (APH - 01/12/15)
//  if (timeout.active()) {
//    emitTime("taskGraph output");
//  }

  mpi_info_.totalreduce = 0;
  mpi_info_.totalsend = 0;
  mpi_info_.totalrecv = 0;
  mpi_info_.totaltask = 0;
  mpi_info_.totalreducempi = 0;
  mpi_info_.totalsendmpi = 0;
  mpi_info_.totalrecvmpi = 0;
  mpi_info_.totaltestmpi = 0;
  mpi_info_.totalwaitmpi = 0;

  numTasksDone = 0;
  abort = false;
  abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  TAU_PROFILE_TIMER(doittimer, "Task execution", "[UnifiedScheduler::execute() loop] ", TAU_USER);TAU_PROFILE_START(doittimer);

  currentIteration = iteration;
  currphase = 0;
  numPhases = tg->getNumTaskPhases();
  phaseTasks.clear();
  phaseTasks.resize(numPhases, 0);
  phaseTasksDone.clear();
  phaseTasksDone.resize(numPhases, 0);
  phaseSyncTask.clear();
  phaseSyncTask.resize(numPhases, NULL);
  dts->setTaskPriorityAlg(taskQueueAlg_);

  // get the number of tasks in each task phase
  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }

  if (unified_dbg.active()) {
    cerrLock.lock();
    {
      unified_dbg << "\n"
                  << "Rank-" << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)\n"
                  << "Total task phases: " << numPhases
                  << "\n";
      for (size_t phase = 0; phase < phaseTasks.size(); ++phase) {
        unified_dbg << "Phase: " << phase << " has " << phaseTasks[phase] << " total tasks\n";
      }
      unified_dbg << std::endl;
    }
    cerrLock.unlock();
  }

  static int totaltasks;

  if (taskdbg.active()) {
    cerrLock.lock();
    taskdbg << "Rank-" << me << " starting task phase " << currphase << ", total phase " << currphase << " tasks = "
            << phaseTasks[currphase] << std::endl;
    cerrLock.unlock();
  }

  // signal worker threads to begin executing tasks
  for (int i = 0; i < numThreads_; i++) {
    t_worker[i]->resetWaittime(Time::currentSeconds());  // reset wait time counter
    // sending signal to threads to wake them up
    t_worker[i]->d_runmutex.lock();
    t_worker[i]->d_idle = false;
    t_worker[i]->d_runsignal.conditionSignal();
    t_worker[i]->d_runmutex.unlock();
  }

  // main thread also executes tasks
  runTasks(Thread::self()->myid());

  TAU_PROFILE_STOP(doittimer);

  // wait for all tasks to finish
  d_nextmutex.lock();
  while (getAvailableThreadNum() < numThreads_) {
    // if any thread is busy, conditional wait here
    d_nextsignal.wait(d_nextmutex);
  }
  d_nextmutex.unlock();

  if (unified_queuelength.active()) {
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

  emitTime("MPI Send time", mpi_info_.totalsendmpi);
  emitTime("MPI Recv time", mpi_info_.totalrecvmpi);
  emitTime("MPI TestSome time", mpi_info_.totaltestmpi);
  emitTime("MPI Wait time", mpi_info_.totalwaitmpi);
  emitTime("MPI reduce time", mpi_info_.totalreducempi);
  emitTime("Total send time", mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
  emitTime("Total recv time", mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
  emitTime("Total task time", mpi_info_.totaltask);
  emitTime("Total reduction time", mpi_info_.totalreduce - mpi_info_.totalreducempi);
  emitTime("Total comm time", mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

  double time = Time::currentSeconds();
  double totalexec = time - d_lasttime;
  d_lasttime = time;

  emitTime("Other excution time",
           totalexec - mpi_info_.totalsend - mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);

  if (d_sharedState != 0) {
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime;  // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
    for (int i = 0; i < numThreads_; i++) {
      d_sharedState->taskWaitThreadTime += t_worker[i]->getWaittime();
    }
  }

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

//#ifdef HAVE_CUDA
//  // clear all registered, page-locked host memory
//  unregisterPageLockedHostMem();
//#endif

  log.finishTimestep();

  if( unified_timeout.active() && !parentScheduler_ ) {  // only do on toplevel scheduler
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

    // now collect timing info
    emitTime("NumPatches", myPatches->size());
    emitTime("NumCells", numCells);
    emitTime("NumParticles", numParticles);
    std::vector<double> d_totaltimes(d_times.size());
    std::vector<double> d_maxtimes(d_times.size());
    std::vector<double> d_avgtimes(d_times.size());
    double avgTask = -1, maxTask = -1;
    double avgComm = -1, maxComm = -1;
    double avgCell = -1, maxCell = -1;

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
      maxTotal += d_maxtimes[i];
      avgTotal += d_avgtimes[i];
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
        out << "UnifiedScheduler: " << d_labels[i] << ": ";
        int len = static_cast<int>((strlen(d_labels[i])) + strlen("UnifiedScheduler: ") + strlen(": "));
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
      unified_timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1 - avgTask / maxTask) * 100 << " load imbalance (exec)%\n";
      unified_timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1 - avgComm / maxComm) * 100 << " load imbalance (comm)%\n";
      unified_timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1 - avgCell / maxCell) * 100 << " load imbalance (theoretical)%\n\n";
    }

    // TODO - need to clean this up (APH - 01/22/15)
    double time = Time::currentSeconds();
//    double rtime = time - d_lasttime;
    d_lasttime = time;
//    unified_timeout << "UnifiedScheduler: TOTAL                                    " << total << '\n';
//    unified_timeout << "UnifiedScheduler: time sum reduction (one processor only): " << rtime << '\n';
  } // end unified_timeout.active()

  if (execout.active()) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {
      std::ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);

      for (std::map<std::string, double>::iterator iter = exectimes.begin(); iter != exectimes.end(); iter++) {
        fout << std::fixed << d_myworld->myrank() << ": TaskExecTime(s): " << iter->second << " Task:" << iter->first << std::endl;
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
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second << " Task:" << iter->first << "\n";
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

  if (unified_dbg.active()) {
    unified_dbg << "Rank-" << me << " - UnifiedScheduler finished" << std::endl;
  }
} // end execute()

//______________________________________________________________________
//

void
UnifiedScheduler::runTasks( int thread_id )
{
  int me = d_myworld->myrank();

  while( numTasksDone < ntasks ) {

    DetailedTask* readyTask = NULL;
    DetailedTask* initTask = NULL;

    int pendingMPIMsgs = 0;
    bool havework = false;

#ifdef HAVE_CUDA
    bool gpuInitReady = false;
    bool gpuRunReady = false;
    bool gpuPending = false;
#endif

    // ----------------------------------------------------------------------------------
    // Part 1:
    //    Check if anything this thread can do concurrently.
    //    If so, then update the various scheduler counters.
    // ----------------------------------------------------------------------------------
    schedulerLock.lock();
    while (!havework) {
      /*
       * (1.1)
       *
       * If it is time to setup for a reduction task, then do so.
       *
       */
      if ((phaseSyncTask[currphase] != NULL) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {
        readyTask = phaseSyncTask[currphase];
        havework = true;
        numTasksDone++;
        if (taskorder.active()) {
          if (me == d_myworld->size() / 2) {
            cerrLock.lock();
            taskorder << "Rank-" << me  << " Running task static order: " << readyTask->getSaticOrder()
                      << " , scheduled order: " << numTasksDone << std::endl;
            cerrLock.unlock();
          }
        }
        phaseTasksDone[readyTask->getTask()->d_phase]++;
        while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase + 1 < numPhases) {
          currphase++;
          if (taskdbg.active()) {
            cerrLock.lock();
            taskdbg << "Rank-" << me << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                    << phaseTasks[currphase] << std::endl;
            cerrLock.unlock();
          }
        }
        break;
      }

      /*
       * (1.2)
       *
       * Run a CPU task that has its MPI communication complete. These tasks get in the external
       * ready queue automatically when their receive count hits 0 in DependencyBatch::received,
       * which is called when a MPI message is delivered.
       *
       * NOTE: This is also where a GPU-enabled task gets into the GPU initially-ready queue
       *
       */
      else if (dts->numExternalReadyTasks() > 0) {
        readyTask = dts->getNextExternalReadyTask();
        if (readyTask != NULL) {
          havework = true;
#ifdef HAVE_CUDA
          /*
           * (1.2.1)
           *
           * If it's a GPU-enabled task, assign it to a device (round robin fashion for now)
           * and initiate its H2D computes and requires data copies. This is where the
           * execution cycle begins for each GPU-enabled Task.
           *
           * gpuInitReady = true
           */
          if (readyTask->getTask()->usesDevice()) {
            readyTask->assignDevice(currentDevice_);
            currentDevice_++;
            currentDevice_ %= numDevices_;
            gpuInitReady = true;
          }
          else {
#endif
          numTasksDone++;
          if (taskorder.active()) {
            if (d_myworld->myrank() == d_myworld->size() / 2) {
              cerrLock.lock();
              taskorder << "Rank-" << me << " Running task static order: " << readyTask->getSaticOrder()
                        << ", scheduled order: " << numTasksDone << std::endl;
              cerrLock.unlock();
            }
          }
          phaseTasksDone[readyTask->getTask()->d_phase]++;
          while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase + 1 < numPhases) {
            currphase++;
            if (taskdbg.active()) {
              cerrLock.lock();
              taskdbg << "Rank-" << me << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                      << phaseTasks[currphase] << std::endl;
              cerrLock.unlock();
            }
          }
#ifdef HAVE_CUDA
        }
#endif
          break;
        }
      }

      /*
       * (1.3)
       *
       * If we have an internally-ready CPU task, initiate its MPI receives, preparing it for
       * CPU external ready queue. The task is moved to the CPU external-ready queue in the
       * call to task->checkExternalDepCount().
       *
       */
      else if (dts->numInternalReadyTasks() > 0) {
        initTask = dts->getNextInternalReadyTask();
        if (initTask != NULL) {
          if (initTask->getTask()->getType() == Task::Reduction || initTask->getTask()->usesMPI()) {
            if (taskdbg.active()) {
              cerrLock.lock();
              taskdbg << "Rank-" << me << " Task internal ready 1 " << *initTask << std::endl;
              cerrLock.unlock();
            }
            phaseSyncTask[initTask->getTask()->d_phase] = initTask;
            ASSERT(initTask->getRequires().size() == 0)
            initTask = NULL;
          }
          else if (initTask->getRequires().size() == 0) {  // no ext. dependencies, then skip MPI sends
            initTask->markInitiated();
            initTask->checkExternalDepCount();  // where tasks get added to external ready queue
            initTask = NULL;
          }
          else {
            havework = true;
            break;
          }
        }
      }
#ifdef HAVE_CUDA
      /*
       * (1.4)
       *
       * Check if highest priority GPU task's asynchronous H2D copies are completed. If so,
       * then reclaim the streams and events it used for these operations, execute the task and
       * then put it into the GPU completion-pending queue.
       *
       * gpuRunReady = true
       */
      else if (dts->numInitiallyReadyDeviceTasks() > 0) {
        readyTask = dts->peekNextInitiallyReadyDeviceTask();
        if (readyTask->queryCUDAStreamCompletion()) {
          // All of this task's h2d copies are complete, so add it to the completion
          // pending GPU task queue and prepare to run.
          readyTask = dts->getNextInitiallyReadyDeviceTask();
          gpuRunReady = true;
          havework = true;
          break;
        }
      }

      /*
       * (1.5)
       *
       * Check to see if any GPU tasks have their D2H copies completed. This means the kernel(s)
       * have executed and all the results are back on the host in the DataWarehouse. This task's
       * MPI sends can then be posted and done() can be called.
       *
       * gpuPending = true
       */
      else if (dts->numCompletionPendingDeviceTasks() > 0) {
        readyTask = dts->peekNextCompletionPendingDeviceTask();
        if (readyTask->queryCUDAStreamCompletion()) {
          readyTask = dts->getNextCompletionPendingDeviceTask();
          havework = true;
          gpuPending = true;
          numTasksDone++;
          if (taskorder.active()) {
            if (d_myworld->myrank() == d_myworld->size() / 2) {
              cerrLock.lock();
              taskorder << "Rank-" << me << " Running task static order: " << readyTask->getSaticOrder()
                        << " , scheduled order: " << numTasksDone << std::endl;
              cerrLock.unlock();
            }
          }
          phaseTasksDone[readyTask->getTask()->d_phase]++;
          while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase + 1 < numPhases) {
            currphase++;
            if (taskdbg.active()) {
              cerrLock.lock();
              taskdbg << "Rank-" << me << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                      << phaseTasks[currphase] << std::endl;
              cerrLock.unlock();
            }
          }
          break;
        }
      }
#endif
      /*
       * (1.6)
       *
       * Otherwise there's nothing to do but process MPI recvs.
       */
      else {
        pendingMPIMsgs = pendingMPIRecvs();
        if (pendingMPIMsgs > 0) {
          havework = true;
          break;
        }
      }
      if (numTasksDone == ntasks) {
        break;
      }
    } // end while (!havework)

    schedulerLock.unlock();

    // ----------------------------------------------------------------------------------
    // Part 2
    //    Concurrent Part:
    //      Each thread does its own thing here... modify this code with caution
    // ----------------------------------------------------------------------------------

    if (initTask != NULL) {
      initiateTask(initTask, abort, abort_point, currentIteration);
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Rank-" << me << " Task internal ready 2 " << *initTask << " deps needed: "
                << initTask->getExternalDepCount() << std::endl;
        cerrLock.unlock();
      }
      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask != NULL) {
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Rank-" << me << " Task external ready " << *readyTask << std::endl;
        cerrLock.unlock();
      }
      if (readyTask->getTask()->getType() == Task::Reduction) {
        initiateReduction(readyTask);
      }
#ifdef HAVE_CUDA
      else if (gpuInitReady) {
        // initiate all asynchronous H2D memory copies for this task's requires
        readyTask->setCUDAStream(getCudaStream(readyTask->getDeviceNum()));
        postH2DCopies(readyTask);
        preallocateDeviceMemory(readyTask);
        for (int i = 0; i < (int)dws.size(); i++) {
          dws[i]->getGPUDW(readyTask->getDeviceNum())->syncto_device();
        }
        dts->addInitiallyReadyDeviceTask(readyTask);
      }
      else if (gpuRunReady) {
        runTask(readyTask, currentIteration, thread_id, Task::GPU);
        postD2HCopies(readyTask);
        dts->addCompletionPendingDeviceTask(readyTask);
      }
      else if (gpuPending) {
        // run post GPU part of task 
        runTask(readyTask, currentIteration, thread_id, Task::postGPU);
        // recycle this task's stream
        reclaimCudaStreams(readyTask);
      }
#endif
      else {
        runTask(readyTask, currentIteration, thread_id, Task::CPU);
        printTaskLevels(d_myworld, taskLevel_dbg, readyTask);
      }
    }
    else if (pendingMPIMsgs > 0) {
      processMPIRecvs(TEST);
    }
    else {
      // This can only happen when all tasks have finished.
      ASSERT(numTasksDone == ntasks);
    }
  }  //end while (numTasksDone < ntasks)
}

//______________________________________________________________________
//

struct CompareDep {

  bool operator()( DependencyBatch* a,
                   DependencyBatch* b )
  {
    return a->messageTag < b->messageTag;
  }
};

//______________________________________________________________________
//

int
UnifiedScheduler::pendingMPIRecvs()
{
  int num = 0;
  recvLock.readLock();
  num = recvs_.numRequests();
  recvLock.readUnlock();
  return num;
}

//______________________________________________________________________
//

int UnifiedScheduler::getAvailableThreadNum()
{
  int num = 0;
  for (int i = 0; i < numThreads_; i++) {
    if (t_worker[i]->d_idle) {
      num++;
    }
  }
  return num;
}

#ifdef HAVE_CUDA
//______________________________________________________________________
//

void
UnifiedScheduler::gpuInitialize( bool reset )
{
  cudaError_t retVal;

  if (use_single_device.active()) {
    numDevices_ = 1;
  } else {
    CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&numDevices_));
  }

  if (reset){
    for (int i=0; i< numDevices_ ; i++) {
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(i));
      CUDA_RT_SAFE_CALL(retVal = cudaDeviceReset());
    }
  }
  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));
  currentDevice_ = 0;
}

//______________________________________________________________________
//

void
UnifiedScheduler::postH2DCopies( DetailedTask* dtask ) {

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::postH2DCopies");
  TAU_PROFILE("UnifiedScheduler::postH2DCopies()", " ", TAU_USER);

  // set the device and CUDA context
  cudaError_t retVal;
  int device = dtask->getDeviceNum();
  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(device));
  const Task* task = dtask->getTask();

  // determine variables the specified task requires
  for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->next) {
    constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());

    int dwIndex = req->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];

    int maxLevels = dw->getGrid()->numLevels();
    LevelP level_n = dw->getGrid()->getLevel(maxLevels - 1);
    LevelP level_0 = dw->getGrid()->getLevel(0);
    const Level* fineLevel = getLevel(dtask->getPatches());
    int fineLevelIdx = level_n->getIndex();
    bool isLevelItem = false;

    void* host_ptr = NULL;    // host base pointer to raw data
    void* device_ptr = NULL;  // device base pointer to raw data
    size_t host_bytes = 0;    // raw byte count to copy to the device
    size_t device_bytes = 0;  // raw byte count to copy to the host
    IntVector host_low, host_high, host_offset, host_size, host_strides;

    int numPatches = patches->size();
    int numMatls = matls->size();
    for (int i = 0; i < numPatches; ++i) {
      for (int j = 0; j < numMatls; ++j) {

        int matlID  = matls->get(j);
        int patchID = patches->get(i)->getID();

        // multi-level check
        int levelID = fineLevelIdx;
        const Level* level = fineLevel;
        if (fineLevelIdx > 0) {
          if (req->patches_dom == Task::CoarseLevel) {
            levelID = fineLevelIdx - req->level_offset;
            ASSERT(levelID >= 0);
            level = dw->getGrid()->getLevel(levelID).get_rep();
            isLevelItem = true;
          }
        }

        const std::string reqVarName = req->var->getName();

        TypeDescription::Type type = req->var->typeDescription()->getType();
        switch (type) {
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable : {

            GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(req->var->typeDescription()->createInstance());

            // check for case when INF ghost cells are requested such as in RMCRT
            //   in this case we need to use getRegion()
            bool uses_SHRT_MAX = (req->numGhostCells == SHRT_MAX);
            if (uses_SHRT_MAX) {
              IntVector domainLo_EC, domainHi_EC;
              level->findCellIndexRange(domainLo_EC, domainHi_EC);  // including extraCells
              dw->getRegion(*gridVar, req->var, matls->get(j), level, domainLo_EC, domainHi_EC, true);
              gridVar->getSizes(domainLo_EC, domainHi_EC, host_offset, host_size, host_strides);
              host_ptr = gridVar->getBasePointer();
              host_bytes = gridVar->getDataSize();
            } else {
              dw->getGridVar(*gridVar, req->var, matlID, patches->get(i), req->gtype, req->numGhostCells);
              gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);
              host_ptr = gridVar->getBasePointer();
              host_bytes = gridVar->getDataSize();
            }

            // check if the variable already exists on the GPU
            if (dw->getGPUDW()->exist(reqVarName.c_str(), patchID, matlID)) {

              int3 device_offset;
              int3 device_size;

              /*
               * Until better type information support is implemented for GPUGridVariables
               *   we need to determine the size of a single element in the Arary3Data object to
               *   know what type of GPUGridVariable to create and use.
               *
               *   "host_strides.x()" == sizeof(T)
               *
               *   This approach currently supports:
               *   ------------------------------------------------------------
               *   GPUGridVariable<int>
               *   GPUGridVariable<double>
               *   GPUGridVariable<GPUStencil7>
               *   ------------------------------------------------------------
               */

              switch (host_strides.x()) {
                case sizeof(int) : {
                  GPUGridVariable<int> device_var;
                  dw->getGPUDW()->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                case sizeof(double) : {
                  GPUGridVariable<double> device_var;
                  dw->getGPUDW()->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                case sizeof(GPUStencil7) : {
                  GPUGridVariable<GPUStencil7> device_var;
                  dw->getGPUDW()->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                default : {
                  SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + reqVarName, __FILE__, __LINE__));
                }
              }

//---------------------------------------------------------------------------------------------------
// TODO - fix this logic (APH 10/5/14)
//        Need to handle Wasatch case where a field exists on the GPU, but requires a ghost update
//---------------------------------------------------------------------------------------------------
//              // if the extents and offsets are the same, then the variable already exists on the GPU... no H2D copy
//              if (device_offset.x == host_offset.x() && device_offset.y == host_offset.y() && device_offset.z == host_offset.z()
//                  && device_size.x == host_size.x() && device_size.y == host_size.y() && device_size.z == host_size.z()) {
//                // report the above fact
//                if (gpu_stats.active()) {
//                  cerrLock.lock();
//                  {
//                    gpu_stats << "GridVariable (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
//                  }
//                  cerrLock.unlock();
//                }
//                continue;
//              } else {
                dw->getGPUDW()->remove(reqVarName.c_str(), patchID, matlID, levelID);
//              }
            }

            // Otherwise, variable doesn't exist on the GPU, so prepare and async copy to the device
            h2dRequiresLock_.writeLock();
            {
              IntVector low = host_offset;
              IntVector high = host_offset + host_size;
              int3 device_low = make_int3(low.x(), low.y(), low.z());
              int3 device_hi = make_int3(high.x(), high.y(), high.z());

              switch (host_strides.x()) {
                case sizeof(int) : {
                  GPUGridVariable<int> device_var;
                  if (isLevelItem) {
                    dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), device_low, device_hi, levelID);
                  }
                  else {
                    dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, levelID);
                  }
                  device_ptr = device_var.getPointer();
                  break;
                }
                case sizeof(double) : {
                  GPUGridVariable<double> device_var;
                  if (isLevelItem) {
                    dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), device_low, device_hi, levelID);
                  }
                  else {
                    dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, levelID);
                  }
                  device_ptr = device_var.getPointer();
                  break;
                }
                case sizeof(GPUStencil7) : {
                  GPUGridVariable<GPUStencil7> device_var;
                  if (isLevelItem) {
                    dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), device_low, device_hi, levelID);
                  }
                  else {
                    dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, levelID);
                  }
                  device_ptr = device_var.getPointer();
                  break;
                }
                default : {
                  SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + reqVarName, __FILE__, __LINE__));
                }
              }

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << "Post H2D copy of REQUIRES (" << reqVarName <<  "), size (bytes) = "  << std::dec << host_bytes
                            << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
                            << ", using stream " << std::hex << dtask->getCUDAStream() << std::dec << std::endl;
                }
                cerrLock.unlock();
              }
              cudaStream_t stream = *(dtask->getCUDAStream());
              CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(device_ptr, host_ptr, host_bytes, cudaMemcpyHostToDevice, stream));
            }
            h2dRequiresLock_.writeUnlock();
            delete gridVar;
            break;

          }  // end GridVariable switch case

          case TypeDescription::ReductionVariable : {

            levelID = -1;
            ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(req->var->typeDescription()->createInstance());
            host_ptr = reductionVar->getBasePointer();
            host_bytes = reductionVar->getDataSize();
            GPUReductionVariable<void*> device_var;

            // check if the variable already exists on the GPU
            if (dw->getGPUDW()->exist(reqVarName.c_str(), patchID, matlID, levelID)) {
              dw->getGPUDW()->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              device_bytes = device_var.getMemSize(); // TODO fix this
              // if the size is the same, assume the variable already exists on the GPU... no H2D copy
              if (host_bytes == device_bytes) {
                // report the above fact
                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << "ReductionVariable (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
                  }
                  cerrLock.unlock();
                }
                continue;
              } else {
                dw->getGPUDW()->remove(reqVarName.c_str(), patchID, matlID, levelID);
              }
            }

            // critical section - prepare and async copy the requires variable data to the device
            h2dRequiresLock_.writeLock();
            {
              dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, levelID);

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << "Post H2D copy of REQUIRES (" << reqVarName <<  "), size (bytes) = "  << std::dec << host_bytes
                            << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
                            << ", using stream " << std::hex << dtask->getCUDAStream() << std::dec << std::endl;
                }
                cerrLock.unlock();
              }
              cudaStream_t stream = *(dtask->getCUDAStream());
              CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(device_var.getPointer(), host_ptr, host_bytes, cudaMemcpyHostToDevice, stream));
            }
            h2dRequiresLock_.writeUnlock();
            delete reductionVar;
            break;

          }  // end ReductionVariable switch case

          case TypeDescription::PerPatch : {

            PerPatchBase* perPatchVar = dynamic_cast<PerPatchBase*>(req->var->typeDescription()->createInstance());
            host_ptr = perPatchVar->getBasePointer();
            host_bytes = perPatchVar->getDataSize();
            GPUPerPatch<void*> device_var;

            // check if the variable already exists on the GPU
            if (dw->getGPUDW()->exist(reqVarName.c_str(), patchID, matlID, levelID)) {
              dw->getGPUDW()->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              device_bytes = device_var.getMemSize(); // TODO fix this
              // if the size is the same, assume the variable already exists on the GPU... no H2D copy
              if (host_bytes == device_bytes) {
                // report the above fact
                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << "PerPatch (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
                  }
                  cerrLock.unlock();
                }
                continue;
              } else {
                dw->getGPUDW()->remove(reqVarName.c_str(), patchID, matlID, levelID);
              }
            }

            // critical section - prepare and async copy the requires variable data to the device
            h2dRequiresLock_.writeLock();
            {
              dw->getGPUDW()->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, levelID);

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << "Post H2D copy of REQUIRES (" << reqVarName <<  "), size (bytes) = "  << std::dec << host_bytes
                            << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
                            << ", using stream " << std::hex << dtask->getCUDAStream() << std::dec << std::endl;
                }
                cerrLock.unlock();
              }
              cudaStream_t stream = *(dtask->getCUDAStream());
              CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(device_var.getPointer(), host_ptr, host_bytes, cudaMemcpyHostToDevice, stream));
            }
            h2dRequiresLock_.writeUnlock();
            delete perPatchVar;
            break;

          }  // end PerPatch switch case

          case TypeDescription::ParticleVariable : {
            SCI_THROW(InternalError("Copying ParticleVariables to GPU not yet supported:" + reqVarName, __FILE__, __LINE__));
            break;
          }  // end ParticleVariable switch case

          default : {
            SCI_THROW(InternalError("Cannot copy unsupported variable types to GPU:" + reqVarName, __FILE__, __LINE__));

          }  // end default switch case

        }  // end switch
      }  // end matl loop
    }  // end patch loop
  }  // end requires gathering loop

}

//______________________________________________________________________
//

void
UnifiedScheduler::preallocateDeviceMemory( DetailedTask* dtask )
{
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::preallocateDeviceMemory");
  TAU_PROFILE("UnifiedScheduler::preallocateDeviceMemory()", " ", TAU_USER);

  // NOTE: the device and CUDA context are set in the call: dw->getGPUDW()->allocateAndPut()

  // determine variables the specified task will compute
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(dtask->getMaterials());

    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];

    void* device_ptr = NULL;  // device base pointer to raw data
    size_t num_bytes = 0;

    int numPatches = patches->size();
    int numMatls = matls->size();
    for (int i = 0; i < numPatches; ++i) {
      for (int j = 0; j < numMatls; ++j) {
        int matlID = matls->get(j);
        int patchID = patches->get(i)->getID();
        const Level* level = getLevel(dtask->getPatches());
        int levelID = level->getID();

        const std::string compVarName = comp->var->getName();

        TypeDescription::Type type = comp->var->typeDescription()->getType();
        switch (type) {
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable : {

            IntVector low, high, lowOffset, highOffset;
            Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
            Patch::getGhostOffsets(type, comp->gtype, comp->numGhostCells, lowOffset, highOffset);
            patches->get(i)->computeExtents(basis, comp->var->getBoundaryLayer(), lowOffset, highOffset, low, high);

            d2hComputesLock_.writeLock();
            {
              /*
               * Until better type information support is implemented for GPUGridVariables
               *   we need to use:
               *
               *   std::string name = comp->var->typeDescription()->getSubType()->getName()
               *
               *   to determine what sub-type the computes variables consist of in order to
               *   create the correct type of GPUGridVariable. Can't use info from getSizes()
               *   as the variable doesn't exist (e.g. allocateAndPut() hasn't been called yet)
               *
               *   This approach currently supports:
               *   ------------------------------------------------------------
               *   GPUGridVariable<int>
               *   GPUGridVariable<double>
               *   GPUGridVariable<GPUStencil7>
               *   ------------------------------------------------------------
               */

              std::string name = comp->var->typeDescription()->getSubType()->getName();
              if (name.compare("int") == 0) {
                GPUGridVariable<int> device_var;
                dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID,
                                               make_int3(low.x(), low.y(), low.z()),
                                               make_int3(high.x(), high.y(), high.z()), levelID);
                device_ptr = device_var.getPointer();
                num_bytes = device_var.getMemSize();
              } else if (name.compare("double") == 0) {
                GPUGridVariable<double> device_var;
                dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID,
                                               make_int3(low.x(), low.y(), low.z()),
                                               make_int3(high.x(), high.y(), high.z()), levelID);
                device_ptr = device_var.getPointer();
                num_bytes = device_var.getMemSize();
              } else if (name.compare("Stencil7") == 0) {
                GPUGridVariable<GPUStencil7> device_var;
                dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID,
                                               make_int3(low.x(), low.y(), low.z()),
                                               make_int3(high.x(), high.y(), high.z()), levelID);
                device_ptr = device_var.getPointer();
                num_bytes = device_var.getMemSize();
              } else {
                SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + compVarName, __FILE__, __LINE__));
              }

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << "Allocated device memory for COMPUTES (" << compVarName << "), size = " << std::dec << num_bytes
                            << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
                            << std::dec << std::endl;
                }
                cerrLock.unlock();
              }
            }
            d2hComputesLock_.writeUnlock();
            break;

          }  // end GridVariable switch case

          case TypeDescription::ReductionVariable : {

            d2hComputesLock_.writeLock();
            {
              levelID = -1;
              GPUReductionVariable<void*> device_var;
              dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              num_bytes = device_var.getMemSize();
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << "Allocated device memory for COMPUTES (" << compVarName << "), size = " << std::dec << num_bytes
                            << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
                            << std::dec << std::endl;
                }
                cerrLock.unlock();
              }
            }
            d2hComputesLock_.writeUnlock();
            break;

          }  // end ReductionVariable switch case

          case TypeDescription::PerPatch : {

                      d2hComputesLock_.writeLock();
                      {
                        GPUPerPatch<void*> device_var;
                        dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID, levelID);
                        device_ptr = device_var.getPointer();
                        num_bytes = device_var.getMemSize();
                        if (gpu_stats.active()) {
                          cerrLock.lock();
                          {
                            gpu_stats << "Allocated device memory for COMPUTES (" << compVarName << "), size = " << std::dec << num_bytes
                                      << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
                                      << std::dec << std::endl;
                          }
                          cerrLock.unlock();
                        }
                      }
                      d2hComputesLock_.writeUnlock();
                      break;

                    }  // end PerPatch switch case

          case TypeDescription::ParticleVariable : {
            SCI_THROW(
                InternalError("Allocating device memory for ParticleVariables not yet supported: " + compVarName, __FILE__, __LINE__));
            break;
          }  // end ParticleVariable switch case

          default : {
            SCI_THROW(InternalError("Cannot allocate device space for unsupported variable types: " + compVarName, __FILE__, __LINE__));
          }  // end default switch case

        }  // end switch
      }  // end matl loop
    }  // end patch loop
  }  // end computes gathering loop

}

//______________________________________________________________________
//

void
UnifiedScheduler::postD2HCopies( DetailedTask* dtask )
{
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::postD2HCopies");
  TAU_PROFILE("UnifiedScheduler::postD2HCopies()", " ", TAU_USER);

  // set the device and CUDA context
  cudaError_t retVal;
  int device = dtask->getDeviceNum();
  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(device));
  const Task* task = dtask->getTask();

  // determine which computes variables to copy back to the host
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(dtask->getMaterials());

    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];

    void* host_ptr = NULL;    // host base pointer to raw data
    void* device_ptr = NULL;  // device base pointer to raw data
    size_t host_bytes = 0;    // raw byte count to copy to the device
    size_t device_bytes = 0;  // raw byte count to copy to the host
    IntVector host_low, host_high, host_offset, host_size, host_strides;

    int numPatches = patches->size();
    int numMatls = matls->size();
    for (int i = 0; i < numPatches; ++i) {
      for (int j = 0; j < numMatls; ++j) {

        int matlID = matls->get(j);
        int patchID = patches->get(i)->getID();
        const Level* level = getLevel(dtask->getPatches());
        int levelID = level->getID();

        const std::string compVarName = comp->var->getName();

        TypeDescription::Type type = comp->var->typeDescription()->getType();
        switch (type) {
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable : {

            GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(comp->var->typeDescription()->createInstance());
            dw->allocateAndPut(*gridVar, comp->var, matlID, patches->get(i), comp->gtype, comp->numGhostCells);
            gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);
            host_ptr = gridVar->getBasePointer();
            host_bytes = gridVar->getDataSize();

            // copy the computes data back to the host
            d2hComputesLock_.writeLock();
            {
              /*
               * Until better type information support is implemented for GPUGridVariables
               *   we need to determine the size of a single element in the Arary3Data object to
               *   know what type of GPUGridVariable to create and use.
               *
               *   "host_strides.x()" == sizeof(T)
               *
               *   This approach currently supports:
               *   ------------------------------------------------------------
               *   GPUGridVariable<int>
               *   GPUGridVariable<double>
               *   GPUGridVariable<GPUStencil7>
               *   ------------------------------------------------------------
               */

              int3 device_offset;
              int3 device_size;

              switch (host_strides.x()) {
                case sizeof(int) : {
                  GPUGridVariable<int> device_var;
                  dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                case sizeof(double) : {
                  GPUGridVariable<double> device_var;
                  dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                case sizeof(GPUStencil7) : {
                  GPUGridVariable<GPUStencil7> device_var;
                  dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                default : {
                  SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + compVarName, __FILE__, __LINE__));
                }
              }

              // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
              if (device_offset.x == host_offset.x() && device_offset.y == host_offset.y() && device_offset.z == host_offset.z()
                  && device_size.x == host_size.x() && device_size.y == host_size.y() && device_size.z == host_size.z()) {

                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << "Post D2H copy of COMPUTES (" << compVarName << "), size = " << std::dec << host_bytes
                              << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
                              << ", using stream " << std::hex << dtask->getCUDAStream() << std::dec << std::endl;
                  }
                  cerrLock.unlock();
                }
                CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *dtask->getCUDAStream()));
                if (retVal == cudaErrorLaunchFailure) {
                  SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                } else {
                  CUDA_RT_SAFE_CALL(retVal);
                }
              }
            }
            d2hComputesLock_.writeUnlock();
            delete gridVar;
            break;

          }  // end GridVariable switch case

          case TypeDescription::ReductionVariable : {

            levelID = -1;
            ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(comp->var->typeDescription()->createInstance());
            dw->put(*reductionVar, comp->var, patches->get(i)->getLevel(), matlID);
            host_ptr = reductionVar->getBasePointer();
            host_bytes = reductionVar->getDataSize();

            d2hComputesLock_.writeLock();
            {
              GPUReductionVariable<void*> device_var;
              dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              device_bytes = device_var.getMemSize();

              // if size is equal to CPU DW, directly copy back to CPU var memory;
              if (host_bytes == device_bytes) {

                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << "Post D2H copy of COMPUTES (" << compVarName << "), size = " << std::dec << host_bytes
                              << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
                              << ", using stream " << std::hex << dtask->getCUDAStream() << std::dec << std::endl;
                  }
                  cerrLock.unlock();
                }
                CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *dtask->getCUDAStream()));
                if (retVal == cudaErrorLaunchFailure) {
                  SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                } else {
                  CUDA_RT_SAFE_CALL(retVal);
                }
              }
            }
            d2hComputesLock_.writeUnlock();
            delete reductionVar;
            break;

          }  // end ReductionVariable switch case

          case TypeDescription::PerPatch : {

            PerPatchBase* perPatchVar = dynamic_cast<PerPatchBase*>(comp->var->typeDescription()->createInstance());
            dw->put(*perPatchVar, comp->var, matlID, patches->get(i));

            host_ptr = perPatchVar->getBasePointer();
            host_bytes = perPatchVar->getDataSize();

            d2hComputesLock_.writeLock();
            {
              GPUPerPatch<void*> device_var;
              dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              device_bytes = device_var.getMemSize();

              // if size is equal to CPU DW, directly copy back to CPU var memory;
              if (host_bytes == device_bytes) {
                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << "Post D2H copy of COMPUTES (" << compVarName << "), size = " << std::dec << host_bytes
                              << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
                              << ", using stream " << std::hex << dtask->getCUDAStream() << std::dec << std::endl;
                  }
                  cerrLock.unlock();
                }
                CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *dtask->getCUDAStream()));
                if (retVal == cudaErrorLaunchFailure) {
                  SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                } else {
                  CUDA_RT_SAFE_CALL(retVal);
                }
              }
            }
            d2hComputesLock_.writeUnlock();
            delete perPatchVar;
            break;

          }  // end PerPatch switch case

          case TypeDescription::ParticleVariable : {
            SCI_THROW(
                InternalError("Copying ParticleVariables to GPU not yet supported:" + compVarName, __FILE__, __LINE__));
            break;
          }  // end ParticleVariable switch case

          default : {
            SCI_THROW(InternalError("Cannot copy unmatched size variable from CPU to host:" + compVarName, __FILE__, __LINE__));
          }  // end default switch case

        }  // end switch
      }  // end matl loop
    }  // end patch loop
  }  // end computes gathering loop

}

//______________________________________________________________________
//

void
UnifiedScheduler::createCudaStreams( int device,
                                     int numStreams /* = 1 */ )
{
  cudaError_t retVal;

  idleStreamsLock_.writeLock();
  {
    for (int j = 0; j < numStreams; j++) {
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(device));
      cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));
      idleStreams[device].push(stream);

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << "Created CUDA stream " << std::hex << stream << " on device "
                    << std::dec << device << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }
  idleStreamsLock_.writeUnlock();
}

//______________________________________________________________________
//

void
UnifiedScheduler::freeCudaStreams()
{
  cudaError_t retVal;

  idleStreamsLock_.writeLock();
  {
    size_t numQueues = idleStreams.size();

    if (gpu_stats.active()) {
      size_t totalStreams = 0;
      for (size_t i = 0; i < numQueues; i++) {
        totalStreams += idleStreams[i].size();
      }
      cerrLock.lock();
      {
        gpu_stats << "Deallocating " << totalStreams << " total CUDA stream(s) for " << numQueues << " device(s)"<< std::endl;
      }
      cerrLock.unlock();
    }

    for (size_t i = 0; i < numQueues; i++) {
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(i));

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << "Deallocating " << idleStreams[i].size() << " CUDA stream(s) on device " << retVal << std::endl;
        }
        cerrLock.unlock();
      }

      while (!idleStreams[i].empty()) {
        cudaStream_t* stream = idleStreams[i].front();
        idleStreams[i].pop();
        CUDA_RT_SAFE_CALL(retVal = cudaStreamDestroy(*stream));
      }
    }
  }
  idleStreamsLock_.writeUnlock();
}

//______________________________________________________________________
//

cudaStream_t *
UnifiedScheduler::getCudaStream( int device )
{
  cudaError_t retVal;
  cudaStream_t* stream;

  idleStreamsLock_.writeLock();
  {
    if (idleStreams[device].size() > 0) {
      stream = idleStreams[device].front();
      idleStreams[device].pop();
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << "Issued CUDA stream " << std::hex << stream
                    << " on device " << std::dec << device << std::endl;
          cerrLock.unlock();
        }
      }
    }
    else {  // shouldn't need any more than the queue capacity, but in case
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(device));
      // this will get put into idle stream queue and ultimately deallocated after final timestep
      stream = ((cudaStream_t*)malloc(sizeof(cudaStream_t)));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << "Needed to create 1 additional CUDA stream " << std::hex << stream
                    << " for device " << std::dec << device << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }
  idleStreamsLock_.writeUnlock();

  return stream;
}

//______________________________________________________________________
//

void
UnifiedScheduler::reclaimCudaStreams( DetailedTask* dtask )
{
  cudaStream_t* stream;
  int deviceNum;

  idleStreamsLock_.writeLock();
  {
    stream = dtask->getCUDAStream();
    deviceNum = dtask->getDeviceNum();
    idleStreams[deviceNum].push(stream);
    dtask->setCUDAStream(NULL);
  }
  idleStreamsLock_.writeUnlock();

  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << "Reclaimed CUDA stream " << std::hex << stream << " on device " << std::dec << deviceNum << std::endl;
    }
    cerrLock.unlock();
  }
}

#endif // end HAVE_CUDA


//------------------------------------------
// UnifiedSchedulerWorker Thread Methods
//------------------------------------------
UnifiedSchedulerWorker::UnifiedSchedulerWorker( UnifiedScheduler*  scheduler,
                                                int                thread_id )
  : d_scheduler( scheduler ),
    d_runsignal( "run condition" ),
    d_runmutex( "run mutex" ),
    d_quit( false ),
    d_idle( true ),
    d_thread_id( thread_id + 1),
    d_rank( scheduler->getProcessorGroup()->myrank() ),
    d_waittime( 0.0 ),
    d_waitstart( 0.0 )
{
  d_runmutex.lock();
}

//______________________________________________________________________
//

void
UnifiedSchedulerWorker::run()
{
  Thread::self()->set_myid(d_thread_id);

  // Set affinity
  if (unified_compactaffinity.active()) {
    if ( (unified_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
      cerrLock.lock();
      std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      unified_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to core " << d_thread_id << "\n";
      cerrLock.unlock();
    }
    Thread::self()->set_affinity(d_thread_id);
  }

  while( true ) {
    d_runsignal.wait(d_runmutex); // wait for main thread signal.
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds() - d_waitstart;

    if (d_quit) {
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Worker " << d_rank << "-" << d_thread_id << " quitting" << "\n";
        cerrLock.unlock();
      }
      return;
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank << "-" << d_thread_id << ": executing tasks \n";
      cerrLock.unlock();
    }

    try {
      d_scheduler->runTasks(d_thread_id);
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
      taskdbg << "Worker " << d_rank << "-" << d_thread_id << ": finished executing tasks   \n";
      cerrLock.unlock();
    }

    // Signal main thread for next group of tasks.
    d_scheduler->d_nextmutex.lock();
    d_runmutex.lock();
    d_waitstart = Time::currentSeconds();
    d_idle = true;
    d_scheduler->d_nextsignal.conditionSignal();
    d_scheduler->d_nextmutex.unlock();
  }
}

//______________________________________________________________________
//

double
UnifiedSchedulerWorker::getWaittime()
{
  return d_waittime;
}

//______________________________________________________________________
//

void
UnifiedSchedulerWorker::resetWaittime( double start )
{
  d_waitstart = start;
  d_waittime  = 0.0;
}
