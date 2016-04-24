/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/Schedulers/ThreadedMPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/Time.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>

#include <sched.h>

#define USE_PACKING

using namespace Uintah;

// sync cout/cerr so they are readable when output by multiple threads
extern std::mutex coutLock;
extern std::mutex cerrLock;

extern DebugStream taskdbg;

static DebugStream threadedmpi_dbg(             "ThreadedMPI_DBG",             false);
static DebugStream threadedmpi_timeout(         "ThreadedMPI_TimingsOut",      false);
static DebugStream threadedmpi_queuelength(     "ThreadedMPI_QueueLength",     false);
static DebugStream threadedmpi_threaddbg(       "ThreadedMPI_ThreadDBG",       false);
static DebugStream threadedmpi_compactaffinity( "ThreadedMPI_CompactAffinity", true);

ThreadedMPIScheduler::ThreadedMPIScheduler( const ProcessorGroup*       myworld,
                                            const Output*               oport,
                                                  ThreadedMPIScheduler* parentScheduler)
  : MPIScheduler(myworld, oport, parentScheduler)
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

  // Create the TaskWorkers here (pinned to cores in TaskWorker::run())
  char name[1024];
  for (int i = 0; i < numThreads_; i++) {
    TaskWorker* worker = scinew TaskWorker(this, i);
    t_worker[i] = worker;
    sprintf(name, "Computing Worker %d-%d", Parallel::getRootProcessorGroup()->myrank(), i);
    std::thread * t = new Thread(worker, name);
    t_thread[i] = t;
  }

  SchedulerCommon::problemSetup(prob_spec, state);
}

//______________________________________________________________________
//

SchedulerP
ThreadedMPIScheduler::createSubScheduler()
{
  UintahParallelPort   * lbp       = getPort( "load balancer" );
  ThreadedMPIScheduler * subsched = scinew ThreadedMPIScheduler( d_myworld, m_outPort_, this );

  subsched->attachPort( "load balancer", lbp );
  subsched->d_sharedState = d_sharedState;
  subsched->numThreads_ = Uintah::Parallel::getNumThreads() - 1;

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
  
  mpi_info_.reset( 0 );

  int numTasksDone = 0;
  bool abort = false;
  int abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  int currphase = 0;
  int numPhases = tg->getNumTaskPhases();
  std::vector<int> phaseTasks(numPhases);
  std::vector<int> phaseTasksDone(numPhases);
  std::vector<DetailedTask*> phaseSyncTask(numPhases);

  dts->setTaskPriorityAlg(taskQueueAlg_);

  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }

  static int totaltasks;
  static std::vector<int> histogram;
  std::set<DetailedTask*> pending_tasks;

  for (int i = 0; i < numThreads_; i++) {
    t_worker[i]->resetWaittime(Time::currentSeconds());
  }

  // The main task loop
  while (numTasksDone < ntasks) {

    if (phaseTasks[currphase] == phaseTasksDone[currphase]) {  // this phase done, goto next phase
      currphase++;
    }
    // if we have an internally-ready task, initiate its recvs
    else if (dts->numInternalReadyTasks() > 0) {
      DetailedTask* task = dts->getNextInternalReadyTask();
      // save the reduction task and once per proc task for later execution
      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {
        phaseSyncTask[task->getTask()->d_phase] = task;
        ASSERT(task->getRequires().size() == 0)
      }
      else {
        initiateTask(task, abort, abort_point, iteration);
        task->markInitiated();
        task->checkExternalDepCount();
      }
    }
    //if it is time to run reduction task
    else if ((phaseSyncTask[currphase] != NULL) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {
      DetailedTask* reducetask = phaseSyncTask[currphase];

      if (reducetask->getTask()->getType() == Task::Reduction) {
        if (!abort) {
          assignTask(reducetask, iteration);
        }
      }
      else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->usesMPI());
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();
        ASSERT(reducetask->getExternalDepCount() == 0)
        assignTask(reducetask, iteration);
      }
      ASSERT(reducetask->getTask()->d_phase == currphase);
      numTasksDone++;
      phaseTasksDone[reducetask->getTask()->d_phase]++;
    }

    // run a task that has its communication complete
    // tasks get in this queue automatically when their receive count hits 0
    //   in DependencyBatch::received, which is called when a message is delivered.
    else if (dts->numExternalReadyTasks() > 0) {

      DetailedTask* task = dts->getNextExternalReadyTask();
      ASSERTEQ(task->getExternalDepCount(), 0);
      assignTask(task, iteration);
      numTasksDone++;
      phaseTasksDone[task->getTask()->d_phase]++;
    }
    else { // nothing to do process MPI
      processMPIRecvs(TEST);
    }

  }  // end while( numTasksDone < ntasks )

  // wait for all tasks to finish
  d_nextmutex.lock();
  while (getAvailableThreadNum() < numThreads_) {
    // if any thread is busy, conditional wait here
    d_nextsignal.wait(d_nextmutex);
  }
  d_nextmutex.unlock();


  if (threadedmpi_timeout.active()) {
    emitTime("MPI send time", mpi_info_[TotalSendMPI]);
    emitTime("MPI Testsome time", mpi_info_[TotalTestMPI]);
    emitTime("Total send time", mpi_info_[TotalSend] - mpi_info_[TotalSendMPI] - mpi_info_[TotalTestMPI]);
    emitTime("MPI recv time", mpi_info_[TotalRecvMPI]);
    emitTime("MPI wait time", mpi_info_[TotalWaitMPI]);
    emitTime("Total recv time", mpi_info_[TotalRecv] - mpi_info_[TotalRecvMPI] - mpi_info_[TotalWaitMPI]);
    emitTime("Total task time", mpi_info_[TotalTask]);
    emitTime("Total MPI reduce time", mpi_info_[TotalReduceMPI]);
    emitTime("Total reduction time", mpi_info_[TotalReduce] - mpi_info_[TotalReduceMPI]);
    emitTime("Total comm time", mpi_info_[TotalRecv] + mpi_info_[TotalSend] + mpi_info_[TotalReduce]);

    double time = Time::currentSeconds();
    double totalexec = time - d_lasttime;

    d_lasttime = time;

    emitTime("Other excution time",
             totalexec - mpi_info_[TotalSend] - mpi_info_[TotalRecv] - mpi_info_[TotalTask] - mpi_info_[TotalReduce]);
  }
  
  // compute the net timings
  if (d_sharedState != 0) {
      
    computeNetRunTimeStats(d_sharedState->d_runTimeStats);
    
    for (int i = 0; i < numThreads_; i++) {
      d_sharedState->d_runTimeStats[SimulationState::TaskWaitThreadTime] +=
          t_worker[i]->getWaittime();
    }
  }

  //if(timeout.active())
  //emitTime("final wait");
  if (restartable && tgnum == (int)graphs.size() - 1) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size() - 1]->timestepRestarted();
    int netrestart;

    MPI::Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR, d_myworld->getComm());

    if (netrestart) {
      dws[dws.size() - 1]->restartTimestep();
      if (dws[0]) {
        dws[0]->setRestarted();
      }
    }
  }

  finalizeTimestep();

  if( threadedmpi_timeout.active() && !parentScheduler_ ) {  // only do on toplevel scheduler
    outputTimingStats("ThreadedMPIScheduler");
  }

  if (threadedmpi_dbg.active()) {
    coutLock.lock();
    threadedmpi_dbg << "Rank-" << me << " - ThreadedMPIScheduler finished" << std::endl;
    coutLock.unlock();
  }
} // end execute()

//______________________________________________________________________
//

int ThreadedMPIScheduler::getAvailableThreadNum()
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
  if (getAvailableThreadNum() == 0) {
    d_nextsignal.wait(d_nextmutex);
  }
  // find an idle thread and assign task
  int targetThread = -1;
  for (int i = 0; i < numThreads_; i++) {
    if (t_worker[i]->d_task == NULL) {
      targetThread = i;
      t_worker[i]->d_numtasks++;
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
    d_quit( false ),
    d_thread_id( thread_id ),
    d_rank( scheduler->getProcessorGroup()->myrank() ),
    d_iteration( 0 ),
    d_waittime( 0.0 ),
    d_waitstart( 0.0 ),
    d_numtasks( 0 )
{
  d_runmutex.lock();
}

TaskWorker::~TaskWorker()
{
  if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
    threadedmpi_threaddbg << "TaskWorker " << d_rank << "-" << d_thread_id << " executed "
                          << d_numtasks << " tasks"  << std::endl;
  }
}

//______________________________________________________________________
//

void
TaskWorker::run()
{
  // set Uintah thread ID, offset by 1 because main execution thread is already threadID 0
  int offsetThreadID = d_thread_id + 1;
  Thread::self()->set_myid(offsetThreadID);

  // compact affinity
  if (threadedmpi_compactaffinity.active()) {
    if ( (threadedmpi_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0) ) {
      cerrLock.lock();
      std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      threadedmpi_threaddbg << "Binding" << threadType << "TaskWorker thread ID " << offsetThreadID << " to core " << offsetThreadID << "\n";
      cerrLock.unlock();
    }
    Thread::self()->set_affinity(offsetThreadID);
  }

  while (true) {
    d_runsignal.wait(d_runmutex); // wait for main thread signal
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds() - d_waitstart;

    if (d_quit) {
      if (taskdbg.active()) {
        cerrLock.lock();
        threadedmpi_threaddbg << "TaskWorker " << d_rank << "-" << d_thread_id << " quitting\n";
        cerrLock.unlock();
      }
      return;
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      threadedmpi_threaddbg << "TaskWorker " << d_rank << "-" << d_thread_id << ": began executing task: " << *d_task << "\n";
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
      std::cerr << "TaskWorker " << d_rank << "-" << d_thread_id << ": Caught exception: " << e.message() << "\n";
      if (e.stackTrace()) {
        std::cerr << "Stack trace: " << e.stackTrace() << '\n';
      }
      cerrLock.unlock();
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      threadedmpi_threaddbg << "Worker " << d_rank << "-" << d_thread_id << ": finished executing task: " << *d_task << std::endl;
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

