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
#include <iomanip>

#define USE_PACKING

using namespace Uintah;

// sync cout/cerr so they are readable when output by multiple threads
extern SCIRun::Mutex coutLock;
extern SCIRun::Mutex cerrLock;

extern DebugStream taskdbg;
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
  static DebugStream gpu_stats(        "GPUStats",     false);
         DebugStream use_single_device("SingleDevice", false);
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
      int availableDevices;
      CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&availableDevices));
      std::cout << "   Using " << numDevices_ << "/" << availableDevices << " available GPU(s)" << std::endl;
      
      for (int device_id = 0; device_id < availableDevices; device_id++) {
        cudaDeviceProp device_prop;
        CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceProperties(&device_prop, device_id));
        printf("   GPU Device %d: \"%s\" with compute capability %d.%d\n", device_id, device_prop.name, device_prop.major, device_prop.minor);
      }
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

  if (waitout.active()) {
    waittimesLock.lock();
    {
      waittimes[task->getTask()->getName()] += Unified_CurrentWaitTime;
      Unified_CurrentWaitTime = 0;
    }
    waittimesLock.unlock();
  }

  // -------------------------< begin task execution timing >-------------------------
  double task_start_time = Time::currentSeconds();

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

  double total_task_time = Time::currentSeconds() - task_start_time;
  // -------------------------< end task execution timing >-------------------------

  dlbLock.lock();
  {
    if (execout.active()) {
      exectimes[task->getTask()->getName()] += total_task_time;
    }

    // If I do not have a sub scheduler
    if (!task->getTask()->getHasSubScheduler()) {
      //add my task time to the total time
      mpi_info_.totaltask += total_task_time;
      if (!d_sharedState->isCopyDataTimestep() && task->getTask()->getType() != Task::Output) {
        // add contribution of task execution time to load balancer
        getLoadBalancer()->addContribution(task, total_task_time);
      }
    }
  }
  dlbLock.unlock();

  // For CPU and postGPU task runs, post MPI sends and call task->done;
  if (event == Task::CPU || event == Task::postGPU) {

    if (Uintah::Parallel::usingMPI()) {
      postMPISends(task, iteration, thread_id);
    }

    task->done(dws);  // should this be part of task execution time? - APH 09/16/15

    // -------------------------< begin MPI test timing >-------------------------
    double test_start_time = Time::currentSeconds();

    if (Uintah::Parallel::usingMPI()) {
      // This is per thread, no lock needed.
      sends_[thread_id].testsome(d_myworld);
    }

    mpi_info_.totaltestmpi += Time::currentSeconds() - test_start_time;
    // -------------------------< end MPI test timing >-------------------------

    // add my timings to the parent scheduler
    if( parentScheduler_ ) {
      parentScheduler_->mpi_info_.totaltask += mpi_info_.totaltask;
      parentScheduler_->mpi_info_.totaltestmpi += mpi_info_.totaltestmpi;
      parentScheduler_->mpi_info_.totalrecv += mpi_info_.totalrecv;
      parentScheduler_->mpi_info_.totalsend += mpi_info_.totalsend;
      parentScheduler_->mpi_info_.totalwaitmpi += mpi_info_.totalwaitmpi;
      parentScheduler_->mpi_info_.totalreduce += mpi_info_.totalreduce;
      mpi_info_.totalreduce    = 0;
      mpi_info_.totalsend      = 0;
      mpi_info_.totalrecv      = 0;
      mpi_info_.totaltask      = 0;
      mpi_info_.totalreducempi = 0;
      mpi_info_.totaltestmpi   = 0;
      mpi_info_.totalwaitmpi   = 0;
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
  bool runSingleThreaded = isMPICopyDataTS || d_isRestartInitTimestep;
  if (runSingleThreaded) {
    MPIScheduler::execute( tgnum, iteration );
    return;
  }

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

  bool emit_timings = unified_timeout.active();
  if (emit_timings) {
    d_labels.clear();
    d_times.clear();
  }

//  // TODO - determine if this TG output code is even working correctly (APH - 09/16/15)
//  makeTaskGraphDoc(dts, d_myworld->myrank());
//  if (useInternalDeps() && emit_timings) {
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

  numTasksDone = 0;
  abort = false;
  abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

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
    coutLock.lock();
    {
      unified_dbg << "\n"
                  << "Rank-" << d_myworld->myrank() << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)\n"
                  << "Total task phases: " << numPhases
                  << "\n";
      for (size_t phase = 0; phase < phaseTasks.size(); ++phase) {
        unified_dbg << "Phase: " << phase << " has " << phaseTasks[phase] << " total tasks\n";
      }
      unified_dbg << std::endl;
    }
    coutLock.unlock();
  }

  static int totaltasks;

  if (taskdbg.active()) {
    coutLock.lock();
    taskdbg << myRankThread() << " starting task phase " << currphase << ", total phase " << currphase << " tasks = "
            << phaseTasks[currphase] << std::endl;
    coutLock.unlock();
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

  emitTime("Other excution time", totalexec - mpi_info_.totalsend - mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);

  if (d_sharedState != 0) {

    d_sharedState->taskExecTime       += mpi_info_.totaltask - d_sharedState->outputTime;  // don't count output time...
    d_sharedState->taskLocalCommTime  += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime   += mpi_info_.totalwaitmpi;
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
  log.finishTimestep();

  if (emit_timings && !parentScheduler_) {  // only do on toplevel scheduler
    outputTimingStats("UnifiedScheduler");
  }

  if (unified_dbg.active()) {
    unified_dbg << "Rank-" << d_myworld->myrank() << " - UnifiedScheduler finished" << std::endl;
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
            coutLock.lock();
            taskorder << myRankThread()  << " Running task static order: " << readyTask->getStaticOrder()
                      << " , scheduled order: " << numTasksDone << std::endl;
            coutLock.unlock();
          }
        }
        phaseTasksDone[readyTask->getTask()->d_phase]++;
        while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase + 1 < numPhases) {
          currphase++;
          if (taskdbg.active()) {
            coutLock.lock();
            taskdbg << myRankThread() << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                    << phaseTasks[currphase] << std::endl;
            coutLock.unlock();
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
              coutLock.lock();
              taskorder << myRankThread() << " Running task static order: " << readyTask->getStaticOrder()
                        << ", scheduled order: " << numTasksDone << std::endl;
              coutLock.unlock();
            }
          }
          phaseTasksDone[readyTask->getTask()->d_phase]++;
          while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase + 1 < numPhases) {
            currphase++;
            if (taskdbg.active()) {
              coutLock.lock();
              taskdbg << myRankThread() << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                      << phaseTasks[currphase] << std::endl;
              coutLock.unlock();
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
              coutLock.lock();
              taskdbg << myRankThread() <<  " Task internal ready 1 " << *initTask << std::endl;
              coutLock.unlock();
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
              coutLock.lock();
              taskorder << myRankThread() << " Running task static order: " << readyTask->getStaticOrder()
                        << " , scheduled order: " << numTasksDone << std::endl;
              coutLock.unlock();
            }
          }
          phaseTasksDone[readyTask->getTask()->d_phase]++;
          while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase + 1 < numPhases) {
            currphase++;
            if (taskdbg.active()) {
              coutLock.lock();
              taskdbg << myRankThread() << " switched to task phase " << currphase << ", total phase " << currphase << " tasks = "
                      << phaseTasks[currphase] << std::endl;
              coutLock.unlock();
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
        coutLock.lock();
        taskdbg << myRankThread() << " Task internal ready 2 " << *initTask << " deps needed: "
                << initTask->getExternalDepCount() << std::endl;
        coutLock.unlock();
      }
      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask != NULL) {
      if (taskdbg.active()) {
        coutLock.lock();
        taskdbg << myRankThread() << " Task external ready " << *readyTask << std::endl;
        coutLock.unlock();
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
    GPUDataWarehouse* gpuDW = dw->getGPUDW();
    
    const Level* level = getLevel(patches.get_rep());
    int levelID = level->getID();
    bool isLevelItem = (req->numGhostCells == SHRT_MAX);           // We should generalize this.
    int numPatches = (isLevelItem)? 1 : patches->size();

    void* host_ptr    = NULL;    // host base pointer to raw data
    void* device_ptr  = NULL;  // device base pointer to raw data
    size_t host_bytes = 0;    // raw byte count to copy to the device
    size_t device_bytes = 0;  // raw byte count to copy to the host
    IntVector host_low, host_high, host_offset, host_size, host_strides;
   
    const std::string reqVarName = req->var->getName();

    int numMatls = matls->size();
 
    bool matl_loop = true;
    //__________________________________
    //
    for (int i = 0; i < numPatches; ++i) {
      for (int j = 0; (j < numMatls && matl_loop); ++j) {

        int matlID  = matls->get(j);
        int patchID = patches->get(i)->getID();

        TypeDescription::Type type = req->var->typeDescription()->getType();
        switch (type) {
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable : {

            GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(req->var->typeDescription()->createInstance());
            
            h2dRequiresLock_.writeLock();
/*lock*/   {
              // logic for to avoid getting level variables multiple times
              bool alreadyCopied = ( gpuDW->existsLevelDB(reqVarName.c_str(), levelID, matlID) );

              if(isLevelItem && alreadyCopied) {
//                coutLock.lock();
//                std::cout <<  "    " << myRankThread() << " Goiing to skip this variable " << reqVarName.c_str() << " Patch: " << patchID << std::endl;
//                coutLock.unlock();
                h2dRequiresLock_.writeUnlock();
                matl_loop = false;
                continue;
              }

              if (isLevelItem) {
                IntVector domainLo_EC, domainHi_EC;
                level->findCellIndexRange(domainLo_EC, domainHi_EC);  // including extraCells

                // FIXME:  we should use getLevel() for a big speed up on large patch counts
                dw->getRegion(*gridVar, req->var, matls->get(j), level, domainLo_EC, domainHi_EC, true);
                gridVar->getSizes(domainLo_EC, domainHi_EC, host_offset, host_size, host_strides);

                host_ptr   = gridVar->getBasePointer();
                host_bytes = gridVar->getDataSize();
              } else {
                dw->getGridVar(*gridVar, req->var, matlID, patches->get(i), req->gtype, req->numGhostCells);
                gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);

                host_ptr   = gridVar->getBasePointer();
                host_bytes = gridVar->getDataSize();
              }

              // check if the variable already exists on the GPU
              if (gpuDW->exists(reqVarName.c_str(), patchID, matlID)) {

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
                    gpuDW->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
                    device_var.getArray3(device_offset, device_size, device_ptr);
                    break;
                  }
                  case sizeof(double) : {
                    GPUGridVariable<double> device_var;
                    gpuDW->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
                    device_var.getArray3(device_offset, device_size, device_ptr);
                    break;
                  }
                  case sizeof(GPUStencil7) : {
                    GPUGridVariable<GPUStencil7> device_var;
                    gpuDW->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
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
  //                  coutLock.lock();
  //                  {
  //                    gpu_stats << "GridVariable (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
  //                  }
  //                  coutLock.unlock();
  //                }
  //                continue;
  //              } else {
                  gpuDW->remove(reqVarName.c_str(), patchID, matlID, levelID);
  //              }
              }

              // Otherwise, variable doesn't exist on the GPU, so prepare and async copy to the device
              IntVector low  = host_offset;
              IntVector high = host_offset + host_size;
              int3 device_low = make_int3(low.x(), low.y(), low.z());
              int3 device_hi  = make_int3(high.x(), high.y(), high.z());

              switch (host_strides.x()) {
                case sizeof(int) : {
                  GPUGridVariable<int> device_var;
                  if (isLevelItem) {
                    gpuDW->allocateAndPut(device_var, reqVarName.c_str(), matlID, device_low, device_hi, levelID);
                  }
                  else {
                    gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, levelID);
                  }
                  device_ptr = device_var.getPointer();
                  break;
                }
                case sizeof(double) : {
                  GPUGridVariable<double> device_var;
                  if (isLevelItem) {
                    gpuDW->allocateAndPut(device_var, reqVarName.c_str(), matlID, device_low, device_hi, levelID);
                  }
                  else {
                    gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, levelID);
                  }
                  device_ptr = device_var.getPointer();
                  break;
                }
                case sizeof(GPUStencil7) : {
                  GPUGridVariable<GPUStencil7> device_var;
                  if (isLevelItem) {
                    gpuDW->allocateAndPut(device_var, reqVarName.c_str(), matlID, device_low, device_hi, levelID);
                  }
                  else {
                    gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, levelID);
                  }
                  device_ptr = device_var.getPointer();
                  break;
                }
                default : {
                  SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + reqVarName, __FILE__, __LINE__));
                }
              }

              if (gpu_stats.active()) {
                coutLock.lock();
                {
                  int3 nCells    = make_int3(device_hi.x-device_low.x, device_hi.y-device_low.y, device_hi.z-device_low.z);
                  gpu_stats << myRankThread() 
                            << " Post H2D copy of REQUIRES (" << std::setw(26) << reqVarName <<  "), L-" << levelID << ", patch: " << patchID<< ", "
                            << std::setw(10) << "Bytes: "  << std::dec << host_bytes <<", "
                            << std::setw(10) << "nCells [" << nCells.x <<","<<nCells.y <<"," << nCells.z <<"]"
                            << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
                            << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
                }
                coutLock.unlock();
              }
              cudaStream_t stream = *(dtask->getCUDAStream());
              CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(device_ptr, host_ptr, host_bytes, cudaMemcpyHostToDevice, stream));
/*lock*/    }
            h2dRequiresLock_.writeUnlock();
            delete gridVar;
            break;

          }  // end GridVariable switch case
          //__________________________________
          //
          case TypeDescription::ReductionVariable : {

            levelID = -1;
            ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(req->var->typeDescription()->createInstance());
            host_ptr   = reductionVar->getBasePointer();
            host_bytes = reductionVar->getDataSize();
            GPUReductionVariable<void*> device_var;

            // check if the variable already exists on the GPU
            if (gpuDW->exists(reqVarName.c_str(), patchID, matlID, levelID)) {
            
              gpuDW->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
              device_ptr   = device_var.getPointer();
              device_bytes = device_var.getMemSize(); // TODO fix this
            
              // if the size is the same, assume the variable already exists on the GPU... no H2D copy
              if (host_bytes == device_bytes) {
                // report the above fact
                if (gpu_stats.active()) {
                  coutLock.lock();
                  {
                    gpu_stats << myRankThread() 
                              << "ReductionVariable (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
                  }
                  coutLock.unlock();
                }
                continue;
              } else {
                gpuDW->remove(reqVarName.c_str(), patchID, matlID, levelID);
              }
            }

            // critical section - prepare and async copy the requires variable data to the device
            h2dRequiresLock_.writeLock();
            {
              gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, levelID);

              if (gpu_stats.active()) {
                coutLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " Post H2D copy of REQUIRES (" << std::setw(26) << reqVarName <<  "), L-" << levelID << ", patch: " << patchID<< ", "
                            << "Bytes = "  << std::dec << host_bytes
                            << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
                            << ", using stream " << std::hex << dtask->getCUDAStream() << std::endl;
                }
                coutLock.unlock();
              }
              cudaStream_t stream = *(dtask->getCUDAStream());
              CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(device_var.getPointer(), host_ptr, host_bytes, cudaMemcpyHostToDevice, stream));
            }
            h2dRequiresLock_.writeUnlock();
            delete reductionVar;
            break;

          }  // end ReductionVariable switch case
          //__________________________________
          //
          case TypeDescription::PerPatch : {

            PerPatchBase* perPatchVar = dynamic_cast<PerPatchBase*>(req->var->typeDescription()->createInstance());
            host_ptr   = perPatchVar->getBasePointer();
            host_bytes = perPatchVar->getDataSize();
            GPUPerPatch<void*> device_var;

            // check if the variable already exists on the GPU
            if (gpuDW->exists(reqVarName.c_str(), patchID, matlID, levelID)) {
              gpuDW->get(device_var, reqVarName.c_str(), patchID, matlID, levelID);
              device_ptr   = device_var.getPointer();
              device_bytes = device_var.getMemSize(); // TODO fix this
              
              // if the size is the same, assume the variable already exists on the GPU... no H2D copy
              if (host_bytes == device_bytes) {
                // report the above fact
                if (gpu_stats.active()) {
                  coutLock.lock();
                  {
                    gpu_stats << myRankThread() 
                              << " PerPatch (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
                  }
                  coutLock.unlock();
                }
                continue;
              } else {
                gpuDW->remove(reqVarName.c_str(), patchID, matlID, levelID);
              }
            }

            // critical section - prepare and async copy the requires variable data to the device
            h2dRequiresLock_.writeLock();
            {
              gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, levelID);

              if (gpu_stats.active()) {
                coutLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " Post H2D copy of REQUIRES (" << std::setw(26) << reqVarName <<  "), L-" << levelID << ", patch: " << patchID<< ", "
                            << "Bytes: "  << std::dec << host_bytes
                            << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
                            << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
                }
                coutLock.unlock();
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
    
    //__________________________________
    //
    for (int i = 0; i < numPatches; ++i) {
      for (int j = 0; j < numMatls; ++j) {
    
        int matlID  = matls->get(j);
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
              int3 nCells;
              if (name.compare("int") == 0) {
                GPUGridVariable<int> device_var;
                dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID,
                                               make_int3(low.x(), low.y(), low.z()),
                                               make_int3(high.x(), high.y(), high.z()), levelID);
                device_ptr = device_var.getPointer();
                num_bytes = device_var.getMemSize();
                int3 lo   = device_var.getLowIndex();
                int3 hi   = device_var.getHighIndex();
                nCells    = make_int3(hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
                
              } else if (name.compare("double") == 0) {
                GPUGridVariable<double> device_var;
                dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID,
                                               make_int3(low.x(), low.y(), low.z()),
                                               make_int3(high.x(), high.y(), high.z()), levelID);
                device_ptr = device_var.getPointer();
                num_bytes = device_var.getMemSize();
                int3 lo   = device_var.getLowIndex();
                int3 hi   = device_var.getHighIndex();
                nCells    = make_int3(hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
                
              } else if (name.compare("Stencil7") == 0) {
                GPUGridVariable<GPUStencil7> device_var;
                dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID,
                                               make_int3(low.x(), low.y(), low.z()),
                                               make_int3(high.x(), high.y(), high.z()), levelID);
                device_ptr = device_var.getPointer();
                num_bytes = device_var.getMemSize();
                int3 lo   = device_var.getLowIndex();
                int3 hi   = device_var.getHighIndex();
                nCells    = make_int3(hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
              } else {
                SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + compVarName, __FILE__, __LINE__));
              }

              if (gpu_stats.active()) {
                coutLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " Allocated device memory for COMPUTES (" << std::setw(15) << compVarName << "), L-" << levelID << ", patch: " << patchID << ", "
                            <<  std::setw(10) << "Bytes: " << std::dec << num_bytes << ", "
                            <<  std::setw(10) << " nCells [" << nCells.x <<","<<nCells.y <<"," << nCells.z <<"], "
                            << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum() 
                            << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
                }
                coutLock.unlock();
              }
            }
            d2hComputesLock_.writeUnlock();
            break;

          }  // end GridVariable switch case
          //__________________________________
          //
          case TypeDescription::ReductionVariable : {

            d2hComputesLock_.writeLock();
            {
              levelID = -1;
              GPUReductionVariable<void*> device_var;
              dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              num_bytes = device_var.getMemSize();
              if (gpu_stats.active()) {
                coutLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " Allocated device memory for COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID
                            << ", Bytes: " << std::dec << num_bytes
                            << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
                            << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
                }
                coutLock.unlock();
              }
            }
            d2hComputesLock_.writeUnlock();
            break;

          }  // end ReductionVariable switch case
          //__________________________________
          //
          case TypeDescription::PerPatch : {

            d2hComputesLock_.writeLock();
            {
              GPUPerPatch<void*> device_var;
              dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID, levelID);
              device_ptr = device_var.getPointer();
              num_bytes = device_var.getMemSize();
              if (gpu_stats.active()) {
                coutLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " Allocated device memory for COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID
                            << ", Bytes: " << std::dec << num_bytes
                            << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
                            << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
                }
                coutLock.unlock();
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

    void* host_ptr   = NULL;    // host base pointer to raw data
    void* device_ptr = NULL;    // device base pointer to raw data
    size_t host_bytes = 0;      // raw byte count to copy to the device
    size_t device_bytes = 0;    // raw byte count to copy to the host
    IntVector host_low, host_high, host_offset, host_size, host_strides;

    int numPatches = patches->size();
    int numMatls = matls->size();
    //__________________________________
    //
    for (int i = 0; i < numPatches; ++i) {
      for (int j = 0; j < numMatls; ++j) {

        int matlID  = matls->get(j);
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
              int3 nCells;

              switch (host_strides.x()) {
                case sizeof(int) : {
                  GPUGridVariable<int> device_var;
                  dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  int3 lo   = device_var.getLowIndex();
                  int3 hi   = device_var.getHighIndex();
                  nCells    = make_int3(hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
                  break;
                }
                case sizeof(double) : {
                  GPUGridVariable<double> device_var;
                  dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  int3 lo   = device_var.getLowIndex();
                  int3 hi   = device_var.getHighIndex();
                  nCells    = make_int3(hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
                  break;
                }
                case sizeof(GPUStencil7) : {
                  GPUGridVariable<GPUStencil7> device_var;
                  dw->getGPUDW()->get(device_var, compVarName.c_str(), patchID, matlID, levelID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  int3 lo   = device_var.getLowIndex();
                  int3 hi   = device_var.getHighIndex();
                  nCells    = make_int3(hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
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
                  coutLock.lock();
                  {
                    gpu_stats << myRankThread()
                              << " Post D2H copy of COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID << ", "
                              << std::setw(10) << "Bytes: " << std::dec << host_bytes << ", "
                              << std::setw(10) << "nCells [" << nCells.x <<","<<nCells.y <<"," << nCells.z <<"]"
                              << ", from " << std::hex << device_ptr << " to " << std::hex << host_ptr
                              << ", using stream " << std::hex << dtask->getCUDAStream() << std::endl;
                  }
                  coutLock.unlock();
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
          //__________________________________
          //
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
                  coutLock.lock();
                  {
                    gpu_stats << myRankThread()
                              << " Post D2H copy of COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID<< ", "
                              << std::setw(10) << "Bytes: " << std::dec << host_bytes
                              << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
                              << ", using stream " << std::hex << dtask->getCUDAStream()<< std::endl;
                  }
                  coutLock.unlock();
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
          //__________________________________
          //
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
                  coutLock.lock();
                  {
                    gpu_stats << myRankThread()
                              << "Post D2H copy of COMPUTES ("<< std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID<< ", "
                              << std::setw(10) << "Bytes: " << std::dec << host_bytes
                              << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
                              << ", using stream " << std::hex << dtask->getCUDAStream() << std::endl;
                  }
                  coutLock.unlock();
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
          //__________________________________
          //
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
        coutLock.lock();
        {
          gpu_stats << myRankThread() << " Created CUDA stream " << std::hex << stream << " on device "
                    << std::dec << device << std::endl;
        }
        coutLock.unlock();
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
      coutLock.lock();
      {
        gpu_stats << myRankThread() <<  " Deallocating " << totalStreams << " total CUDA stream(s) for " << numQueues << " device(s)"<< std::endl;
      }
      coutLock.unlock();
    }

    for (size_t i = 0; i < numQueues; i++) {
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(i));

      if (gpu_stats.active()) {
        coutLock.lock();
        {
          gpu_stats << myRankThread() << " Deallocating " << idleStreams[i].size() << " CUDA stream(s) on device " << retVal << std::endl;
        }
        coutLock.unlock();
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
        coutLock.lock();
        {
          gpu_stats << myRankThread()
                    << " Issued CUDA stream " << std::hex << stream
                    << " on device " << std::dec << device << std::endl;
          coutLock.unlock();
        }
      }
    }
    else {  // shouldn't need any more than the queue capacity, but in case
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(device));
      // this will get put into idle stream queue and ultimately deallocated after final timestep
      stream = ((cudaStream_t*)malloc(sizeof(cudaStream_t)));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));

      if (gpu_stats.active()) {
        coutLock.lock();
        {
          gpu_stats << myRankThread()
                    << " Needed to create 1 additional CUDA stream " << std::hex << stream
                    << " for device " << std::dec << device << std::endl;
        }
        coutLock.unlock();
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
    coutLock.lock();
    {
      gpu_stats << myRankThread() 
                << " Reclaimed CUDA stream " << std::hex << stream << " on device " << std::dec << deviceNum << std::endl;
    }
    coutLock.unlock();
  }
}

#endif // end HAVE_CUDA

//______________________________________________________________________
//  generate string   <MPI rank>.<Thread ID>
//  useful to see who running what    
std::string
UnifiedScheduler::myRankThread()
{
  std::ostringstream out;
  out<< Uintah::Parallel::getMPIRank()<< "." << Thread::self()->myid();
  return out.str();
}

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
      coutLock.lock();
      std::string threadType = (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      unified_threaddbg << "Binding" << threadType << "thread ID " << d_thread_id << " to core " << d_thread_id << "\n";
      coutLock.unlock();
    }
    Thread::self()->set_affinity(d_thread_id);
  }

  while( true ) {
    d_runsignal.wait(d_runmutex); // wait for main thread signal.
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds() - d_waitstart;

    if (d_quit) {
      if (taskdbg.active()) {
        coutLock.lock();
        unified_threaddbg << "Worker " << d_rank << "-" << d_thread_id << " quitting" << "\n";
        coutLock.unlock();
      }
      return;
    }

    if (taskdbg.active()) {
      coutLock.lock();
      unified_threaddbg << "Worker " << d_rank << "-" << d_thread_id << ": executing tasks \n";
      coutLock.unlock();
    }

    try {
      d_scheduler->runTasks(d_thread_id);
    }
    catch (Exception& e) {
      coutLock.lock();
      std::cerr << "Worker " << d_rank << "-" << d_thread_id << ": Caught exception: " << e.message() << "\n";
      if (e.stackTrace()) {
        std::cerr << "Stack trace: " << e.stackTrace() << '\n';
      }
      coutLock.unlock();
    }

    if (taskdbg.active()) {
      coutLock.lock();
      unified_threaddbg << "Worker " << d_rank << "-" << d_thread_id << ": finished executing tasks   \n";
      coutLock.unlock();
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
