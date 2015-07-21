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
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUStencil7.h>
#endif

#include <sci_defs/cuda_defs.h>

#include <cstring>
#include <iomanip>
#define USE_PACKING

using namespace std;
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

static DebugStream unified_dbg("Unified_DBG", false);
static DebugStream unified_timeout("Unified_TimingsOut", false);
static DebugStream unified_queuelength("Unified_QueueLength", false);
static DebugStream unified_threaddbg("Unified_ThreadDBG", false);
static DebugStream unified_compactaffinity("Unified_CompactAffinity", true);

#ifdef HAVE_CUDA
static DebugStream gpu_stats("Unified_GPUStats", false);
DebugStream use_single_device("Unified_SingleDevice", false);
DebugStream simulate_multiple_gpus("SimulateMultipleGPUs", false);
DebugStream gpudbg("GPUDataWarehouse", false);
#endif

//______________________________________________________________________
//

UnifiedScheduler::UnifiedScheduler(const ProcessorGroup* myworld,
    const Output* oport, UnifiedScheduler* parentScheduler) :
    MPIScheduler(myworld, oport, parentScheduler), d_nextsignal(
        "next condition"), d_nextmutex("next mutex"), schedulerLock(
        "scheduler lock")
#ifdef HAVE_CUDA
        , d2hComputesLock_("Device-DB computes copy lock"), idleStreamsLock_(
        "CUDA streams lock"), h2dRequiresLock_("Device-DB requires copy lock")
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

    //assignPatchesToGpus(currentGrid);

  }

  int numThreads = Uintah::Parallel::getNumThreads();
  if (numThreads == -1) {
    numThreads = 1;
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

UnifiedScheduler::~UnifiedScheduler() {
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

void UnifiedScheduler::problemSetup(const ProblemSpecP& prob_spec,
    SimulationStateP& state) {
  // Default taskReadyQueueAlg
  taskQueueAlg_ = MostMessages;
  std::string taskQueueAlg = "MostMessages";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
    if (taskQueueAlg == "FCFS") {
      taskQueueAlg_ = FCFS;
    } else if (taskQueueAlg == "Random") {
      taskQueueAlg_ = Random;
    } else if (taskQueueAlg == "Stack") {
      taskQueueAlg_ = Stack;
    } else if (taskQueueAlg == "MostChildren") {
      taskQueueAlg_ = MostChildren;
    } else if (taskQueueAlg == "LeastChildren") {
      taskQueueAlg_ = LeastChildren;
    } else if (taskQueueAlg == "MostAllChildren") {
      taskQueueAlg_ = MostChildren;
    } else if (taskQueueAlg == "LeastAllChildren") {
      taskQueueAlg_ = LeastChildren;
    } else if (taskQueueAlg == "MostL2Children") {
      taskQueueAlg_ = MostL2Children;
    } else if (taskQueueAlg == "LeastL2Children") {
      taskQueueAlg_ = LeastL2Children;
    } else if (taskQueueAlg == "MostMessages") {
      taskQueueAlg_ = MostMessages;
    } else if (taskQueueAlg == "LeastMessages") {
      taskQueueAlg_ = LeastMessages;
    } else if (taskQueueAlg == "PatchOrder") {
      taskQueueAlg_ = PatchOrder;
    } else if (taskQueueAlg == "PatchOrderRandom") {
      taskQueueAlg_ = PatchOrderRandom;
    }
  }

  proc0cout << "   Using \"" << taskQueueAlg
      << "\" task queue priority algorithm" << std::endl;

  numThreads_ = Uintah::Parallel::getNumThreads() - 1;
  if (numThreads_ < 1
      && (Uintah::Parallel::usingMPI() || Uintah::Parallel::usingDevice())) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: no thread number specified for Unified Scheduler"
          << std::endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads to be in the range [2, 64],\n.... please use -nthreads <num>, and -gpu if using GPUs",
          __FILE__, __LINE__);
    }
  } else if (numThreads_ > MAX_THREADS) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: Number of threads too large..." << std::endl;
      throw ProblemSetupException(
          "Too many threads. Reduce MAX_THREADS and recompile.", __FILE__,
          __LINE__);
    }
  }

  if (d_myworld->myrank() == 0) {
    std::string plural = (numThreads_ == 1) ? " thread" : " threads";
    std::cout
        << "   WARNING: Multi-threaded Unified scheduler is EXPERIMENTAL, not all tasks are thread safe yet.\n"
        << "   Creating " << numThreads_ << " additional "
        << plural + " for task execution (total task execution threads = "
        << numThreads_ + 1 << ")." << std::endl;
#ifdef HAVE_CUDA
    if (Uintah::Parallel::usingDevice()) {
      cudaError_t retVal;
      int availableDevices;
      CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&availableDevices));
      std::cout << "   Using " << numDevices_ << "/" << availableDevices
          << " available GPU(s)" << std::endl;

      for (int device_id = 0; device_id < availableDevices; device_id++) {
        cudaDeviceProp device_prop;
        CUDA_RT_SAFE_CALL(
            retVal = cudaGetDeviceProperties(&device_prop, device_id));
        printf("   GPU Device %d: \"%s\" with compute capability %d.%d\n",
            device_id, device_prop.name, device_prop.major, device_prop.minor);
      }
    }
#endif
  }

  if (unified_compactaffinity.active()) {
    if ((unified_threaddbg.active()) && (d_myworld->myrank() == 0)) {
      unified_threaddbg << "   Binding main thread (ID "
          << Thread::self()->myid() << ") to core 0\n";
    }
    Thread::self()->set_affinity(0);  // Bind main thread to core 0
  }

  // Create the UnifiedWorkers here (pinned to cores in UnifiedSchedulerWorker::run())
  char name[1024];
  for (int i = 0; i < numThreads_; i++) {
    UnifiedSchedulerWorker* worker = scinew UnifiedSchedulerWorker(this, i);
    t_worker[i] = worker;
    sprintf(name, "Computing Worker %d-%d",
        Parallel::getRootProcessorGroup()->myrank(), i);
    Thread* t = scinew Thread(worker, name);
    t_thread[i] = t;
  }

  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);

#ifdef HAVE_CUDA
  //Now pick out the materials out of the file.  This is done with an assumption that there
  //will only be ICE or MPM problems, and no problem will have both ICE and MPM materials in it.
  //I am unsure if this assumption is correct.
  //TODO: Add in MPM material support, just needs to look for an MPM block instead of an ICE block.
  ProblemSpecP mp = prob_spec->findBlockWithOutAttribute("MaterialProperties");
  if (mp) {
    ProblemSpecP group = mp->findBlock("ICE");
    if (group) {
      for (ProblemSpecP child = group->findBlock("material"); child != 0;
          child = child->findNextBlock("material")) {
        ProblemSpecP EOS_ps = child->findBlock("EOS");
        if (!EOS_ps) {
          throw ProblemSetupException("ERROR ICE: Cannot find EOS tag",
              __FILE__, __LINE__);
        }

        std::string EOS;
        if (!EOS_ps->getAttribute("type", EOS)) {
          throw ProblemSetupException("ERROR ICE: Cannot find EOS 'type' tag",
              __FILE__, __LINE__);
        }

        //add this material to the collection of materials
        materialsNames.push_back(EOS);
      }
    }
  }
#endif

}

//______________________________________________________________________
//

SchedulerP UnifiedScheduler::createSubScheduler() {
  UnifiedScheduler* subsched = scinew UnifiedScheduler(d_myworld, m_outPort,
      this);
  UintahParallelPort* lbp = getPort("load balancer");
  subsched->attachPort("load balancer", lbp);
  subsched->d_sharedState = d_sharedState;
  subsched->numThreads_ = Uintah::Parallel::getNumThreads() - 1;

  if (subsched->numThreads_ > 0) {

    proc0cout << "\n" << "   Using EXPERIMENTAL multi-threaded sub-scheduler\n"
        << "   WARNING: Component tasks must be thread safe.\n"
        << "   Creating " << subsched->numThreads_
        << " subscheduler threads for task execution.\n\n" << std::endl;

    // Bind main execution thread
    if (unified_compactaffinity.active()) {
      if ((unified_threaddbg.active()) && (d_myworld->myrank() == 0)) {
        unified_threaddbg << "Binding main subscheduler thread (ID "
            << Thread::self()->myid() << ") to core 0\n";
      }
      Thread::self()->set_affinity(0); // bind subscheduler main thread to core 0
    }

    // Create UnifiedWorker threads for the subscheduler
    char name[1024];
    ThreadGroup* subGroup = new ThreadGroup("subscheduler-group", 0); // 0 is main/parent thread group
    for (int i = 0; i < subsched->numThreads_; i++) {
      UnifiedSchedulerWorker* worker = scinew UnifiedSchedulerWorker(subsched,
          i);
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

void UnifiedScheduler::runTask(DetailedTask* task, int iteration,
    int thread_id /* = 0 */, Task::CallBackEvent event) {
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
  for (int i = 0; i < (int) dws.size(); i++) {
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
      if (!d_sharedState->isCopyDataTimestep()
          && task->getTask()->getType() != Task::Output) {
        getLoadBalancer()->addContribution(task, dtask);
      }
    }
  }
  dlbLock.unlock();

  // For CPU and postGPU task runs, post MPI sends and call task->done;
  if (event == Task::CPU || event == Task::postGPU) {
#ifdef HAVE_CUDA
    if (Uintah::Parallel::usingDevice()) {
      //Go through internal dependencies and external dependencies.  For
      //anything in a GPU, copy the ghost cells into a contiguous chunk of data.
      //Then copy the internal dependencies from one GPU to another.
      //Then also copy external dependencies to the CPU so the MPI engine can handle those.

      DeviceGridVariables deviceVars; //Holds variables that will need to be copied into the GPU
      DeviceGridVariables taskVars; //Holds variables that will be needed for a GPU task (a Task DW has a snapshot of
                                    //all important pointer info from the host-side GPU DW)
      DeviceGhostCells ghostVars;  //Holds ghost cell meta data copy information

      findIntAndExtGpuDependencies(deviceVars, taskVars, ghostVars, task, iteration, thread_id);
      //The ghost cell destinations indicate which devices we're using,
      //and which ones we'll need streams for.
      assignDevicesAndStreams(ghostVars, task);
      createTaskGpuDWs(task, taskVars, ghostVars, 0); //TODO: Fix for multiple GPUs

      //place everything in the GPU data warehouses
      prepareDeviceVars(task, deviceVars);
      prepareTaskVarsIntoTaskDW(task, taskVars);
      prepareGhostCellsIntoTaskDW(task, ghostVars);
      syncTaskGpuDWs(task);  //TODO: Fix for multiple GPUs

      //get these ghost cells to contiguous arrays so they can be copied to host.
      performInternalGhostCellCopies(task);

      //copy all dependencies to arrays
      copyAllExtGpuDependenciesToHost(task, deviceVars, ghostVars);

      //In order to help copy values to another on-node GPU or another MPI rank, ghost cell data
      //was placed in a var in the patch it is *going to*.  It helps reuse gpu dw engine code this way.
      //But soon, after this task is done, we are likely going to receive a different region of that patch
      //from a neighboring on-node GPU or neighboring MPI rank.  So we need to remove this foreign variable
      //now so it can be used again.
      clearForeignGpuVars(deviceVars);
    }
#endif
    if (Uintah::Parallel::usingMPI()) {
      postMPISends(task, iteration, thread_id);
    }
#ifdef HAVE_CUDA

    //Go through internal dependencies and external dependencies.  For
    //anything in a GPU, copy the ghost cells into a contiguous chunk of data.
    //Then copy the internal dependencies from one GPU to another.
    //Then also copy external dependencies to the CPU so the MPI engine can handle those.
    if (Uintah::Parallel::usingDevice()) {
      //copyGpuInternalDependencies(task, iteration, thread_id);
      task->deleteTaskGpuDataWarehouses();  //TODO: Fix for multiple GPUs
    }
    reclaimCudaStreams(task);

#endif
    task->done(dws);  // should this be timed with taskstart? - BJW
    double teststart = Time::currentSeconds();

    if (Uintah::Parallel::usingMPI()) {
      // This is per thread, no lock needed.
      sends_[thread_id].testsome(d_myworld);
    }

    mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;

    // add my timings to the parent scheduler
    if (parentScheduler_) {
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

void UnifiedScheduler::execute(int tgnum /* = 0 */, int iteration /* = 0 */) {
  // copy data and restart timesteps must be single threaded for now
  bool isMPICopyDataTS = Uintah::Parallel::usingMPI()
      && d_sharedState->isCopyDataTimestep();
  bool isRestartTS = d_isInitTimestep || d_isRestartInitTimestep;
  if (isMPICopyDataTS || isRestartTS) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::execute");

  TAU_PROFILE("UnifiedScheduler::execute()", " ", TAU_USER); TAU_PROFILE_TIMER(reducetimer, "Reductions", "[UnifiedScheduler::execute()] " , TAU_USER); TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[UnifiedScheduler::execute()] " , TAU_USER); TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[UnifiedScheduler::execute()] " , TAU_USER); TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[UnifiedScheduler::execute()] ", TAU_USER); TAU_PROFILE_TIMER(testsometimer, "Test Some", "[UnifiedScheduler::execute()] ", TAU_USER); TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[UnifiedScheduler::execute()] ", TAU_USER); TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[UnifiedScheduler::execute()] ", TAU_USER); TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[UnifiedScheduler::execute()] ", TAU_USER);

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
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(),
        reloc_new_posLabel_, iteration);
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
      unified_dbg << "\n" << "Rank-" << me << " Executing " << dts->numTasks()
          << " tasks (" << ntasks << " local)\n" << "Total task phases: "
          << numPhases << "\n";
      for (size_t phase = 0; phase < phaseTasks.size(); ++phase) {
        unified_dbg << "Phase: " << phase << " has " << phaseTasks[phase]
            << " total tasks\n";
      }
      unified_dbg << std::endl;
    }
    cerrLock.unlock();
  }

  static int totaltasks;

  if (taskdbg.active()) {
    cerrLock.lock();
    taskdbg << myRankThread() << " starting task phase " << currphase
        << ", total phase " << currphase << " tasks = " << phaseTasks[currphase]
        << std::endl;
    cerrLock.unlock();
  }

  // signal worker threads to begin executing tasks
  for (int i = 0; i < numThreads_; i++) {
    t_worker[i]->resetWaittime(Time::currentSeconds()); // reset wait time counter
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
    MPI_Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0,
        d_myworld->getComm());

    proc0cout << "average queue length:" << allqueuelength / d_myworld->size()
        << std::endl;
  }

  emitTime("MPI Send time", mpi_info_.totalsendmpi);
  emitTime("MPI Recv time", mpi_info_.totalrecvmpi);
  emitTime("MPI TestSome time", mpi_info_.totaltestmpi);
  emitTime("MPI Wait time", mpi_info_.totalwaitmpi);
  emitTime("MPI reduce time", mpi_info_.totalreducempi);
  emitTime("Total send time",
      mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
  emitTime("Total recv time",
      mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
  emitTime("Total task time", mpi_info_.totaltask);
  emitTime("Total reduction time",
      mpi_info_.totalreduce - mpi_info_.totalreducempi);
  emitTime("Total comm time",
      mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

  double time = Time::currentSeconds();
  double totalexec = time - d_lasttime;
  d_lasttime = time;

  emitTime("Other excution time",
      totalexec - mpi_info_.totalsend - mpi_info_.totalrecv
          - mpi_info_.totaltask - mpi_info_.totalreduce);

  if (d_sharedState != 0) {
    d_sharedState->taskExecTime += mpi_info_.totaltask
        - d_sharedState->outputTime;  // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv
        + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
    for (int i = 0; i < numThreads_; i++) {
      d_sharedState->taskWaitThreadTime += t_worker[i]->getWaittime();
    }
  }

  if (restartable && tgnum == (int) graphs.size() - 1) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size() - 1]->timestepRestarted();
    int netrestart;

    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR,
        d_myworld->getComm());

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

  if (unified_timeout.active() && !parentScheduler_) { // only do on toplevel scheduler
    // add number of cells, patches, and particles
    int numCells = 0, numParticles = 0;
    OnDemandDataWarehouseP dw = dws[dws.size() - 1];
    const GridP grid(const_cast<Grid*>(dw->getGrid()));
    const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(
        grid)->getSubset(d_myworld->myrank());
    for (int p = 0; p < myPatches->size(); p++) {
      const Patch* patch = myPatches->get(p);
      IntVector range = patch->getExtraCellHighIndex()
          - patch->getExtraCellLowIndex();
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
    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int) d_times.size(), MPI_DOUBLE,
        MPI_SUM, 0, comm);
    MPI_Reduce(&d_times[0], &d_maxtimes[0], (int) d_times.size(), MPI_DOUBLE,
        MPI_MAX, 0, comm);

    double total = 0, avgTotal = 0, maxTotal = 0;
    for (int i = 0; i < static_cast<int>(d_totaltimes.size()); i++) {
      d_avgtimes[i] = d_totaltimes[i] / d_myworld->size();
      if (strcmp(d_labels[i], "Total task time") == 0) {
        avgTask = d_avgtimes[i];
        maxTask = d_maxtimes[i];
      } else if (strcmp(d_labels[i], "Total comm time") == 0) {
        avgComm = d_avgtimes[i];
        maxComm = d_maxtimes[i];
      } else if (strncmp(d_labels[i], "Num", 3) == 0) {
        if (strcmp(d_labels[i], "NumCells") == 0) {
          avgCell = d_avgtimes[i];
          maxCell = d_maxtimes[i];
        }
        // these are independent stats - not to be summed
        continue;
      }

      total += d_times[i];
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
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep()
          << std::endl;
      for (int i = 0; i < static_cast<int>((*data[file]).size()); i++) {
        out << "UnifiedScheduler: " << d_labels[i] << ": ";
        int len = static_cast<int>((strlen(d_labels[i]))
            + strlen("UnifiedScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++) {
          out << ' ';
        }
        double percent = 0.0;
        if (strncmp(d_labels[i], "Num", 3) == 0) {
          percent =
              d_totaltimes[i] == 0 ?
                  100 : (*data[file])[i] / d_totaltimes[i] * 100;
        } else {
          percent = (*data[file])[i] / total * 100;
        }
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << std::endl << std::endl;
    }

    if (me == 0) {
      unified_timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask
          << " = " << (1 - avgTask / maxTask) * 100
          << " load imbalance (exec)%\n";
      unified_timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm
          << " = " << (1 - avgComm / maxComm) * 100
          << " load imbalance (comm)%\n";
      unified_timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell
          << " = " << (1 - avgCell / maxCell) * 100
          << " load imbalance (theoretical)%\n\n";
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
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(),
          d_myworld->myrank());
      fout.open(filename);

      // Report which timesteps TaskExecTime values have been accumulated over
      fout << "Reported values are cumulative over 10 timesteps ("
          << d_sharedState->getCurrentTopLevelTimeStep() - 9 << " through "
          << d_sharedState->getCurrentTopLevelTimeStep() << ")" << std::endl;

      for (std::map<std::string, double>::iterator iter = exectimes.begin();
          iter != exectimes.end(); iter++) {
        fout << std::fixed << d_myworld->myrank() << ": TaskExecTime(s): "
            << iter->second << " Task:" << iter->first << std::endl;
      }
      fout.close();
      exectimes.clear();
    }
  }

  if (waitout.active()) {
    static int count = 0;

    // only output the wait times every 10 timesteps
    if (++count % 10 == 0) {

      if (d_myworld->myrank() == 0
          || d_myworld->myrank() == d_myworld->size() / 2
          || d_myworld->myrank() == d_myworld->size() - 1) {
        std::ofstream wout;
        char fname[100];
        sprintf(fname, "waittimes.%d.%d", d_myworld->size(),
            d_myworld->myrank());
        wout.open(fname);

        for (std::map<std::string, double>::iterator iter = waittimes.begin();
            iter != waittimes.end(); iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(TO): "
              << iter->second << " Task:" << iter->first << "\n";
        }

        for (std::map<std::string, double>::iterator iter =
            DependencyBatch::waittimes.begin();
            iter != DependencyBatch::waittimes.end(); iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): "
              << iter->second << " Task:" << iter->first << std::endl;
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

void UnifiedScheduler::runTasks(int thread_id) {
  int me = d_myworld->myrank();

  while (numTasksDone < ntasks) {

    DetailedTask* readyTask = NULL;
    DetailedTask* initTask = NULL;

    int pendingMPIMsgs = 0;
    bool havework = false;

#ifdef HAVE_CUDA
    bool usingDevice = Uintah::Parallel::usingDevice();
    bool gpuInitReady = false;
    bool gpuRunReady = false;
    bool gpuFinalizeDevicePreparation = false;
    bool gpuPending = false;
    bool gpuDataCopyPending = false;
    bool gpuDataCopyReady = false;
    bool cpuRunReady = false;

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
      if ((phaseSyncTask[currphase] != NULL)
          && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {
        readyTask = phaseSyncTask[currphase];
        havework = true;
        numTasksDone++;
#ifdef HAVE_CUDA
        cpuRunReady = true;
#endif
        if (taskorder.active()) {
          if (me == d_myworld->size() / 2) {
            cerrLock.lock();
            taskorder << myRankThread() << " Running task static order: "
                << readyTask->getStaticOrder() << " , scheduled order: "
                << numTasksDone << std::endl;
            cerrLock.unlock();
          }
        }
        phaseTasksDone[readyTask->getTask()->d_phase]++;
        while (phaseTasks[currphase] == phaseTasksDone[currphase]
            && currphase + 1 < numPhases) {
          currphase++;
          if (taskdbg.active()) {
            cerrLock.lock();
            taskdbg << myRankThread() << " switched to task phase " << currphase
                << ", total phase " << currphase << " tasks = "
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
           * If it's a GPU-enabled task, assign it to a device (patches were assigned devices previously)
           * and initiate its H2D computes and requires data copies. This is where the
           * execution cycle begins for each GPU-enabled Task.
           *
           * gpuInitReady = true
           */
          if (readyTask->getTask()->usesDevice()) {
            assignDevice(readyTask);
            gpuInitReady = true;
          } else {
#endif

#ifdef HAVE_CUDA
            if (usingDevice == false || readyTask->getPatches() == NULL) {
              //These tasks won't ever have anything to pull out of the device
              //so go ahead and mark the task "done" and say that it's ready
              //to start running as a CPU task.
              numTasksDone++;
              cpuRunReady = true;
            }
#else
            //if NOT compiled with device support, then this is a CPU task and we can mark the task "done"
            numTasksDone++;
#endif
            if (taskorder.active()) {
              if (d_myworld->myrank() == d_myworld->size() / 2) {
                cerrLock.lock();
                taskorder << myRankThread() << " Running task static order: "
                    << readyTask->getStaticOrder() << ", scheduled order: "
                    << numTasksDone << std::endl;
                cerrLock.unlock();
              }
            }
            phaseTasksDone[readyTask->getTask()->d_phase]++;
            while (phaseTasks[currphase] == phaseTasksDone[currphase]
                && currphase + 1 < numPhases) {
              currphase++;
              if (taskdbg.active()) {
                cerrLock.lock();
                taskdbg << myRankThread() << " switched to task phase "
                    << currphase << ", total phase " << currphase << " tasks = "
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
          if (initTask->getTask()->getType() == Task::Reduction
              || initTask->getTask()->usesMPI()) {
            if (taskdbg.active()) {
              cerrLock.lock();
              taskdbg << myRankThread() << " Task internal ready 1 "
                  << *initTask << std::endl;
              cerrLock.unlock();
            }
            phaseSyncTask[initTask->getTask()->d_phase] = initTask;
            ASSERT(initTask->getRequires().size() == 0)
            initTask = NULL;
          } else if (initTask->getRequires().size() == 0) { // no ext. dependencies, then skip MPI sends
            initTask->markInitiated();
            initTask->checkExternalDepCount(); // where tasks get added to external ready queue
            initTask = NULL;
          } else {
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
       * then reclaim the streams and events it used for these operations, and start
       * the GPU to GPU ghost cell copies.  Also make GPU data as being valid as it is now
       * copied into the device.
       *
       * gpuFinalizeDevicePreparation = true
       */
      else if (usingDevice == true && dts->numFinalizeDevicePreparation() > 0) {
        readyTask = dts->peekNextFinalizeDevicePreparationTask();
        if (readyTask->checkCUDAStreamDone()) {
          readyTask = dts->getNextFinalizeDevicePreparationTask();
          gpuFinalizeDevicePreparation = true;
          havework = true;
          break;
        }
      }
      /*
       * (1.4.1)
       *
       * Check if highest priority GPU task's asynchronous device to device ghost cell copies are
       * finished. If so, then reclaim the streams and events it used for these operations, execute
       * the task and then put it into the GPU completion-pending queue.
       *
       * gpuRunReady = true
       */
      else if (usingDevice == true && dts->numInitiallyReadyDeviceTasks() > 0) {
        //TODO: This should check both if statements.
        //Suppose the numInitiallyReadyDeviceTasks has an item but
        //copies are not complete.  Will this thread be stuck waiting in an
        //infinite loop waiting for the copy to complete?  And since this loop
        //is governed by a mutex, all other threads get stuck too?   This same
        //possible problem exists twice more below.  BRP -  5/10/2015
        //Further, suppose there are 1000 tasks in a node, each wanting to do
        //an H2D copy to the GPU.  The way this is laid out we would do
        //all 1000 H2Ds before we would start a compute.  This is not a
        //good way to do it.  BRP - 7/9/2015
        readyTask = dts->peekNextInitiallyReadyDeviceTask();
        if (readyTask->checkCUDAStreamDone()) {

          // printf("Ghost cell copies done...\n");
          // All of this task's h2d copies is complete, so add it to the completion
          // pending GPU task queue and prepare to run.
          readyTask = dts->getNextInitiallyReadyDeviceTask();
          gpuRunReady = true;
          havework = true;

          break;
        }
      }

      /*
       * (1.4.2)
       *
       * Check if highest priority GPU task's asynchronous D2H copies are completed. If so,
       * then reclaim the streams and events it used for these operations, execute the task and
       * then put it into the CPU completion-pending queue.
       *
       * cpuRunReady = true
       */
      else if (usingDevice == true && dts->numInitiallyReadyHostTasks() > 0) {
        readyTask = dts->peekNextInitiallyReadyHostTask();
        if (readyTask->checkAllCUDAStreamsDone()) {
          numTasksDone++; //If there's a GPU, then all host run tasks have to go through here.
          // recycle this task's D2H copies streams and events
          reclaimCudaStreams(readyTask);
          readyTask = dts->getNextInitiallyReadyHostTask();
          cpuRunReady = true;
          havework = true;
          break;
        }
      }
      /*
       * (1.5)
       *
       * Check to see if any GPU tasks have been completed. This means the kernel(s)
       * have executed (which prevents out of order kernels, and also keeps tasks that depend on
       * its data to wait until the async kernel call is done).
       * This task's MPI sends can then be posted and done() can be called.
       *
       * gpuPending = true
       */
      else if (usingDevice == true
          && dts->numCompletionPendingDeviceTasks() > 0) {
        readyTask = dts->peekNextCompletionPendingDeviceTask();
        if (readyTask->checkCUDAStreamDone()) {
          readyTask = dts->getNextCompletionPendingDeviceTask();
          havework = true;
          gpuPending = true;
          numTasksDone++;
          if (taskorder.active()) {
            if (d_myworld->myrank() == d_myworld->size() / 2) {
              cerrLock.lock();
              taskorder << myRankThread() << " Running task static order: "
                  << readyTask->getStaticOrder() << " , scheduled order: "
                  << numTasksDone << std::endl;
              cerrLock.unlock();
            }
          }
          phaseTasksDone[readyTask->getTask()->d_phase]++;
          while (phaseTasks[currphase] == phaseTasksDone[currphase]
              && currphase + 1 < numPhases) {
            currphase++;
            if (taskdbg.active()) {
              cerrLock.lock();
              taskdbg << myRankThread() << " switched to task phase "
                  << currphase << ", total phase " << currphase << " tasks = "
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
        taskdbg << myRankThread() << " Task internal ready 2 " << *initTask
            << " deps needed: " << initTask->getExternalDepCount() << std::endl;
        cerrLock.unlock();
      }
      initTask->markInitiated();
      initTask->checkExternalDepCount();
    } else if (readyTask != NULL) {
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << myRankThread() << " Task external ready " << *readyTask
            << std::endl;
        cerrLock.unlock();
      }
      if (readyTask->getTask()->getType() == Task::Reduction) {
        initiateReduction(readyTask);
      }
#ifdef HAVE_CUDA
      else if (gpuInitReady) {
        //prepare to run a GPU task.

        //Ghost cells from CPU same device to variable not yet on GPU -> Managed already by getGridVar()
        //Ghost cells from CPU same device to variable already on GPU -> Managed in initiateH2DCopies(), then copied with performInternalGhostCellCopies()
        //Ghost cells from GPU other device to variable not yet on GPU -> new MPI code and getGridVar()
        //Ghost cells from GPU other device to variable already on GPU -> new MPI code, then initiateH2DCopies(), and copied with performInternalGhostCellCopies()
        //Ghost cells from GPU same device to variable not yet on GPU -> managed in initiateH2DCopies(), and copied with performInternalGhostCellCopies()
        //Ghost cells from GPU same device to variable already on GPU -> Managed in initiateH2DCopies()?

        // initiate all asynchronous H2D memory copies for this task's requires
        readyTask->setCUDAStream(getCudaStream(readyTask->getDeviceNum()));
        initiateH2DCopies(readyTask);
        syncTaskGpuDWs(readyTask);

        dts->addFinalizeDevicePreparation(readyTask);
      } else if (gpuFinalizeDevicePreparation) {
        //The requires data is either already there or has just been copied in so mark it as valid.
        markDeviceRequiresDataAsValid(readyTask);

        //TODO: Move this into the gpuInitReady section and then test it.
        //Do GPU to GPU ghost cell copies if device ghost cells exist.
        performInternalGhostCellCopies(readyTask);
        dts->addInitiallyReadyDeviceTask(readyTask);

      } else if (gpuRunReady) {
        runTask(readyTask, currentIteration, thread_id, Task::GPU);
        //postD2HCopies(readyTask);
        //initiateD2HCopies(readyTask);
        dts->addCompletionPendingDeviceTask(readyTask);
      } else if (gpuPending) {

        //The GPU task has completed. All of the computes data is now valid and should be marked as such.
        markDeviceComputesDataAsValid(readyTask);

        //The Task GPU Datawarehouses are no longer needed.  Delete them on the host and device.
        readyTask->deleteTaskGpuDataWarehouses();

        // run post GPU part of task
        //processD2HCopies(readyTask);
        runTask(readyTask, currentIteration, thread_id, Task::postGPU);
        // recycle this task's stream
        reclaimCudaStreams(readyTask);

      }
#endif
      else {
        //prepare to run a CPU task.
#ifdef HAVE_CUDA

        if (!cpuRunReady) {
          // Figure out which device this patch was assigned to.
          // If a task has multiple patches, then assume that all patches for this
          // task are on the same device, so the 0th patch will work.
          //TODO, bad assumption as output var task will work on multiple patches.
          assignDevice(readyTask);

          //See if there are any copies needed from the GPU.

          //If the task graph has scheduled to output variables, make sure that we are
          //going to actually output variables before pulling data out of the GPU.
          //(It would be nice if the task graph didn't have this OutputVariables task if
          //it wasn't going to output data, but that would require more task graph recompilations,
          //which can be even costlier overall.  So we do the check here.)
          //So check everything, except for ouputVariables tasks when it's not an output timestep.
          if ((m_outPort->isOutputTimestep())
              || ((readyTask->getTask()->getName()
                  != "DataArchiver::outputVariables")
                  && (readyTask->getTask()->getName()
                      != "DataArchiver::outputVariables(checkpoint)"))) {
            if (readyTask->getTask()->getName()
                == "DataArchiver::outputVariables(checkpoint)") {
              cerr
                  << "WARNING: A bug in the unified scheduler means checkpoint DataArchiver can conflict with regular data archiving."
                  << endl;
            }
            initiateD2H(readyTask);
          }
          dts->addInitiallyReadyHostTask(readyTask);

        } else {

          //See comment above why this exists
          if (usingDevice == true
              && (m_outPort->isOutputTimestep()
                  || (readyTask->getTask()->getName()
                      != "DataArchiver::outputVariables"))) {
            markHostRequiresDataAsValid(readyTask);
          }
#endif
          //run CPU task.
          runTask(readyTask, currentIteration, thread_id, Task::CPU);
          printTaskLevels(d_myworld, taskLevel_dbg, readyTask);
#ifdef HAVE_CUDA
        }
#endif
      }
    } else if (pendingMPIMsgs > 0) {
      processMPIRecvs(TEST);
    } else {
      // This can only happen when all tasks have finished.
      ASSERT(numTasksDone == ntasks);
    }
  }  //end while (numTasksDone < ntasks)
}

//______________________________________________________________________
//

struct CompareDep {

  bool operator()(DependencyBatch* a, DependencyBatch* b) {
    return a->messageTag < b->messageTag;
  }
};

//______________________________________________________________________
//

int UnifiedScheduler::pendingMPIRecvs() {
  int num = 0;
  recvLock.readLock();
  num = recvs_.numRequests();
  recvLock.readUnlock();
  return num;
}

//______________________________________________________________________
//
int UnifiedScheduler::getAvailableThreadNum() {
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
void UnifiedScheduler::prepareGpuDependencies(DeviceGridVariables& deviceVars,
    DeviceGridVariables& taskVars, DeviceGhostCells& ghostVars,
    DetailedTask* task, DependencyBatch* batch, const VarLabel* pos_var,
    OnDemandDataWarehouse* dw, OnDemandDataWarehouse* old_dw,
    const DetailedDep* dep, LoadBalancer* lb,
    DeviceGhostCells::Destination dest) {

  //This should handle the following scenarios:
  //GPU -> different GPU same node  (write to GPU array, move to other device memory, copy in via copyGPUGhostCellsBetweenDevices)
  //GPU -> different GPU another node (write to GPU array, move to host memory, copy via MPI)
  //GPU -> CPU another node (write to GPU array, move to host memory, copy via MPI)
  //It should not handle
  //GPU -> CPU same node (handled in initateH2D)
  //GPU -> same GPU same node (handled in initateH2D)

  if (dep->isNonDataDependency()) {
    return;
  }
  const VarLabel* label = dep->req->var;
  const Patch* fromPatch = dep->fromPatch;
  const int matlIndx = dep->matl;
  const Level* level = fromPatch->getLevel();
  const int levelID = level->getID();

  bool fromGPU;
  bool toGPU;

  //TODO: Ask Alan about everything in the dep object.
  //The toTasks (will there be more than one?)
  //the dep->comp (computes?)
  //the dep->req (requires?)

  DetailedTask* toTask = NULL;
  //Go through all toTasks
  for (list<DetailedTask*>::const_iterator iter = dep->toTasks.begin();
      iter != dep->toTasks.end(); ++iter) {
    toTask = (*iter);
    //cout << (*iter)->getPatches()->get(0)->getID() << endl;

    const Patch* toPatch = toTask->getPatches()->get(0);
    //const int toPatchID = toPatch->getID();

    const int fromresource = task->getAssignedResourceIndex();
    const int toresource = toTask->getAssignedResourceIndex();

    const int fromDeviceIndex = GpuUtilities::getGpuIndexForPatch(fromPatch);
    //For now, assume that task will only work on one device
    const int toDeviceIndex = GpuUtilities::getGpuIndexForPatch(toTask->getPatches()->get(0));

    const size_t elementSize = OnDemandDataWarehouse::getTypeDescriptionSize(
        dep->req->var->typeDescription()->getSubType()->getType());

    //printf("In OnDemandDataWarehouse::prepareGPUDependencies from patch %d to patch %d, from task %p to task %p\n", fromPatch->getID(), toTask->getPatches()->get(0)->getID(), task, toTask);

    if ((fromresource == toresource) && (fromDeviceIndex == toDeviceIndex)) {
      //don't handle GPU -> same GPU same node here
      continue;
    }

    GPUDataWarehouse* gpudw = NULL;
    if (fromDeviceIndex != -1) {
      gpudw = dw->getGPUDW(fromDeviceIndex);
      if (!gpudw->getValidOnCPU(label->d_name.c_str(), fromPatch->getID(),
          matlIndx, levelID)
          && gpudw->getValidOnGPU(label->d_name.c_str(), fromPatch->getID(),
              matlIndx, levelID)) {
        fromGPU = true;
      } else {
        //no need to prepare CPU -> other ghost cell copies here.
        continue;
      }
    } else {
      SCI_THROW(
          InternalError("Device index not found for "+label->getFullName(matlIndx, fromPatch), __FILE__, __LINE__));
    }
    if (toTask->getTask()->usesDevice()) {
      //The above tells us that it going to a GPU.
      //For now, when the receiving end calls getGridVar, it will piece together
      //the ghost cells, and move it into the GPU if needed.
      //Therefore, we don't at this moment need to know if it's going to a GPU.
      //But in the future, if we can manage direct GPU->GPU communication avoiding
      //the CPU then this is important to know
      toGPU = true;
    }

    switch (label->typeDescription()->getType()) {
    case TypeDescription::ParticleVariable: {
    }
      break;
    case TypeDescription::NCVariable:
    case TypeDescription::CCVariable:
    case TypeDescription::SFCXVariable:
    case TypeDescription::SFCYVariable:
    case TypeDescription::SFCZVariable: {
      /*
       d_lock.readLock();
       if( ! dw->exists(label, matlIndx, fromPatch)) {
       cout << d_myworld->myrank() << "  Needed by " << *dep << " on task " << *dep->toTasks.front() << endl;
       SCI_THROW(UnknownVariable(label->getName(), dw->getID(), fromPatch, matlIndx,
       "in OnDemandDataWarehouse::prepareGPUDependencies", __FILE__, __LINE__));
       }
       IntVector host_low, host_high, host_offset, host_size, host_strides;

       //TODO: Does this need to be deallocated?
       GridVariableBase* var = dynamic_cast<GridVariableBase*>(dep->req->var->typeDescription()->createInstance());

       // check for case when INF ghost cells are requested such as in RMCRT
       //   in this case we need deto use getRegion()
       bool uses_SHRT_MAX = (dep->req->numGhostCells == SHRT_MAX);
       if (uses_SHRT_MAX) {
       const Level* level = getLevel(task->getPatches());
       IntVector domainLo_EC, domainHi_EC;
       level->findCellIndexRange(domainLo_EC, domainHi_EC);  // including extraCells
       dw->getRegion(*var, dep->req->var, fromPatch->getID(), level, domainLo_EC, domainHi_EC, true);
       var->getSizes(domainLo_EC, domainHi_EC, host_offset, host_size, host_strides);
       } else {
       dw->getGridVar(*var, dep->req->var, matlIndx, fromPatch->getID(), GhostCells::None, 0);
       var->getSizes(host_low, host_high, host_offset, host_size, host_strides);
       }

       //var = dynamic_cast<GridVariableBase*>(d_varDB.get(label, matlIndx, fromPatch));
       d_lock.readUnlock();
       */
      //
      //var->getSizes(host_low, host_high, host_offset, host_size, host_strides);
      //TODO, This compiles a list of regions we need to copy into contiguous arrays.
      //We don't yet handle a scenario where the ghost cell region is exactly the same
      //size as the variable, meaning we don't need to create an array and copy to it.

      if (gpu_stats.active()) {
        gpu_stats << myRankThread()
            << " prepareGpuDependencies - Handling this task's dependency for "
            << " --> Preparing GPU dependencies for sending " << *dep
            << ", ghost: " << dep->req->gtype << ", " << dep->req->numGhostCells
            << " from dw " << dw->getID() << '\n';
      }

      //We're going to copy the ghost vars from the source variable (already in the GPU) to a destination
      //array (not yet in the GPU).  So make sure there is a destination.

      //TODO: Doesn't yet work if two local patches are sending ghost cells to one remote patch (i.e. a corner case)

      //TODO: Clean this up var up
      //make the temp array on the host
      GridVariableBase* tempGhostVar =
          dynamic_cast<GridVariableBase*>(label->typeDescription()->createInstance());
      tempGhostVar->allocate(dep->low, dep->high);
      IntVector host_low, host_high, host_offset, host_size, host_strides;
      tempGhostVar->getSizes(host_low, host_high, host_offset, host_size,
          host_strides);

      //TODO: Make Taskvars not pass in two extra parameters.
      //make the temp array in the GPU

      //This is somewhat hacky.  We can't have duplicate variables with the same
      //patch/label/matl/level/datawarehouseindex tuple.  But we want to have the GPU DataWarehouse
      //store the this temporary ghost cell region.  So indicate that it is being stored in the
      //patch it is going to.  This will allow us later to invoked a kernel to which can copy
      //the ghost cell region from a patch to an array to the "other patch", so that array can
      //then be copied to another on-node GPU or another MPI rank.  By keeping this variable
      //in the "other patch", it allows us to reuse much of the existing GPU DW code.
      deviceVars.add(toPatch, matlIndx, levelID, host_size,
          tempGhostVar->getDataSize(), host_strides.x(), host_offset,
           dep->req, Ghost::None, 0, fromDeviceIndex, tempGhostVar, false, true);
      //let this Task GPU DW know about it
      taskVars.addTaskGpuDWVar(toPatch, matlIndx, levelID, host_strides.x(),
          dep->req, fromDeviceIndex);

      //let this Task GPU DW know about the source location.
      taskVars.addTaskGpuDWVar(fromPatch, matlIndx, levelID, host_strides.x(),
          dep->req, fromDeviceIndex);

      //TODO: If we were doing a GPU->another GPU same device transfer, we should not create the host array but instead create a
      //destination array.
      if (dest == DeviceGhostCells::anotherMpiRank) {

      }

      if (gpu_stats.active()) {
        gpu_stats << myRankThread()
            << " prepareGpuDependencies - Adding in a ghost cell"
            << " for " << dep->req->var->getName().c_str()
            << " from patch " << fromPatch->getID()
            << " to patch " << toPatch->getID()
            << " between shared low (" << dep->low.x() << ", " << dep->low.y() << ", " << dep->low.z() << ")"
            << " and shared high (" << dep->high.x() << ", " << dep->high.y() << ", " << dep->high.z() << ")"
            << " and host offset (" << host_offset.x() << ", " << host_offset.y() << ", " << host_offset.z() << ")"
            << endl;
      }
      IntVector temp(0,0,0);
      ghostVars.add(dep->req->var, fromPatch, toPatch,
          matlIndx, levelID, dep->low, dep->high,
          OnDemandDataWarehouse::getTypeDescriptionSize(
              dep->req->var->typeDescription()->getSubType()->getType()), temp,
          fromDeviceIndex, toDeviceIndex, fromresource, toresource,
          (Task::WhichDW) dep->req->mapDataWarehouse(), dest);


      /*gpudw->prepareGpuGhostCellIntoGpuArray(task,
       make_int3(dep->low.x(), dep->low.y(), dep->low.z()),
       make_int3(dep->high.x(), dep->high.y(), dep->high.z()),
       OnDemandDataWarehouse::getTypeDescriptionSize(dep->req->var->typeDescription()->getSubType()->getType()),
       label->d_name.c_str(),
       matlIndx,
       fromPatch->getID(),
       toTask->getPatches()->get(0)->getID(),
       fromDeviceIndex,
       toDeviceIndex,
       fromresource,
       toresource);
       */
      //Previously we made sure the ghost cell data is a contiguous array on the GPU.
      //Here we will copy that data from the GPU to the CPU.
      //Note: In order to make *GPU context aware* MPI work, the receiving end will have had to prepare
      //space in GPU memory to receive such ghost cell data, *and* that address information
      //will have to be known among the sending processes.  So for now, do it the slower, manual way.

      //gpudw->copyTempGhostCellsToHostVar(tempGhostvar->getBasePointer(),
      //                               make_int3(dep->low.x(), dep->low.y(), dep->low.z()),
      //                               make_int3(dep->high.x(), dep->high.y(), dep->high.z()),
      //                               label->d_name.c_str(),
      //                               patch->getID(),
      //                               matlIndx,
      //                               0);
      //Test data:
      //double* tmp = (double*)tempGhostvar->getBasePointer();
      //for (int i = 0; i < 101 * 101; i++) {
      //   tmp[i] = (double)i;
      //}
      //I only need this var to live long enough to have the MPI_Send complete, then
      //it cleans it up for us.
      //Will setting it to foreign properly accomplish this?
      //tempGhostvar->setForeign();
      //d_varDB.putForeign(label, matlIndx, patch, tempGhostvar, d_scheduler->isCopyDataTimestep()); //put new var in data warehouse
      //send it all
      //make the temp array on the host

    }
      break;
    default: {
      cerr
          << "UnifiedScheduler::prepareGPUDependencies(), unsupported variable type"
          << endl;
    }

    }
  }
}

//______________________________________________________________________
//

void UnifiedScheduler::gpuInitialize(bool reset) {

  cudaError_t retVal;
  if (simulate_multiple_gpus.active()) {
    printf("SimulateMultipleGPUs is on, simulating 3 GPUs\n");
    numDevices_ = 3;
  } else if (use_single_device.active()) {
    numDevices_ = 1;
  } else {
    int numDevices = 0;
    CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&numDevices));
    numDevices_ = numDevices;
  }

  if (simulate_multiple_gpus.active()) {

    //we're simulating many, but we only will use one.
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));
    if (reset) {
      CUDA_RT_SAFE_CALL(retVal = cudaDeviceReset());
    }
  } else {
    for (int i = 0; i < numDevices_; i++) {
      if (reset) {
        CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(i));
        CUDA_RT_SAFE_CALL(retVal = cudaDeviceReset());
      }
    }
    //set it back to the 0th device
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));
    currentDevice_ = 0;
  }

}

//______________________________________________________________________
//
void UnifiedScheduler::postH2DCopies(DetailedTask* dtask) {
  /*
   MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::postH2DCopies");
   TAU_PROFILE("UnifiedScheduler::postH2DCopies()", " ", TAU_USER);

   // set the device and CUDA context
   cudaError_t retVal;
   int device = dtask->getDeviceNum();
   OnDemandDataWarehouse::uintahSetCudaDevice(device);
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
   {  //lock
   // logic for to avoid getting level variables multiple times
   bool alreadyCopied = ( gpuDW->existsLevelDB(reqVarName.c_str(), levelID, matlID) );

   if(isLevelItem && alreadyCopied) {
   cerrLock.lock();
   //          std::cout <<  "    " << myRankThread() << " Goiing to skip this variable " << reqVarName.c_str() << " Patch: " << patchID << std::endl;
   cerrLock.unlock();
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


   // * Until better type information support is implemented for GPUGridVariables
   // *   we need to determine the size of a single element in the Arary3Data object to
   // *   know what type of GPUGridVariable to create and use.
   // *
   // *   "host_strides.x()" == sizeof(T)
   // *
   // *   This approach currently supports:
   // *   ------------------------------------------------------------
   // *   GPUGridVariable<int>
   // *   GPUGridVariable<double>
   // *   GPUGridVariable<GPUStencil7>
   // *   ------------------------------------------------------------

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
   //                  cerrLock.lock();
   //                  {
   //                    gpu_stats << "GridVariable (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
   //                  }
   //                  cerrLock.unlock();
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
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), matlID, device_low, device_hi, host_strides.x(), levelID);
   }
   else {
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, host_strides.x());
   }
   device_ptr = device_var.getPointer();
   break;
   }
   case sizeof(double) : {
   GPUGridVariable<double> device_var;

   if (isLevelItem) {
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), matlID, device_low, device_hi, , host_strides.x(), levelID);
   }
   else {
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, host_strides.x(), levelID);
   }
   device_ptr = device_var.getPointer();
   break;
   }
   case sizeof(GPUStencil7) : {
   GPUGridVariable<GPUStencil7> device_var;
   if (isLevelItem) {
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), matlID, device_low, device_hi, host_strides.x(), levelID);
   }
   else {
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, device_low, device_hi, host_strides.x(), levelID);
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
   int3 nCells    = make_int3(device_hi.x-device_low.x, device_hi.y-device_low.y, device_hi.z-device_low.z);
   gpu_stats << myRankThread()
   << " Post H2D copy of REQUIRES (" << std::setw(26) << reqVarName <<  "), L-" << levelID << ", patch: " << patchID<< ", "
   << std::setw(10) << "Bytes: "  << std::dec << host_bytes <<", "
   << std::setw(10) << "nCells [" << nCells.x <<","<<nCells.y <<"," << nCells.z <<"]"
   << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
   << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
   }
   cerrLock.unlock();
   }
   cudaStream_t stream = *(dtask->getCUDAStream());
   CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(device_ptr, host_ptr, host_bytes, cudaMemcpyHostToDevice, stream));
   } //lock
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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << "ReductionVariable (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
   }
   cerrLock.unlock();
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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Post H2D copy of REQUIRES (" << std::setw(26) << reqVarName <<  "), L-" << levelID << ", patch: " << patchID<< ", "
   << "Bytes = "  << std::dec << host_bytes
   << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
   << ", using stream " << std::hex << dtask->getCUDAStream() << std::endl;
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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " PerPatch (" << reqVarName << ") already exists, skipping H2D copy..." << std::endl;
   }
   cerrLock.unlock();
   }
   continue;
   } else {
   gpuDW->remove(reqVarName.c_str(), patchID, matlID, levelID);
   }
   }

   // critical section - prepare and async copy the requires variable data to the device
   h2dRequiresLock_.writeLock();
   {
   gpuDW->allocateAndPut(device_var, reqVarName.c_str(), patchID, matlID, 0 levelID);

   if (gpu_stats.active()) {
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Post H2D copy of REQUIRES (" << std::setw(26) << reqVarName <<  "), L-" << levelID << ", patch: " << patchID<< ", "
   << "Bytes: "  << std::dec << host_bytes
   << " from " << std::hex << host_ptr << " to " << std::hex <<  device_ptr
   << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
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
   */
}

//______________________________________________________________________
//
//______________________________________________________________________
//

void UnifiedScheduler::preallocateDeviceMemory(DetailedTask* dtask) {
  /*
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

   // * Until better type information support is implemented for GPUGridVariables
   // *   we need to use:
   // *
   // *   std::string name = comp->var->typeDescription()->getSubType()->getName()
   // *
   // *   to determine what sub-type the computes variables consist of in order to
   // *   create the correct type of GPUGridVariable. Can't use info from getSizes()
   // *   as the variable doesn't exist (e.g. allocateAndPut() hasn't been called yet)
   // *
   // *   This approach currently supports:
   // *   ------------------------------------------------------------
   // *   GPUGridVariable<int>
   // *   GPUGridVariable<double>
   // *   GPUGridVariable<GPUStencil7>
   // *   ------------------------------------------------------------

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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Allocated device memory for COMPUTES (" << std::setw(15) << compVarName << "), L-" << levelID << ", patch: " << patchID << ", "
   <<  std::setw(10) << "Bytes: " << std::dec << num_bytes << ", "
   <<  std::setw(10) << " nCells [" << nCells.x <<","<<nCells.y <<"," << nCells.z <<"], "
   << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
   << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
   }
   cerrLock.unlock();
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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Allocated device memory for COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID
   << ", Bytes: " << std::dec << num_bytes
   << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
   << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
   }
   cerrLock.unlock();
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
   dw->getGPUDW()->allocateAndPut(device_var, compVarName.c_str(), patchID, matlID, 0, levelID);
   device_ptr = device_var.getPointer();
   num_bytes = device_var.getMemSize();
   if (gpu_stats.active()) {
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Allocated device memory for COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID
   << ", Bytes: " << std::dec << num_bytes
   << " at " << std::hex << device_ptr << " on device " << std::dec << dtask->getDeviceNum()
   << ", using stream " << std::hex << dtask->getCUDAStream()  << std::endl;
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
   */
}

//______________________________________________________________________
//

void UnifiedScheduler::postD2HCopies(DetailedTask* dtask) {
  /*
   MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::postD2HCopies");
   TAU_PROFILE("UnifiedScheduler::postD2HCopies()", " ", TAU_USER);

   // set the device and CUDA context
   cudaError_t retVal;
   int device = dtask->getDeviceNum();
   OnDemandDataWarehouse::uintahSetCudaDevice(device);
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

   // * Until better type information support is implemented for GPUGridVariables
   // *   we need to determine the size of a single element in the Arary3Data object to
   // *   know what type of GPUGridVariable to create and use.
   // *
   // *   "host_strides.x()" == sizeof(T)
   // *
   // *   This approach currently supports:
   // *   ------------------------------------------------------------
   // *   GPUGridVariable<int>
   // *   GPUGridVariable<double>
   // *   GPUGridVariable<GPUStencil7>
   // *   ------------------------------------------------------------


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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Post D2H copy of COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID << ", "
   << std::setw(10) << "Bytes: " << std::dec << host_bytes << ", "
   << std::setw(10) << "nCells [" << nCells.x <<","<<nCells.y <<"," << nCells.z <<"]"
   << ", from " << std::hex << device_ptr << " to " << std::hex << host_ptr
   << ", using stream " << std::hex << dtask->getCUDAStream() << std::endl;
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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << " Post D2H copy of COMPUTES (" << std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID<< ", "
   << std::setw(10) << "Bytes: " << std::dec << host_bytes
   << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
   << ", using stream " << std::hex << dtask->getCUDAStream()<< std::endl;
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
   cerrLock.lock();
   {
   gpu_stats << myRankThread()
   << "Post D2H copy of COMPUTES ("<< std::setw(26) << compVarName << "), L-" << levelID << ", patch: " << patchID<< ", "
   << std::setw(10) << "Bytes: " << std::dec << host_bytes
   << " from " << std::hex << device_ptr << " to " << std::hex << host_ptr
   << ", using stream " << std::hex << dtask->getCUDAStream() << std::endl;
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
   */
}

void UnifiedScheduler::initiateH2DCopies(DetailedTask* dtask) {
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::initiateH2DCopies"); TAU_PROFILE("UnifiedScheduler::initiateH2DCopies()", " ", TAU_USER);

  const Task* task = dtask->getTask();

  //Store information about each set of grid variables.
  //This will help later when we figure out the best way to store data into the GPU.
  //It may be stored contiguously.  It may handle material data.  It just helps to gather it all up
  //in one bunch in case we need some piece of data.
  DeviceGridVariables deviceVars; //Holds variables that will need to be copied into the GPU
  DeviceGridVariables taskVars; //Holds variables that will be needed for a GPU task (a Task DW has a snapshot of
                                //all important pointer info from the host-side GPU DW)
  DeviceGhostCells ghostVars;  //Holds ghost cell meta data copy information

  //Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
  //transfer some variables twice).
  //Note: A task can only run on one level at a time.  It could run multiple patches and multiple
  //materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires();
      dependantVar != 0; dependantVar = dependantVar->next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(
        dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->var->getName().c_str(),
            patches->get(i)->getID(), matls->get(j), Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(
              std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(
                  lpmd, dependantVar));
        }
      }
    }
  }
  for (const Task::Dependency* dependantVar = task->getComputes();
      dependantVar != 0; dependantVar = dependantVar->next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(
        dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->var->getName().c_str(),
            patches->get(i)->getID(), matls->get(j), Task::Computes);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(
              std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(
                  lpmd, dependantVar));
        }
      }
    }
  }

  unsigned int device_id = -1;
  //The task runs on one device.  The first patch we see can be used to tell us
  //which device we should be on.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  varIter = vars.begin();
  if (varIter != vars.end()) {
    device_id = GpuUtilities::getGpuIndexForPatch(
        varIter->second->getPatchesUnderDomain(dtask->getPatches())->get(0));
    OnDemandDataWarehouse::uintahSetCudaDevice(device_id);
  }

  //Go through each unique dependent var and see if we should allocate space and/or
  //queue it to be copied H2D.
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* curDependency = varIter->second;

    constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(
        dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    // for now, we're only interested in grid variables
    const TypeDescription::Type type =
        curDependency->var->typeDescription()->getType();
    for (int i = 0; i < numPatches; i++) {

      const int patchID = patches->get(i)->getID();
      const Level* level = getLevel(dtask->getPatches());
      const int levelID = level->getID();

      // a fix for when INF ghost cells are requested such as in RMCRT
      //  e.g. tsk->requires(abskg_dw, d_abskgLabel, gac, SHRT_MAX);
      bool uses_SHRT_MAX = (curDependency->numGhostCells == SHRT_MAX);

      if (type == TypeDescription::CCVariable
          || type == TypeDescription::NCVariable
          || type == TypeDescription::SFCXVariable
          || type == TypeDescription::SFCYVariable
          || type == TypeDescription::SFCZVariable
          || type == TypeDescription::PerPatch) {

        //For this dependency, get its CPU Data Warehouse and GPU Datawarehouse.
        const int dwIndex = curDependency->mapDataWarehouse();
        OnDemandDataWarehouseP dw = dws[dwIndex];
        GPUDataWarehouse* gpudw = dw->getGPUDW(
            GpuUtilities::getGpuIndexForPatch(patches->get(i)));

        //Get all size information about this dependency.
        IntVector low, high, lowOffset, highOffset;
        Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
        Patch::getGhostOffsets(type, curDependency->gtype,
            curDependency->numGhostCells, lowOffset, highOffset);
        patches->get(i)->computeExtents(basis,
            curDependency->var->getBoundaryLayer(), lowOffset, highOffset, low,
            high);
        const IntVector host_size = high - low;
        const size_t elementDataSize =
            OnDemandDataWarehouse::getTypeDescriptionSize(
                curDependency->var->typeDescription()->getSubType()->getType());

        for (int j = 0; j < numMatls; j++) {

          const int matlID = matls->get(j);
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread()
                  << " InitiateH2D - Handling this task's dependency for "
                  << curDependency->var->getName() << " for patch: " << patchID
                  << " material: " << matlID << " level: " << levelID;
              if (curDependency->deptype == Task::Requires) {
                gpu_stats << " - A REQUIRES dependency" << endl;
              } else if (curDependency->deptype == Task::Computes) {
                gpu_stats << " - A COMPUTES dependency" << endl;
              }
            }
            cerrLock.unlock();
          }

          if (curDependency->deptype == Task::Requires) {

            //See if this variable still exists on the device from some prior task
            //and if it's valid.  If so, use it.  If not, queue it to be possibly allocated
            //and then copy it in H2D.
            const bool exists = gpudw->exist(
                curDependency->var->getName().c_str(), patchID, matlID,
                levelID);
            bool existsCorrectSize = exists;
            if (type == TypeDescription::CCVariable
                || type == TypeDescription::NCVariable
                || type == TypeDescription::SFCXVariable
                || type == TypeDescription::SFCYVariable
                || type == TypeDescription::SFCZVariable) {
              existsCorrectSize = gpudw->exist(
                  curDependency->var->getName().c_str(), patchID, matlID,
                  levelID,
                  make_int3(host_size.x(), host_size.y(), host_size.z()),
                  make_int3(low.x(), low.y(), low.z()));
            }
            const bool validOnGpu = gpudw->getValidOnGPU(
                curDependency->var->getName().c_str(), patchID, matlID,
                levelID);

            if (exists && existsCorrectSize && validOnGpu) {
              //This requires variable already exists on the GPU.  Queue it to be added
              //to this tasks's TaskDW.
              taskVars.addTaskGpuDWVar(patches->get(i), matlID, levelID,
                  elementDataSize, curDependency,
                  GpuUtilities::getGpuIndexForPatch(patches->get(i)));

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                      << " GridVariable: "
                      << curDependency->var->getName().c_str()
                      << " already exists, skipping H2D copy..." << std::endl;
                }
                cerrLock.unlock();
              }

              //The variable's space already exists on the GPU, and it's valid on the GPU.  So we copy in any ghost
              //cells into the GPU and let the GPU handle the ghost cell copying logic.

              //Get all valid adjacent neighbors, and see if ghost cells are needed.
              //If so, queue them to be set up in the GPU.
              vector<OnDemandDataWarehouse::ValidNeighbors> validNeighbors;
              dw->getValidNeighbors(curDependency->var, matlID, patches->get(i),
                  curDependency->gtype, curDependency->numGhostCells,
                  validNeighbors);
              for (vector<OnDemandDataWarehouse::ValidNeighbors>::iterator iter =
                  validNeighbors.begin(); iter != validNeighbors.end();
                  ++iter) {

                const Patch* sourcePatch = NULL;
                if (iter->neighborPatch->getID() >= 0) {
                  sourcePatch = iter->neighborPatch;
                } else {
                  //This occurs on virtual patches.  They can be "wrap around" patches, meaning if you go to one end of a domain
                  //you will show up on the other side.  Virtual patches have negative patch IDs, but they know what real patch they
                  //are referring to.
                  sourcePatch = iter->neighborPatch->getRealPatch();
                }
                int sourceDeviceNum = GpuUtilities::getGpuIndexForPatch(
                    sourcePatch);
                int destDeviceNum = GpuUtilities::getGpuIndexForPatch(
                    patches->get(i));
                //ghost cell copies between GPUs occurs in copyGPUGhostCellsBetweenDevices()
                //around the same time MPI data is processed.
                if ((sourceDeviceNum == destDeviceNum)
                    || (dw->getGPUDW(sourceDeviceNum)->getValidOnCPU(
                        curDependency->var->getName().c_str(),
                        sourcePatch->getID(), matlID, levelID))) {

                  //See if we need to push into the device ghost cell data from the CPU.
                  //This will happen one for one of two reasons.
                  //First, if the ghost cell source data isn't valid on the GPU, then we assume it must be
                  //on the CPU.  (This can happen from a prior MPI transfer from another node).
                  //Second, if the ghost cell source data is valid on the GPU *and* CPU *and*
                  //the ghost cell is from one GPU to another GPU on the same machine
                  //then we are going to use the CPU data.  (This happens after an output task, when
                  //the data in a GPU is copied to the CPU to be copied on the hard drive.)
                  //In either case, we copy the ghost cell from the CPU to the appropriate GPU
                  //then a kernel can merge these ghost cells into the GPU grid var.
                  //Note: In the future when modifies support is included, then
                  //it would be possible to be valid on the CPU and not valid on the GPU
                  //We can run into a tricky situation where a GPU datawarehouse may try to bring in
                  //a patch multiple times for different ghost cells (e.g. patch 1 needs 0's ghost cells
                  //and patch 2 needs 0's ghost cells.  It will try to bring in patch 0 twice).  The fix
                  //is to see if patch 0 in the GPUDW exists, is queued up, but not valid.
                  bool useCpuGhostCells = false;
                  if (!(dw->getGPUDW(sourceDeviceNum)->exist(
                      curDependency->var->getName().c_str(),
                      sourcePatch->getID(), matlID, levelID))
                      || !(dw->getGPUDW(sourceDeviceNum)->getValidOnGPU(
                          curDependency->var->getName().c_str(),
                          sourcePatch->getID(), matlID, levelID))) {
                    useCpuGhostCells = true;
                  } else if (dw->getGPUDW(sourceDeviceNum)->getValidOnCPU(
                      curDependency->var->getName().c_str(),
                      sourcePatch->getID(), matlID, levelID)
                      && sourceDeviceNum != destDeviceNum) {
                    useCpuGhostCells = true;
                  } else if (!(dw->getGPUDW(sourceDeviceNum)->getValidOnCPU(
                      curDependency->var->getName().c_str(),
                      sourcePatch->getID(), matlID, levelID))
                      && !(dw->getGPUDW(sourceDeviceNum)->getValidOnGPU(
                          curDependency->var->getName().c_str(),
                          sourcePatch->getID(), matlID, levelID))) {
                    printf(
                        "Needed ghost cell data not found on the CPU or a GPU\n");
                    exit(-1);
                  } //else it's on the GPU.

                  if (useCpuGhostCells) {
                    //Add it to the collection so it will be copied into a GPU
                    IntVector ghost_host_low, ghost_host_high,
                        ghost_host_offset, ghost_host_size, ghost_host_strides;
                    iter->validNeighbor->getSizes(ghost_host_low,
                        ghost_host_high, ghost_host_offset, ghost_host_size,
                        ghost_host_strides);
                    GridVariableBase* srcvar = iter->validNeighbor->cloneType();
                    srcvar->copyPointer(*(iter->validNeighbor));
                    if (gpu_stats.active()) {
                      cerrLock.lock();
                      {
                        gpu_stats << myRankThread()
                            << " The CPU has ghost cells needed from patch "
                            << sourcePatch->getID() << " to "
                            << patches->get(i)->getID() << " from device "
                            << sourceDeviceNum << " to device " << destDeviceNum
                            << ".  The neighbor variable is at (" << ghost_host_low.x()
                            << ", " << ghost_host_low.y()
                            << ", " << ghost_host_low.z()
                            << ") with size (" << ghost_host_size.x()
                            << ", " << ghost_host_size.y()
                            << ", " << ghost_host_size.z()
                            << ") with offset (" << ghost_host_offset.x()
                            << ", " << ghost_host_offset.y()
                            << ", " << ghost_host_offset.z() << ")"
                            << endl;
                      }
                      cerrLock.unlock();
                    }
                    //Copy this patch into the GPU.
                    deviceVars.add(sourcePatch, matlID, levelID,
                        ghost_host_size, srcvar->getDataSize(),
                        ghost_host_strides.x(), ghost_host_offset,
                        curDependency, Ghost::None, 0, destDeviceNum,
                        srcvar, true, false);
                    //This task needs to know of this ghost cell as well.
                    taskVars.addTaskGpuDWVar(sourcePatch, matlID, levelID,
                        ghost_host_strides.x(), curDependency, destDeviceNum);
                  } else {
                    if (gpu_stats.active()) {
                      cerrLock.lock();
                      {
                        gpu_stats << myRankThread()
                            << "The CPU does not need to supply ghost cells from patch "
                            << sourcePatch->getID() << " to "
                            << patches->get(i)->getID() << " from device "
                            << sourceDeviceNum << " to device " << destDeviceNum
                            << endl;
                      }
                      cerrLock.unlock();
                    }

                    //The Task DW will need to know about this patch containing ghost cell data.
                    //This sourcePatch isn't a patch to be computed, so if we didn't put this
                    //taskVars.addTaskGpuDWVar() below, then the TaskDW would never have metadata for this patch.
                    if (!patches->contains(sourcePatch)) {
                      IntVector ghost_host_low, ghost_host_high,
                          ghost_host_offset, ghost_host_size,
                          ghost_host_strides;
                      iter->validNeighbor->getSizes(ghost_host_low,
                          ghost_host_high, ghost_host_offset, ghost_host_size,
                          ghost_host_strides);
                      //In a corner case or a periodic boundary condition case, the same source patch could be added multiple times.
                      taskVars.addTaskGpuDWVar(sourcePatch, matlID, levelID,
                          ghost_host_strides.x(), curDependency, destDeviceNum);
                    }

                  }

                  //Store the source and destination patch, and the range of the ghost cells
                  //A GPU kernel will use this collection to do all internal GPU ghost cell copies for
                  //that one specific GPU.

                  IntVector virtualOffset =
                      iter->neighborPatch->getVirtualOffset();
                  ghostVars.add(curDependency->var,
                      sourcePatch, patches->get(i), matlID, levelID, iter->low,
                      iter->high, elementDataSize, virtualOffset,
                      sourceDeviceNum, destDeviceNum, -1, -1,
                      (Task::WhichDW) curDependency->mapDataWarehouse(),
                      DeviceGhostCells::sameDeviceSameMpiRank);
                  if (gpu_stats.active()) {
                    cerrLock.lock();
                    {
                      gpu_stats << myRankThread()
                          << " Internal GPU ghost cell copy queued for "
                          << curDependency->var->getName().c_str() << " from patch "
                          << sourcePatch->getID() << " to patch " << patchID << " from ("
                          << iter->low.x() << ", " << iter->low.y() << ", "
                          << iter->low.z() << ")" << " to (" << iter->high.x()
                          << ", " << iter->high.y() << ", " << iter->high.z()
                          << ")" << " with virtual patch offset ("
                          << virtualOffset.x() << ", " << virtualOffset.y()
                          << ", " << virtualOffset.z() << ")"
                          << "." << endl;
                    }
                    cerrLock.unlock();
                  }
                }
              }

            } else if (exists && !existsCorrectSize) {
              //At the moment this isn't allowed. So it does an exit(-1).  There are two reasons for this.
              //First, the current CPU system always needs to "resize" variables when ghost cells are required.
              //Essentially the variables weren't created with room for ghost cells, and so room  needs to be created.
              //This step can be somewhat costly (I've seen a benchmark where it took 5% of the total computation time).
              //And at the moment this hasn't been coded to resize on the GPU.  It would require an additional step and
              //synchronization to make it work.
              //The second reason is with concurrency.  Suppose a patch that CPU thread A own needs
              //ghost cells from a patch that CPU thread B owns.
              //A can recognize that B's data is valid on the GPU, and so it stores for the future to copy B's
              //data on in.  Meanwhile B notices it needs to resize.  So A could start trying to copy in B's
              //ghost cell data while B is resizing its own data.
              //I believe both issues can be fixed with proper checkpoints.  But in reality
              //we shouldn't be resizing variables on the GPU, so this event should never happenn.
              gpudw->remove(curDependency->var->getName().c_str(), patchID,
                  matlID, levelID);
              cerr
                  << "Resizing of GPU grid vars not implemented at this time.  For the GPU, computes need to be declared with scratch computes to have room for ghost cells."
                  << endl;
              exit(-1);

            } else if (!exists
                || (exists && existsCorrectSize && !validOnGpu)) {

              //It's either not on the GPU, or space exists on the GPU for it but it is invalid.
              //Either way, gather all ghost cells host side (if needed), then queue the data to be
              //copied in H2D.  If the data doesn't exist in the GPU, then the upcoming allocateAndPut
              //will allocate space for it.  Otherwise if it does exist on the GPU, the upcoming
              //allocateAndPut will notice that and simply configure it to reuse the pointer.
              if (type == TypeDescription::CCVariable
                  || type == TypeDescription::NCVariable
                  || type == TypeDescription::SFCXVariable
                  || type == TypeDescription::SFCYVariable
                  || type == TypeDescription::SFCZVariable) {

                // Since the var is on the host, this will manage ghost cells by
                // creating a host var, rewindowing it, then copying in the regions needed for the ghost cells.
                // If this is the case, then ghost cells for this specific instance of this var is completed.
                //Note: Unhandled scenario:  If the adjacent patch is only in the GPU, this code doesn't gather it.

                GridVariableBase* gridVar =
                    dynamic_cast<GridVariableBase*>(curDependency->var->typeDescription()->createInstance());
                if (uses_SHRT_MAX) {
                  IntVector domainLo_EC, domainHi_EC;
                  level->findCellIndexRange(domainLo_EC, domainHi_EC); // including extraCells
                  dw->getRegion(*gridVar, curDependency->var, matlID, level,
                      domainLo_EC, domainHi_EC, true);
                } else {
                  dw->getGridVar(*gridVar, curDependency->var, matlID,
                      patches->get(i), curDependency->gtype,
                      curDependency->numGhostCells);
                }
                IntVector host_low, host_high, host_offset, host_size,
                    host_strides;
                gridVar->getSizes(host_low, host_high, host_offset, host_size,
                    host_strides);

                //Queue this CPU var to go into the host-side GPU DW.
                //Also queue that this GPU DW var should also be found in this tasks's Task DW.
                deviceVars.add(patches->get(i), matlID, levelID, host_size,
                    gridVar->getDataSize(), host_strides.x(), host_offset,
                    curDependency, curDependency->gtype, curDependency->numGhostCells,
                    GpuUtilities::getGpuIndexForPatch(patches->get(i)),
                    gridVar, true, false);
                taskVars.addTaskGpuDWVar(patches->get(i), matlID, levelID,
                    host_strides.x(), curDependency,
                    GpuUtilities::getGpuIndexForPatch(patches->get(i)));
              } else if (type == TypeDescription::PerPatch) {
                PerPatchBase* patchVar =
                    dynamic_cast<PerPatchBase*>(curDependency->var->typeDescription()->createInstance());
                dw->get(*patchVar, curDependency->var, matlID, patches->get(i));
                deviceVars.add(patches->get(i), matlID, levelID, patchVar->getDataSize(),
                    elementDataSize, curDependency,
                    GpuUtilities::getGpuIndexForPatch(patches->get(i)),
                    patchVar, true);
                taskVars.addTaskGpuDWVar(patches->get(i), matlID, levelID,
                    elementDataSize, curDependency,
                    GpuUtilities::getGpuIndexForPatch(patches->get(i)));
              } else {
                cerr << "UnifiedScheduler::initiateH2D(), unsupported variable type for computes variable "
                                   << curDependency->var->getName() << endl;
              }
            }

          } else if (curDependency->deptype == Task::Computes) {
            //compute the amount of space the host needs to reserve on the GPU for this variable.

            if (gpu_stats.active()) {
              cerrLock.lock();
              {
                gpu_stats << myRankThread()
                    << " The CPU is allocating space"
                    << " for " << curDependency->var->getName()
                    << " patch " << patches->get(i)->getID()
                    << " material " << matlID
                    << " level " << levelID
                    << " on device "
                    << GpuUtilities::getGpuIndexForPatch(patches->get(i))
                    << endl;
              }
              cerrLock.unlock();
            }

            if (type == TypeDescription::PerPatch) {
              //For PerPatch, it's not a mesh of variables, it's just a single variable, so elementDataSize is the memSize.
              size_t memSize = elementDataSize;

              PerPatchBase* patchVar =
                  dynamic_cast<PerPatchBase*>(curDependency->var->typeDescription()->createInstance());
              dw->put(*patchVar, curDependency->var, matlID, patches->get(i));
              delete patchVar;
              patchVar = NULL;
              deviceVars.add(patches->get(i), matlID, levelID,
                  memSize, elementDataSize, curDependency,
                  GpuUtilities::getGpuIndexForPatch(patches->get(i)),
                  NULL, false);
              taskVars.addTaskGpuDWVar(patches->get(i), matlID, levelID,
                  elementDataSize, curDependency,
                  GpuUtilities::getGpuIndexForPatch(patches->get(i)));
            } else if (type == TypeDescription::CCVariable
                || type == TypeDescription::NCVariable
                || type == TypeDescription::SFCXVariable
                || type == TypeDescription::SFCYVariable
                || type == TypeDescription::SFCZVariable) {

              //TODO, not correct if scratch ghost is used?

              GridVariableBase* gridVar =
                  dynamic_cast<GridVariableBase*>(curDependency->var->typeDescription()->createInstance());

              //Get variable size. Scratch computes means we need to factor that in when computing the size.
              Patch::VariableBasis basis = Patch::translateTypeToBasis(
                  curDependency->var->typeDescription()->getType(), false);
              IntVector lowIndex, highIndex;
              IntVector lowOffset, highOffset;

              Patch::getGhostOffsets(
                  gridVar->virtualGetTypeDescription()->getType(),
                  curDependency->gtype, curDependency->numGhostCells, lowOffset,
                  highOffset);
              patches->get(i)->computeExtents(basis,
                  curDependency->var->getBoundaryLayer(), lowOffset, highOffset,
                  lowIndex, highIndex);
              size_t memSize = (highIndex.x() - lowIndex.x())
                  * (highIndex.y() - lowIndex.y())
                  * (highIndex.z() - lowIndex.z()) * elementDataSize;

              //Even though it's only on the device, we to create space for the var on the host.
              //This makes it easy in case we ever need to perform any D2Hs.
              //TODO: Is this even needed?  InitiateD2H seems to create the var when it's needed host side.
              //SO wHat happens if we remove this?
              dw->allocateAndPut(*gridVar, curDependency->var, matlID,
                  patches->get(i), curDependency->gtype,
                  curDependency->numGhostCells);

              delete gridVar;
              gridVar = NULL;
              deviceVars.add(patches->get(i), matlID, levelID, host_size,
                  memSize, elementDataSize, low, curDependency,
                  curDependency->gtype, curDependency->numGhostCells,
                  GpuUtilities::getGpuIndexForPatch(patches->get(i)),
                  NULL, false, false);
              taskVars.addTaskGpuDWVar(patches->get(i), matlID, levelID,
                  elementDataSize, curDependency,
                  GpuUtilities::getGpuIndexForPatch(patches->get(i)));
            } else {
              cerr << "UnifiedScheduler::initiateH2D(), unsupported variable type for computes variable "
                   << curDependency->var->getName() << endl;
            }
          }
        }
      }
    }
  }

  // We've now gathered up all possible things that need to go on the device.  Copy it over.
  if (taskVars.numItems() > 0 || ghostVars.numItems() > 0) {

    //Create the data warehouses that will be for this task and sent
    //to the GPU.  This starts by allocating and sizing them.
    createTaskGpuDWs(dtask, taskVars, ghostVars, device_id);

    prepareDeviceVars(dtask, deviceVars);

    prepareTaskVarsIntoTaskDW(dtask, taskVars);

    prepareGhostCellsIntoTaskDW(dtask, ghostVars);

  }
}

void UnifiedScheduler::prepareDeviceVars(DetailedTask* dtask,
    DeviceGridVariables& deviceVars) {
  //There are two competing factors in play.  The latency of each cuda API copy call vs
  //the latency to do a host to host copy.  If there are far too many cuda copies, then that
  //API latency will be dominant, so the goal is to copy the data to a contiguous host array
  //then copy that array to the device, do computations, then copy the data back. {
  //If there are only a few cuda copies, it will be faster to skip the contiguous array step

  //If some threshold is met where we should use contiguous arrays
  //(to disable contiguous arrays, just set if (deviceVars.numItems() >= 0)


  string taskID = dtask->getName();

  if (deviceVars.numItems() >= 0) {
    //No contiguous arrays section


    void* device_ptr = NULL;  // device base pointer to raw data

    //Get all data in the GPU, and store it on the GPU Data Warehouse on the host, as only it
    //is responsible for management of data.  So this processes the previously collected deviceVars.
    map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap =
        deviceVars.getMap();
    for (map<DeviceGridVariableInfo::LabelPatchMatlLevelDw,
        DeviceGridVariableInfo>::iterator it = varMap.begin();
        it != varMap.end(); ++it) {
      int dwIndex = it->second.dep->mapDataWarehouse();
      IntVector offset = it->second.offset;
      IntVector size = it->second.sizeVector;
      IntVector low = offset;
      IntVector high = offset + size;

      //Allocate the vars if needed.  If they've already been allocated, then
      //this simply sets the var to reuse the existing pointer.
      switch (it->second.dep->var->typeDescription()->getType()) {
      case TypeDescription::PerPatch: {
        GPUPerPatchBase* patchVar = OnDemandDataWarehouse::createGPUPerPatch(
            it->second.sizeOfDataType);
        dws[dwIndex]->getGPUDW(it->second.whichGPU)->allocateAndPut(*patchVar,
            it->second.dep->var->getName().c_str(), it->first.patchID,
            it->first.matlIndx, it->first.levelIndx,
            it->second.sizeOfDataType);
        device_ptr = patchVar->getVoidPointer();
        delete patchVar;
      }
        break;
      case TypeDescription::CCVariable:
      case TypeDescription::NCVariable:
      case TypeDescription::SFCXVariable:
      case TypeDescription::SFCYVariable:
      case TypeDescription::SFCZVariable: {
        GridVariableBase* tempGhostvar =
            dynamic_cast<GridVariableBase*>(it->second.dep->var->typeDescription()->createInstance());
        tempGhostvar->allocate(low, high);
        delete tempGhostvar;

        GPUGridVariableBase* device_var =
            OnDemandDataWarehouse::createGPUGridVariable(
                it->second.sizeOfDataType);
        dws[dwIndex]->getGPUDW(it->second.whichGPU)->allocateAndPut(
            *device_var, it->second.dep->var->getName().c_str(),
            it->first.patchID, it->first.matlIndx, it->first.levelIndx,
            make_int3(low.x(), low.y(), low.z()),
            make_int3(high.x(), high.y(), high.z()),
            it->second.sizeOfDataType,
            (GPUDataWarehouse::GhostType) (it->second.gtype),
            it->second.numGhostCells);
        device_ptr = device_var->getVoidPointer();
        delete device_var;
      }
        break;
      default: {
        cerrLock.lock();
        cerr << "This variable's type is not supported." << endl;
        cerrLock.unlock();
      }
      }

      //If it's a requires, copy the data on over.  If it's a computes, leave it as allocated but unused space.
      if (it->second.dep->deptype == Task::Requires) {
        cudaStream_t stream = *(dtask->getCUDAStream(it->second.whichGPU));
        OnDemandDataWarehouse::uintahSetCudaDevice(it->second.whichGPU);
        switch (it->second.dep->var->typeDescription()->getType()) {
        case TypeDescription::PerPatch: {
          if (it->second.copyHostDataToDevice) {
            CUDA_RT_SAFE_CALL(
                cudaMemcpyAsync(device_ptr,
                    dynamic_cast<PerPatchBase*>(it->second.var)->getBasePointer(),
                    it->second.varMemSize, cudaMemcpyHostToDevice, stream));
            delete it->second.var;
          }
          }
          break;
        case TypeDescription::CCVariable:
        case TypeDescription::NCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: {
          if (it->second.copyHostDataToDevice) {

            if (gpu_stats.active()) {
              gpu_stats << myRankThread()
                  << " prepareDeviceVars - Copying into GPU #" << it->second.whichGPU
                  << " data for variable " << it->first.label
                  << " patch: " << it->first.patchID
                  << " material: " << it->first.matlIndx
                  << " level: " << it->first.levelIndx
                  << " into REQUIRES GPUDW " << endl;
            }

            CUDA_RT_SAFE_CALL(
                cudaMemcpyAsync(device_ptr,
                    dynamic_cast<GridVariableBase*>(it->second.var)->getBasePointer(),
                    it->second.varMemSize, cudaMemcpyHostToDevice, stream));
            //Now that this requires grid variable is copied onto the device,
            //we no longer need to hold onto this particular reference of the host side
            //version of it.  So go ahead and remove our reference to it.
            delete it->second.var;
          }
          }
          break;
        default:
          {
            cerrLock.lock();
            cerr << "Variable " << it->second.dep->var->getName() << " is of a type that is not supported on GPUs yet." << endl;
            cerrLock.unlock();
          }
        }
      }
    }

    //TODO: Materials have to go on as well.  Should they go on contiguously?
    //put materials on the datawarehouse
    //for (int dwIndex = 0; dwIndex < (int)dws.size(); dwIndex++) {
    //  if (deviceVars.getSizeForDataWarehouse(dwIndex) > 0) {
    //    dws[dwIndex]->getGPUDW()->putMaterials(materialsNames);
    //  }
    //}

  } else {

    /*
     //Create a large array on the host.  We'll copy this one array over, instead of copying
     //many little individual arrays for each label/material/patch.
     //Those arrays can copy fast, but the API call overhead to prepare a copy is huge.
     //So we're trying to reduce the amount of device API calls.

     //This array will also contain blank space for computes variables.

     //TODO: Support perPatch variables here?

     //Get a task ID.  All grid variables that are about to be placed in a device array belong to
     //a task/patch/timestep tuple ID.  So we'll use this ID to let the GPU DW scheduler know which
     //array to work with
     //TODO: If there are two threads and two patches, they share the same TaskID.
     string taskID = dtask->getName();
     for (int dwIndex = 0; dwIndex < (int)dws.size(); dwIndex++) {
     if (deviceVars.getSizeForDataWarehouse(dwIndex) > 0) {
     //create a contiguous array on the host and on the device for this datawarehouse.
     //TODO: Make it work for multiple GPUS/multiple
     dws[dwIndex]->getGPUDW()->allocate(taskID.c_str(), deviceVars.getSizeForDataWarehouse(dwIndex));

     if (gpu_stats.active()) {
     cerrLock.lock();
     {
     gpu_stats << "Allocated buffer of size " << deviceVars.getSizeForDataWarehouse(dwIndex) << endl;
     }
     cerrLock.unlock();
     }

     //put materials on the datawarehouse
     dws[dwIndex]->getGPUDW()->putMaterials(materialsNames);
     }
     }

     //first loop through all data that needs to be copied to the device.
     for (unsigned int i = 0; i < deviceVars.numItems(); i++ ) {
     if (deviceVars.getDependency(i)->deptype == Task::Requires) {
     int dwIndex =  deviceVars.getDependency(i)->mapDataWarehouse();
     OnDemandDataWarehouseP dw = dws[dwIndex];
     const Task::Dependency* dep = deviceVars.getDependency(i);
     IntVector offset = deviceVars.getOffset(i);
     IntVector size = deviceVars.getSizeVector(i);
     IntVector low = offset;
     IntVector high = offset + size;


     switch (deviceVars.getSizeOfDataType(i)) {
     case sizeof(int) : {
     GPUGridVariable<int> device_var;
     dws[dwIndex]->getGPUDW()->putContiguous(device_var, taskID.c_str(), dep->var->getName().c_str(),
     deviceVars.getPatchPointer(i)->getID(), deviceVars.getMatlIndx(i), 0,
     make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()), deviceVars.getSizeOfDataType(i), dynamic_cast<GridVariableBase*>(deviceVars.getVar(i)), true);

     break;
     } case sizeof(double) : {
     GPUGridVariable<double> device_var;
     dws[dwIndex]->getGPUDW()->putContiguous(device_var, taskID.c_str(), dep->var->getName().c_str(),
     deviceVars.getPatchPointer(i)->getID(), deviceVars.getMatlIndx(i), 0,
     make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()), deviceVars.getSizeOfDataType(i), dynamic_cast<GridVariableBase*>(deviceVars.getVar(i)), true);

     break;
     } case sizeof(GPUStencil7) : {
     GPUGridVariable<GPUStencil7> device_var;
     dws[dwIndex]->getGPUDW()->putContiguous(device_var, taskID.c_str(), dep->var->getName().c_str(),
     deviceVars.getPatchPointer(i)->getID(), deviceVars.getMatlIndx(i), 0,
     make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()), deviceVars.getSizeOfDataType(i), dynamic_cast<GridVariableBase*>(deviceVars.getVar(i)), true);

     break;
     } default : {
     SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + dep->var->getName(), __FILE__, __LINE__));
     }
     }

     //Now that this requires grid variable is copied onto the device,
     //we no longer need to hold onto this particular reference of the host side
     //version of it.  So go ahead and remove our reference to it. (Note: this was removed
     //as in some situations it was the last remaining reference to it, so it would remove it
     //for good, but the data warehouse was not aware that it was gone and tried to delete
     //it again when scrubbing.
     delete deviceVars.getVar(i);
     }
     }

     //now loop through the computes data where we just need to assign space (but not copy) on the device.
     for (unsigned int i = 0; i < deviceVars.numItems(); i++ ) {
     if (deviceVars.getDependency(i)->deptype != Task::Requires) {
     int dwIndex =  deviceVars.getDependency(i)->mapDataWarehouse();
     OnDemandDataWarehouseP dw = dws[dwIndex];
     const Task::Dependency* dep = deviceVars.getDependency(i);
     IntVector offset = deviceVars.getOffset(i);
     IntVector size = deviceVars.getSizeVector(i);
     IntVector low = offset;
     IntVector high = offset + size;
     switch (deviceVars.getSizeOfDataType(i)) {
     case sizeof(int) : {
     GPUGridVariable<int> device_var;
     dws[dwIndex]->getGPUDW()->putContiguous(device_var, taskID.c_str(), dep->var->getName().c_str(),
     deviceVars.getPatchPointer(i)->getID(), deviceVars.getMatlIndx(i), 0,
     make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()), deviceVars.getSizeOfDataType(i), NULL, false);

     break;
     } case sizeof(double) : {
     GPUGridVariable<double> device_var;
     dws[dwIndex]->getGPUDW()->putContiguous(device_var, taskID.c_str(), dep->var->getName().c_str(),
     deviceVars.getPatchPointer(i)->getID(), deviceVars.getMatlIndx(i), 0,
     make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()), deviceVars.getSizeOfDataType(i), NULL, false);

     break;
     } case sizeof(GPUStencil7) : {
     GPUGridVariable<GPUStencil7> device_var;
     dws[dwIndex]->getGPUDW()->putContiguous(device_var, taskID.c_str(), dep->var->getName().c_str(),
     deviceVars.getPatchPointer(i)->getID(), deviceVars.getMatlIndx(i), 0,
     make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()), deviceVars.getSizeOfDataType(i), NULL, false);

     break;
     } default : {
     SCI_THROW(InternalError("Unsupported GPUGridVariable type: " + dep->var->getName(), __FILE__, __LINE__));
     }
     }
     //At this point, space is still allocated on the host side for this computes grid
     //variable.  Do not delete deviceVars.getGridVar(i) yet, we want to leave a
     //reference open and space open until we copy it back.  In processD2HCopies() we
     //remove the reference to it, possibly deleting it if the reference count is zero.
     }
     }

     for (int dwIndex = 0; dwIndex < (int)dws.size(); dwIndex++) {
     if (deviceVars.getSizeForDataWarehouse(dwIndex) > 0) {
     //copy it to the device.  Any data warehouse that has requires should be doing copies here.
     //copyDataHostToDevice knows how to copy only the initialized portion of the array, and not
     //everything including the uninitialized garbage data in the array.
     cudaError_t retVal = dws[dwIndex]->getGPUDW()->copyDataHostToDevice(taskID.c_str(), dtask->getCUDAStream());
     if (retVal != cudaSuccess) {
     SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
     }
     }
     }
     */
  } //end if for contiguous or not if block > 0

}


void UnifiedScheduler::prepareTaskVarsIntoTaskDW(DetailedTask* dtask,
    DeviceGridVariables& taskVars) {
  //Copy all task variables metadata into the Task GPU DW.
  //All necessary metadata information must already exist in the host-side GPU DWs.
  map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & taskVarMap =
      taskVars.getMap();
  for (map<DeviceGridVariableInfo::LabelPatchMatlLevelDw,
      DeviceGridVariableInfo>::iterator it = taskVarMap.begin();
      it != taskVarMap.end(); ++it) {
    int dwIndex = it->second.dep->mapDataWarehouse();
    switch (it->second.dep->var->typeDescription()->getType()) {
    case TypeDescription::PerPatch: {
      GPUPerPatchBase* patchVar = OnDemandDataWarehouse::createGPUPerPatch(
          it->second.sizeOfDataType);
      GPUDataWarehouse* gpudw = dws[dwIndex]->getGPUDW(it->second.whichGPU);
      int patchID = it->first.patchID;
      int matlIndx = it->first.matlIndx;
      int levelIndx = it->first.levelIndx;
      gpudw->get(*patchVar, it->second.dep->var->getName().c_str(), patchID,
          matlIndx, levelIndx);
      dtask->getTaskGpuDataWarehouse((Task::WhichDW) dwIndex)->put(*patchVar,
          OnDemandDataWarehouse::getTypeDescriptionSize(
              it->second.dep->var->typeDescription()->getSubType()->getType()),
          it->second.dep->var->getName().c_str(), patchID, matlIndx,
          levelIndx, 0);
      delete patchVar;
    }
      break;
    case TypeDescription::CCVariable:
    case TypeDescription::NCVariable:
    case TypeDescription::SFCXVariable:
    case TypeDescription::SFCYVariable:
    case TypeDescription::SFCZVariable: {

      GPUGridVariableBase* gpuGridVar =
          OnDemandDataWarehouse::createGPUGridVariable(
              it->second.sizeOfDataType);
      GPUDataWarehouse* gpudw = dws[dwIndex]->getGPUDW(it->second.whichGPU);
      int patchID = it->first.patchID;
      int matlIndx = it->first.matlIndx;
      int levelIndx = it->first.levelIndx;
      gpudw->get(*gpuGridVar, it->second.dep->var->getName().c_str(), patchID,
          matlIndx, levelIndx);
      dtask->getTaskGpuDataWarehouse((Task::WhichDW) dwIndex)->put(
          *gpuGridVar,
          OnDemandDataWarehouse::getTypeDescriptionSize(
              it->second.dep->var->typeDescription()->getSubType()->getType()),
          it->second.dep->var->getName().c_str(), patchID, matlIndx,
          levelIndx, (GPUDataWarehouse::GhostType) (it->second.dep->gtype),
          it->second.dep->numGhostCells);
      delete gpuGridVar;
    }
      break;
    default: {
      cerrLock.lock();
      cerr << "Variable " << it->second.dep->var->getName() << " is of a type that is not supported on GPUs yet." << endl;
      cerrLock.unlock();
    }
    }
  } //end if task dw preparation
}

void UnifiedScheduler::prepareGhostCellsIntoTaskDW(DetailedTask* dtask,
    DeviceGhostCells& ghostVars) {

  //Now tell the Task DWs about any ghost cells they will need to process.
  //map<GPUDataWarehouse::labelPatchMatlLevel, DeviceGridVariableInfo> & ghostVarMap = ghostVars.getMap();
  //for (map<GPUDataWarehouse::labelPatchMatlLevel, DeviceGridVariableInfo>::iterator it=ghostVarMap.begin(); it!=ghostVarMap.end(); ++it) {

  for (unsigned int i = 0; i < ghostVars.numItems(); i++) {
    //If the neighbor is valid on the GPU, we just send in from and to coordinates
    //and call a kernel to copy those coordinates
    //If it's not valid on the GPU, we copy in the grid var and send in from and to coordinates
    //and call a kernel to copy those coordinates.

    int dwIndex = ghostVars.getDwIndex(i);
    int sourceDeviceNum = ghostVars.getSourceDeviceNum(i);
    int destDeviceNum = ghostVars.getDestDeviceNum(i);

    //We can copy it manually internally within the device via a kernel.
    //This apparently goes faster overall
    IntVector low = ghostVars.getLow(i);
    IntVector high = ghostVars.getHigh(i);
    IntVector virtualOffset = ghostVars.getVirtualOffset(i);
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << myRankThread()
            << " prepareGpuDependencies() - Preparing ghost cell upcoming copy from patch "
            << ghostVars.getSourcePatchPointer(i)->getID() << " to patch "
            << ghostVars.getDestPatchPointer(i)->getID()
            << " in the Task GPU DW " << dwIndex << endl;
      }
      cerrLock.unlock();
    }

    //This task DW may not have an adjacent variable containing the ghost cell data for this node
    //as previously we *only* let this Task DW know about variables in patches assigned to this task.
    //So if any ghost cell data is in other patches not assigned to this task, just let this task know
    //of it.
    dtask->getTaskGpuDataWarehouse((Task::WhichDW) dwIndex)->putGhostCell(
        ghostVars.getLabelName(i), ghostVars.getSourcePatchPointer(i)->getID(),
        ghostVars.getDestPatchPointer(i)->getID(), ghostVars.getMatlIndx(i),
        ghostVars.getLevelIndx(i), make_int3(low.x(), low.y(), low.z()),
        make_int3(high.x(), high.y(), high.z()),
        make_int3(virtualOffset.x(), virtualOffset.y(), virtualOffset.z()),
        false, NULL, make_int3(0, 0, 0), make_int3(0, 0, 0), 0);

  }  //end if ghost cell preparation
}

void UnifiedScheduler::markDeviceRequiresDataAsValid(DetailedTask* dtask) {
  //Go through device requires vars and mark them as valid on the device.  They are either already
  //valid because they were there previously.  Or they just got copied in and the stream completed.

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::markDeviceRequiresDataAsValid"); TAU_PROFILE("UnifiedScheduler::markDeviceRequiresDataAsValid()", " ", TAU_USER);

  void* host_ptr = NULL;    // host base pointer to raw data
  void* device_ptr = NULL;    // host base pointer to raw data
  size_t host_bytes = 0;    // raw byte count to copy to the device
  //The only thing we need to process is the requires.
  const Task* task = dtask->getTask();
  for (const Task::Dependency* dependantVar = task->getRequires();
      dependantVar != 0; dependantVar = dependantVar->next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(
        dtask->getMaterials());
    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = dependantVar->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(
          GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      if (gpudw != NULL) {

        for (int j = 0; j < numMatls; j++) {
          int patchID = patches->get(i)->getID();
          int matlID = matls->get(j);
          const Level* level = getLevel(dtask->getPatches());
          int levelID = level->getID();
          if (gpudw->exist(dependantVar->var->getName().c_str(), patchID,
              matlID, levelID)) {
            gpudw->setValidOnGPU(dependantVar->var->getName().c_str(), patchID,
                matlID, levelID);
          }
        }
      }
    }

  }
}

void UnifiedScheduler::markDeviceComputesDataAsValid(DetailedTask* dtask) {
  //Go through device computes vars and mark them as valid on the device.

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::markDeviceComputesDataAsValid"); TAU_PROFILE("UnifiedScheduler::markDeviceComputesDataAsValid()", " ", TAU_USER);

  void* host_ptr = NULL;    // host base pointer to raw data
  void* device_ptr = NULL;    // host base pointer to raw data
  size_t host_bytes = 0;    // raw byte count to copy to the device
  //The only thing we need to process is the requires.
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp =
      comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(
        dtask->getMaterials());
    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(
          GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      if (gpudw != NULL) {
        for (int j = 0; j < numMatls; j++) {
          int patchID = patches->get(i)->getID();
          int matlID = matls->get(j);
          const Level* level = getLevel(dtask->getPatches());
          int levelID = level->getID();
          if (gpudw->exist(comp->var->getName().c_str(), patchID, matlID,
              levelID)) {
            gpudw->setValidOnGPU(comp->var->getName().c_str(), patchID, matlID,
                levelID);
          }
        }
      }
    }
  }
}

void UnifiedScheduler::markHostRequiresDataAsValid(DetailedTask* dtask) {
  //Data has been copied from the device to the host.  The stream has completed.
  //Go through all variables that this CPU task depends on and mark them as valid on the CPU

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::markHostRequiresDataAsValid"); TAU_PROFILE("UnifiedScheduler::markHostRequiresDataAsValid()", " ", TAU_USER);

  void* host_ptr = NULL;    // host base pointer to raw data
  void* device_ptr = NULL;    // host base pointer to raw data
  size_t host_bytes = 0;    // raw byte count to copy to the device
  //The only thing we need to process is the requires.
  const Task* task = dtask->getTask();
  for (const Task::Dependency* dependantVar = task->getRequires();
      dependantVar != 0; dependantVar = dependantVar->next) {
    if (dtask->getPatches() != NULL) {
      constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(
          dtask->getPatches());
      constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(
          dtask->getMaterials());
      // this is so we can allocate persistent events and streams to distribute when needed
      //   one stream and one event per variable per H2D copy (numPatches * numMatls)

      int numPatches = patches->size();
      int numMatls = matls->size();
      int dwIndex = dependantVar->mapDataWarehouse();
      OnDemandDataWarehouseP dw = dws[dwIndex];
      for (int i = 0; i < numPatches; i++) {
        GPUDataWarehouse * gpudw = dw->getGPUDW(
            GpuUtilities::getGpuIndexForPatch(patches->get(i)));
        if (gpudw != NULL) {
          for (int j = 0; j < numMatls; j++) {
            int patchID = patches->get(i)->getID();
            int matlID = matls->get(j);
            const Level* level = getLevel(dtask->getPatches());
            int levelID = level->getID();
            if (gpudw->exist(dependantVar->var->getName().c_str(), patchID,
                matlID, levelID)) {
              gpudw->setValidOnCPU(dependantVar->var->getName().c_str(),
                  patchID, matlID, levelID);
            }
          }
        }
      }
    }
  }
}

bool UnifiedScheduler::initiateD2H(DetailedTask* dtask) {
  //Request that all contiguous device arrays from the device be sent to their contiguous host array counterparts.
  //We only copy back the data needed for an upcoming task.  If data isn't needed, it can stay on the device and
  //potentially even die on the device

  //Returns true if no device data is required, thus allowing a CPU task to immediately proceed.

  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::initiateD2H"); TAU_PROFILE("UnifiedScheduler::initiateD2H()", " ", TAU_USER);

  void* host_ptr = NULL;    // host base pointer to raw data
  void* device_ptr = NULL;    // host base pointer to raw data
  size_t host_bytes = 0;    // raw byte count to copy to the device

  const Task* task = dtask->getTask();

  //The only thing we need to process is the requires.

  //Gather up all possible dependents and remove duplicate (we don't want to transfer some variables twice)
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires();
      dependantVar != 0; dependantVar = dependantVar->next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(
        dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->var->getName().c_str(),
            patches->get(i)->getID(), matls->get(j), Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(
              std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(
                  lpmd, dependantVar));
        }
      }
    }
  }

  //Go through each unique dependent var and see if we should queue up a D2H copy
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* dependantVar = varIter->second;
    //for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(
        dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = dependantVar->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];
    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(
          GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      const int patchID = patches->get(i)->getID();
      const Level* level = getLevel(dtask->getPatches());
      const int levelID = level->getID();
      if (gpudw != NULL) {
        for (int j = 0; j < numMatls; j++) {
          const int matlID = matls->get(j);
          TypeDescription::Type type =
              dependantVar->var->typeDescription()->getType();
          switch (type) {
          case TypeDescription::CCVariable:
          case TypeDescription::NCVariable:
          case TypeDescription::SFCXVariable:
          case TypeDescription::SFCYVariable:
          case TypeDescription::SFCZVariable: {

            if (gpu_stats.active()) {
              cerrLock.lock();
              {
                gpu_stats << myRankThread() << " InitiateD2H, task: " << dtask->getName()
                    << ".  Checking if we should copy of \""
                    << dependantVar->var->getName() << "\" Patch " << patchID
                    << " Material " << matlID << " on thread "
                    << Thread::self()->myid() << endl;
              }
              cerrLock.unlock();
            }

            //size the host var to be able to fit all room needed.
            IntVector host_low, host_high, host_lowOffset, host_highOffset,
                host_offset, host_size, host_strides;
            Patch::VariableBasis basis = Patch::translateTypeToBasis(type,
                false);
            Patch::getGhostOffsets(type, dependantVar->gtype,
                dependantVar->numGhostCells, host_lowOffset, host_highOffset);
            patches->get(i)->computeExtents(basis,
                dependantVar->var->getBoundaryLayer(), host_lowOffset,
                host_highOffset, host_low, host_high);
            int dwIndex = dependantVar->mapDataWarehouse();
            OnDemandDataWarehouseP dw = dws[dwIndex];

            //TODO: Add check to make sure the data isn't valid on the CPU.
            //if (gpudw->exist(dependantVar->var->getName().c_str(),
            //    patchID,
            //    matlID,
            //    make_int3(host_high.x() - host_low.x(), host_high.y() - host_low.y(), host_high.z() - host_low.z()),
            //    make_int3(host_low.x(), host_low.y(), host_low.z()),
            //    true) &&
            //See if it exists on the GPU
            if (gpudw->getValidOnGPU(dependantVar->var->getName().c_str(),
                patchID, matlID, levelID)) {

              //It's possible the computes data may contain ghost cells.  But a task needing to get the data
              //out of the GPU may not know this.  It may just want the var data.
              //This creates a dilemma, as the GPU var is sized differently than the CPU var.
              //So ask the GPU what size it has for the var.  Size the CPU var to match so it can copy all GPU data in.
              //When the GPU->CPU copy is done, then we need to resize the CPU var if needed to match
              //what the CPU is expecting it to be.
              //GPUGridVariableBase* gpuGridVar;

              //gpudw->get(*gpuGridVar, dependantVar->var->getName().c_str(), patchID, matlID);
              int3 low;
              int3 high;
              int3 size;
              GPUDataWarehouse::GhostType tempgtype;
              Ghost::GhostType gtype;
              int numGhostCells;
              gpudw->getSizes(low, high, size, tempgtype, numGhostCells,
                  dependantVar->var->getName().c_str(), patchID, matlID,
                  levelID);
              gtype = (Ghost::GhostType) tempgtype;
              //gpudw->getSizes(low, high, size)
              //IntVector templow(low.x, low.y, low.z);
              //IntVector temphigh(high.x, high.y, high.z);
              GridVariableBase* gridVar =
                  dynamic_cast<GridVariableBase*>(dependantVar->var->typeDescription()->createInstance());
              //gridVar->rewindow(templow, temphigh);
              //dw->put(*gridVar, dependantVar->var, matlID, patches->get(i), true);
              dw->allocateAndPut(*gridVar, dependantVar->var, matlID,
                  patches->get(i), gtype, numGhostCells);
              gridVar->getSizes(host_low, host_high, host_offset, host_size,
                  host_strides);
              host_ptr = gridVar->getBasePointer();
              host_bytes = gridVar->getDataSize();

              // copy the computes data back to the host
              d2hComputesLock_.writeLock();
              {

                int3 device_offset;
                int3 device_size;

                switch (host_strides.x()) {
                case sizeof(int): {
                  GPUGridVariable<int> device_var;
                  gpudw->get(device_var, dependantVar->var->getName().c_str(),
                      patchID, matlID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                case sizeof(double): {
                  GPUGridVariable<double> device_var;
                  gpudw->get(device_var, dependantVar->var->getName().c_str(),
                      patchID, matlID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                case sizeof(GPUStencil7): {
                  GPUGridVariable<GPUStencil7> device_var;
                  gpudw->get(device_var, dependantVar->var->getName().c_str(),
                      patchID, matlID);
                  device_var.getArray3(device_offset, device_size, device_ptr);
                  break;
                }
                default: {
                  SCI_THROW(
                      InternalError("Unsupported GPUGridVariable type: " + dependantVar->var->getName(), __FILE__, __LINE__));
                }
                }

                // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
                if (device_offset.x == host_low.x()
                    && device_offset.y == host_low.y()
                    && device_offset.z == host_low.z()
                    && device_size.x == host_size.x()
                    && device_size.y == host_size.y()
                    && device_size.z == host_size.z()) {

                  // The following is only efficient for large single copies. With multiple smaller copies
                  // the faster PCIe transfers never outweigh the CUDA API latencies. We can revive this idea
                  // once we're doing large, single, aggregated cuda memcopies. [APH]
                  //                const bool pinned = (*(pinnedHostPtrs.find(host_ptr)) == host_ptr);
                  //                if (!pinned) {
                  //                  // pin/page-lock host memory for H2D cudaMemcpyAsync
                  //                  // memory returned using cudaHostRegisterPortable flag will be considered pinned by all CUDA contexts
                  //                  CUDA_RT_SAFE_CALL(retVal = cudaHostRegister(host_ptr, host_bytes, cudaHostRegisterPortable));
                  //                  if (retVal == cudaSuccess) {
                  //                    pinnedHostPtrs.insert(host_ptr);
                  //                  }
                  //                }

                  if (gpu_stats.active()) {
                    cerrLock.lock();
                    {
                      gpu_stats << myRankThread() << " InitiateD2H - Copy of \""
                          << dependantVar->var->getName() << "\", size = "
                          << std::dec << host_bytes << " to " << std::hex
                          << host_ptr << " from " << std::hex << device_ptr
                          << ", using stream " << std::hex
                          << dtask->getCUDAStream() << std::dec << endl;
                    }
                    cerrLock.unlock();
                  }
                  cudaError_t retVal;

                  //Since we know we need a stream, obtain one.
                  //readyTask->setCUDAStream(getCudaStream(dtask->getDeviceNum()));
                  int deviceNum = GpuUtilities::getGpuIndexForPatch(
                      patches->get(i));
                  cudaStream_t* stream = dtask->getCUDAStream(deviceNum);
                  if (stream == NULL) {
                    stream = getCudaStream(deviceNum);
                    dtask->setCUDAStream(deviceNum, stream);
                  }

                  //printf("***InitiateD2H invoked***\n");
                  CUDA_RT_SAFE_CALL(
                      retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes,
                          cudaMemcpyDeviceToHost, *stream));
                  if (retVal == cudaErrorLaunchFailure) {
                    SCI_THROW(
                        InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                  } else {
                    CUDA_RT_SAFE_CALL(retVal);
                  }
                }
              }
              d2hComputesLock_.writeUnlock();
              delete gridVar;
            }
            break;
          }
          case TypeDescription::PerPatch: {
            if (gpudw->getValidOnGPU(dependantVar->var->getName().c_str(),
                patchID, matlID, levelID)) {
              PerPatchBase* hostPerPatchVar =
                  dynamic_cast<PerPatchBase*>(dependantVar->var->typeDescription()->createInstance());
              dw->put(*hostPerPatchVar, dependantVar->var, matlID,
                  patches->get(i));

              //gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);
              host_ptr = hostPerPatchVar->getBasePointer();
              host_bytes = hostPerPatchVar->getDataSize();

              // copy the computes data back to the host
              d2hComputesLock_.writeLock();

              GPUPerPatchBase* gpuPerPatchVar =
                  OnDemandDataWarehouse::createGPUPerPatch(host_bytes);
              device_ptr = gpuPerPatchVar->getVoidPointer();

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread() << "Post D2H copy of \""
                      << dependantVar->var->getName() << "\", size = "
                      << std::dec << host_bytes << " to " << std::hex
                      << host_ptr << " from " << std::hex << device_ptr
                      << ", using stream " << std::hex << dtask->getCUDAStream()
                      << std::dec << endl;
                }
                cerrLock.unlock();
              }
              cudaError_t retVal;

              //Since we know we need a stream, obtain one.
              //readyTask->setCUDAStream(getCudaStream(dtask->getDeviceNum()));
              int deviceNum = GpuUtilities::getGpuIndexForPatch(
                  patches->get(i));
              cudaStream_t* stream = dtask->getCUDAStream(deviceNum);
              if (stream == NULL) {
                stream = getCudaStream(deviceNum);
                dtask->setCUDAStream(deviceNum, stream);
              }

              CUDA_RT_SAFE_CALL(
                  retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes,
                      cudaMemcpyDeviceToHost, *stream));
              if (retVal == cudaErrorLaunchFailure) {
                SCI_THROW(
                    InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
              } else {
                CUDA_RT_SAFE_CALL(retVal);
              }

              d2hComputesLock_.writeUnlock();
              delete hostPerPatchVar;
            }

            break;
          }
          default: {
            cerrLock.lock();
            cerr << "Variable " << dependantVar->var->getName() << " is of a type that is not supported on GPUs yet." << endl;
            cerrLock.unlock();
          }
          }
        }
      }
    }
  }
  //TODO, figure out contiguous data in this model
  //Now do any contiguous data.
  //cudaError_t retVal;
  //int device = dtask->getDeviceNum();
  //CUDA_RT_SAFE_CALL(retVal = OnDemandDataWarehouse::uintahSetCudaDevice(device);

  //string taskID = dtask->getName();
  //retVal = dws[dwmap[Task::NewDW]]->getGPUDW()->copyDataDeviceToHost(taskID.c_str(), dtask->getCUDAStream());
  //if (retVal != cudaSuccess) {
  //   printf("Error code is %d\n", retVal);
  //   SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
  //}

}
void UnifiedScheduler::copyAllDataD2H(DetailedTask* dtask) {

  //Request that all contiguous device arrays from the device be sent to their contiguous host array counterparts.
  //We only copy back the computes data, the requires data doesn't need to be copied
  //back to the host.
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::copyAllDataD2H"); TAU_PROFILE("UnifiedScheduler::copyAllDataD2H()", " ", TAU_USER);
  void* host_ptr = NULL;    // host base pointer to raw data
  void* device_ptr = NULL;    // host base pointer to raw data
  size_t host_bytes = 0;    // raw byte count to copy to the device

  //The only thing we need to process  is the computes.
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp =
      comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(
        dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];
    for (int i = 0; i < numPatches; i++) {
      const int patchID = patches->get(i)->getID();
      const Level* level = getLevel(dtask->getPatches());
      const int levelID = level->getID();
      for (int j = 0; j < numMatls; j++) {
        const int matlID = matls->get(j);
        TypeDescription::Type type = comp->var->typeDescription()->getType();
        switch (type) {
        case TypeDescription::CCVariable:
        case TypeDescription::NCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: {
          //See if it exists on the GPU, not as part of a contiguous array

          IntVector host_low, host_high, host_lowOffset, host_highOffset,
              host_offset, host_size, host_strides;
          Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
          Patch::getGhostOffsets(type, comp->gtype, comp->numGhostCells,
              host_lowOffset, host_highOffset);
          patches->get(i)->computeExtents(basis, comp->var->getBoundaryLayer(),
              host_lowOffset, host_highOffset, host_low, host_high);
          int dwIndex = comp->mapDataWarehouse();
          OnDemandDataWarehouseP dw = dws[dwIndex];
          if (dw->getGPUDW()->exist(comp->var->getName().c_str(), patchID,
              matlID, levelID,
              make_int3(host_high.x() - host_low.x(),
                  host_high.y() - host_low.y(), host_high.z() - host_low.z()),
              make_int3(host_low.x(), host_low.y(), host_low.z()), true)) {

            GridVariableBase* gridVar =
                dynamic_cast<GridVariableBase*>(comp->var->typeDescription()->createInstance());
            dw->allocateAndPut(*gridVar, comp->var, matlID, patches->get(i),
                comp->gtype, comp->numGhostCells);
            gridVar->getSizes(host_low, host_high, host_offset, host_size,
                host_strides);
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
              case sizeof(int): {
                GPUGridVariable<int> device_var;
                dw->getGPUDW()->get(device_var, comp->var->getName().c_str(),
                    patchID, matlID);
                device_var.getArray3(device_offset, device_size, device_ptr);
                break;
              }
              case sizeof(double): {
                GPUGridVariable<double> device_var;
                dw->getGPUDW()->get(device_var, comp->var->getName().c_str(),
                    patchID, matlID);
                device_var.getArray3(device_offset, device_size, device_ptr);
                break;
              }
              case sizeof(GPUStencil7): {
                GPUGridVariable<GPUStencil7> device_var;
                dw->getGPUDW()->get(device_var, comp->var->getName().c_str(),
                    patchID, matlID);
                device_var.getArray3(device_offset, device_size, device_ptr);
                break;
              }
              default: {
                SCI_THROW(
                    InternalError("Unsupported GPUGridVariable type: " + comp->var->getName(), __FILE__, __LINE__));
              }
              }

              // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
              if (device_offset.x == host_low.x()
                  && device_offset.y == host_low.y()
                  && device_offset.z == host_low.z()
                  && device_size.x == host_size.x()
                  && device_size.y == host_size.y()
                  && device_size.z == host_size.z()) {

                // The following is only efficient for large single copies. With multiple smaller copies
                // the faster PCIe transfers never outweigh the CUDA API latencies. We can revive this idea
                // once we're doing large, single, aggregated cuda memcopies. [APH]
                //                const bool pinned = (*(pinnedHostPtrs.find(host_ptr)) == host_ptr);
                //                if (!pinned) {
                //                  // pin/page-lock host memory for H2D cudaMemcpyAsync
                //                  // memory returned using cudaHostRegisterPortable flag will be considered pinned by all CUDA contexts
                //                  CUDA_RT_SAFE_CALL(retVal = cudaHostRegister(host_ptr, host_bytes, cudaHostRegisterPortable));
                //                  if (retVal == cudaSuccess) {
                //                    pinnedHostPtrs.insert(host_ptr);
                //                  }
                //                }

                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << myRankThread() << " Post D2H copy of \"" << comp->var->getName()
                        << "\", size = " << std::dec << host_bytes << " to "
                        << std::hex << host_ptr << " from " << std::hex
                        << device_ptr << ", using stream " << std::hex
                        << dtask->getCUDAStream() << std::dec << endl;
                  }
                  cerrLock.unlock();
                }
                cudaError_t retVal;
                CUDA_RT_SAFE_CALL(
                    retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes,
                        cudaMemcpyDeviceToHost, *dtask->getCUDAStream()));
                if (retVal == cudaErrorLaunchFailure) {
                  SCI_THROW(
                      InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                } else {
                  //TODO: Handle setting it as valid on the GPU.
                  CUDA_RT_SAFE_CALL(retVal);
                }
              }
            }
            d2hComputesLock_.writeUnlock();
            delete gridVar;
          }

          break;
        }
        default: {
          cerrLock.lock();
	  cerr << "Variable " << comp->var->getName() << " is of a type that is not supported on GPUs yet." << endl;
          cerrLock.unlock();
        }
        }
      }
    }
  }

  //Now do any contiguous data.
  cudaError_t retVal;
  int device = dtask->getDeviceNum();
  OnDemandDataWarehouse::uintahSetCudaDevice(device);

  string taskID = dtask->getName();
  retVal = dws[dwmap[Task::NewDW]]->getGPUDW()->copyDataDeviceToHost(
      taskID.c_str(), dtask->getCUDAStream());
  if (retVal != cudaSuccess) {
    printf("Error code is %d\n", retVal);
    SCI_THROW(
        InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));

  }
}

void UnifiedScheduler::processD2HCopies(DetailedTask* dtask) {
  //Contiguous arrays from the device have now been copied to
  //corresponding contiguous arrays on the host.  Only the computes
  //portion has been copied.
  //So now copy the computes back from the host contiguous array back into
  //individual host grid vars.

  //Request that all contiguous device arrays form the device be sent to their contiguous host array counterparts.
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::processD2HCopies"); TAU_PROFILE("UnifiedScheduler::processD2HCopies()", " ", TAU_USER);

  //The only thing we need to process  is the computes.
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp =
      comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(
        dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(
        dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = dws[dwIndex];
    for (int i = 0; i < numPatches; i++) {
      const int patchID = patches->get(i)->getID();
      const Level* level = getLevel(dtask->getPatches());
      const int levelID = level->getID();
      for (int j = 0; j < numMatls; j++) {
        TypeDescription::Type type = comp->var->typeDescription()->getType();
        const int matlID = matls->get(j);
        switch (type) {
        case TypeDescription::CCVariable:
        case TypeDescription::NCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: {

          IntVector host_low, host_high, host_lowOffset, host_highOffset,
              host_offset, host_size, host_strides;
          Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
          Patch::getGhostOffsets(type, comp->gtype, comp->numGhostCells,
              host_lowOffset, host_highOffset);
          patches->get(i)->computeExtents(basis, comp->var->getBoundaryLayer(),
              host_lowOffset, host_highOffset, host_low, host_high);
          int dwIndex = comp->mapDataWarehouse();
          OnDemandDataWarehouseP dw = dws[dwIndex];
          if (dw->getGPUDW()->exist(comp->var->getName().c_str(), patchID,
              matlID, levelID,
              make_int3(host_high.x() - host_low.x(),
                  host_high.y() - host_low.y(), host_high.z() - host_low.z()),
              make_int3(host_low.x(), host_low.y(), host_low.z()), false,
              true)) {

            //this computes data existed on the device, but we don't have it on the host.  So make room for it on the host
            GridVariableBase* gridVar =
                dynamic_cast<GridVariableBase*>(comp->var->typeDescription()->createInstance());
            dw->allocateAndPut(*gridVar, comp->var, matls->get(j),
                patches->get(i), comp->gtype, comp->numGhostCells);
            //IntVector host_low, host_high, host_offset, host_size, host_strides;
            //gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);
            //TODO: Is this writelock needed?  Will there ever be more than one thread writing to this
            //specific label/patch/matl tuple?
            d2hComputesLock_.writeLock();
            switch (host_strides.x()) {
            case sizeof(int): {
              GPUGridVariable<int> device_var;
              dw->getGPUDW()->copyHostContiguousToHost(device_var, gridVar,
                  comp->var->getName().c_str(), patchID, matlID, levelID);
              break;
            }
            case sizeof(double): {
              GPUGridVariable<double> device_var;
              dw->getGPUDW()->copyHostContiguousToHost(device_var, gridVar,
                  comp->var->getName().c_str(), patchID, matlID, levelID);
              break;
            }
            case sizeof(GPUStencil7): {
              GPUGridVariable<GPUStencil7> device_var;
              dw->getGPUDW()->copyHostContiguousToHost(device_var, gridVar,
                  comp->var->getName().c_str(), patchID, matlID, levelID);
              break;
            }
            }
            delete gridVar;
            d2hComputesLock_.writeUnlock();
            //TODO: This should happen in a later function once we have confirmed we have copied it out.
            dw->getGPUDW()->setValidOnCPU(comp->var->getName().c_str(), patchID,
                matlID, levelID);
          }
        }
          break;
        default: {
          cerrLock.lock();
          cerr << "This variable's type is not implemented" << endl;
          cerrLock.unlock();
        }
        }
      }
    }
  }
}

//______________________________________________________________________
//

void UnifiedScheduler::createCudaStreams(int device, int numStreams /* = 1 */) {
  cudaError_t retVal;

  idleStreamsLock_.writeLock();
  {
    OnDemandDataWarehouse::uintahSetCudaDevice(device);
    for (int j = 0; j < numStreams; j++) {
      cudaStream_t* stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));
      idleStreams[device].push(stream);

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread() << " Created CUDA stream " << std::hex
              << stream << " on device " << std::dec << device << std::endl;
        }
        cerrLock.unlock();
      }
      cerrLock.unlock();
    }
  }
  idleStreamsLock_.writeUnlock();
}

//______________________________________________________________________
//

void UnifiedScheduler::freeCudaStreams() {
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
        gpu_stats << myRankThread() << " Deallocating " << totalStreams
            << " total CUDA stream(s) for " << numQueues << " device(s)"
            << std::endl;
      }
      cerrLock.unlock();
    }

    for (size_t i = 0; i < numQueues; i++) {
      //TODO: Bug, not every device is going to be assigned, or even assigned in order.
      //the idleStreams is just a vector. - BRP 7/10/2015
      OnDemandDataWarehouse::uintahSetCudaDevice(i);

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread() << " Deallocating "
              << idleStreams[i].size() << " CUDA stream(s) on device " << retVal
              << std::endl;
        }
        cerrLock.unlock();
      }

      while (!idleStreams[i].empty()) {
        cudaStream_t* stream = idleStreams[i].front();
        idleStreams[i].pop();
        CUDA_RT_SAFE_CALL(retVal = cudaStreamDestroy(*stream));
        free(stream);
      }
    }
  }
  idleStreamsLock_.writeUnlock();
}

//______________________________________________________________________
//

cudaStream_t *
UnifiedScheduler::getCudaStream(int device) {
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
          gpu_stats << myRankThread() << " Issued CUDA stream " << std::hex
              << stream << " on device " << std::dec << device << std::endl;
          cerrLock.unlock();
        }
      }
    } else {  // shouldn't need any more than the queue capacity, but in case
      OnDemandDataWarehouse::uintahSetCudaDevice(device);
      // this will get put into idle stream queue and ultimately deallocated after final timestep
      stream = ((cudaStream_t*) malloc(sizeof(cudaStream_t)));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
              << " Needed to create 1 additional CUDA stream " << std::hex
              << stream << " for device " << std::dec << device << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }
  idleStreamsLock_.writeUnlock();

  return stream;
}

void UnifiedScheduler::reclaimCudaStreams(DetailedTask* dtask) {
  idleStreamsLock_.writeLock();
  // reclaim DetailedTask streams
  std::set<int> deviceNums = dtask->getDeviceNums();
  for (std::set<int>::iterator iter = deviceNums.begin();
      iter != deviceNums.end(); ++iter) {
    cudaStream_t* stream = dtask->getCUDAStream(*iter);
    if (stream != NULL) {
      idleStreams[*iter].push(stream);
      dtask->setCUDAStream(*iter, NULL);
    }
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << myRankThread() << " Reclaimed CUDA stream " << std::hex
            << stream << " on device " << std::dec << *iter << std::endl;
      }
      cerrLock.unlock();
    }
  }
  idleStreamsLock_.writeUnlock();
}

/*
 cudaError_t
 UnifiedScheduler::unregisterPageLockedHostMem()
 {
 cudaError_t retVal;
 std::set<void*>::iterator iter;

 // unregister the page-locked host memory
 for (iter = pinnedHostPtrs.begin(); iter != pinnedHostPtrs.end(); iter++) {
 CUDA_RT_SAFE_CALL(retVal = cudaHostUnregister(*iter));
 }
 pinnedHostPtrs.clear();

 return retVal;
 } */

void UnifiedScheduler::createTaskGpuDWs(DetailedTask * task,
    const DeviceGridVariables& taskVars, const DeviceGhostCells& ghostVars,
    const int device_id) {
  //Create GPU datawarehouses for this specific task only.  They will get copied into the GPU.
  //This is sizing these datawarehouses dynamically and doing it all in only one alloc per datawarehouse.
  //See the bottom of the GPUDataWarehouse.h for more information.

  unsigned int numItemsInDW = taskVars.getTotalVars(Task::OldDW) + ghostVars.getNumGhostCellCopies(Task::OldDW);
  if (numItemsInDW > 0) {
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << myRankThread()
           << " Creating an old Task GPU DW for task" <<  task->getTask()->getName()
           << " to contain " << taskVars.getTotalVars(Task::OldDW)
           << " task variables and " << ghostVars.getNumGhostCellCopies(Task::OldDW)
           << " ghost cell copies." << endl;
      }
      cerrLock.unlock();
    }

    size_t objectSizeInBytes = sizeof(GPUDataWarehouse)
        - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS
        + sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;
    void *placementNewBuffer = malloc(objectSizeInBytes);
    GPUDataWarehouse* old_taskGpuDW = new (placementNewBuffer) GPUDataWarehouse(
        device_id, placementNewBuffer);
    old_taskGpuDW->setDebug(gpudbg.active());
    //GPUDataWarehouse* old_taskGpuDW = (GPUDataWarehouse *) malloc(objectSizeInBytes);
    old_taskGpuDW->init_device(objectSizeInBytes);
    task->setTaskGpuDataWarehouse(Task::OldDW, old_taskGpuDW);
  }

  numItemsInDW = taskVars.getTotalVars(Task::NewDW) + ghostVars.getNumGhostCellCopies(Task::NewDW);
  if (numItemsInDW > 0) {
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << myRankThread()
           << " Creating an new Task GPU DW for task" <<  task->getTask()->getName()
           << " to contain " << taskVars.getTotalVars(Task::NewDW)
           << " task variables and " << ghostVars.getNumGhostCellCopies(Task::NewDW)
           << " ghost cell copies." << endl;
      }
      cerrLock.unlock();
    }
    size_t objectSizeInBytes = sizeof(GPUDataWarehouse)
        - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS
        + sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;
    void *placementNewBuffer = malloc(objectSizeInBytes);
    GPUDataWarehouse* new_taskGpuDW = new (placementNewBuffer) GPUDataWarehouse(
        device_id, placementNewBuffer);
    //GPUDataWarehouse* new_taskGpuDW = (GPUDataWarehouse *) malloc(objectSizeInBytes);
    new_taskGpuDW->setDebug(gpudbg.active());
    new_taskGpuDW->init_device(objectSizeInBytes);
    task->setTaskGpuDataWarehouse(Task::NewDW, new_taskGpuDW);
  }
  //Determine the sizes of each GPU Datawarehouse.  We want these DWs sized as small as possible to
  //minimize copy times to the GPU.



}

void UnifiedScheduler::assignDevice(DetailedTask* task) {

  // Figure out which device this patch was assigned to.
  // If a task has multiple patches, then assign all.  Most tasks should
  // only end up on one device.  Only tasks like data archiver's output variables
  // work on multiple patches which can be on multiple devices.
  std::map<const Patch *, int>::iterator it;
  for (int i = 0; i < task->getPatches()->size(); i++) {
    const Patch* patch = task->getPatches()->get(i);
    int index = GpuUtilities::getGpuIndexForPatch(patch);
    if (index >= 0) {
      task->assignDevice(index);
    } else {
      cerrLock.lock();
      cerr << "Could not find the assigned GPU for this patch." << endl;
      cerrLock.unlock();
      //For some reaosn, this patch wasn't assigned a GPU.  Give it the zeroth GPU.
      //readyTask->assignDevice(0);
    }
  }
}

void UnifiedScheduler::assignDevicesAndStreams(DeviceGhostCells& ghostVars, DetailedTask* task) {
  //Go through the ghostVars collection and look at the patch where all ghost cells are going.
  set<int> & destinationDevices = ghostVars.getDestinationDevices();
  for (set<int>::iterator iter=destinationDevices.begin(); iter != destinationDevices.end(); ++iter) {
    task->assignDevice(*iter);
    task->setCUDAStream(getCudaStream(*iter));
  }
}

void UnifiedScheduler::findIntAndExtGpuDependencies(
    DeviceGridVariables& deviceVars, DeviceGridVariables& taskVars,
    DeviceGhostCells& ghostVars, DetailedTask* task, int iteration, int t_id) {
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::findIntAndExtGpuDependencies");

  if (gpu_stats.active()) {
    cerrLock.lock();
    gpu_stats << myRankThread() << " findIntAndExtGpuDependencies - task "
        << *task << '\n';
    cerrLock.unlock();
  }

  // Prepare internal dependencies.  Only makes sense if we have GPUs that we are using.
//    if (task->getTask()->usesDevice()) {
  if (Uintah::Parallel::usingDevice()) {
    //If we have ghost cells coming from a GPU to another GPU on the same node
    //This does not cover going to another node (the loop below handles external
    //dependencies) That is handled in initiateH2D()
    //This does not handle preparing from a CPU to a same node GPU.  That is handled in initiateH2D()
    for (DependencyBatch* batch = task->getInternalComputes(); batch != 0;
        batch = batch->comp_next) {
      for (DetailedDep* req = batch->head; req != 0; req = req->next) {
        if ((req->condition == DetailedDep::FirstIteration && iteration > 0)
            || (req->condition == DetailedDep::SubsequentIterations
                && iteration == 0)
            || (notCopyDataVars_.count(req->req->var->getName()) > 0)) {
          // See comment in DetailedDep about CommCondition
          if (gpu_stats.active()) {
            gpu_stats << myRankThread()
                << "   Preparing GPU dependencies, ignoring conditional send for "
                << *req << std::endl;
          }
          continue;
        }
        // if we send/recv to an output task, don't send/recv if not an output timestep
        if (req->toTasks.front()->getTask()->getType() == Task::Output
            && !oport_->isOutputTimestep() && !oport_->isCheckpointTimestep()) {
          if (gpu_stats.active()) {
            gpu_stats << myRankThread()
                << "   Preparing GPU dependencies, ignoring non-output-timestep send for "
                << *req << std::endl;
          }
          continue;
        }
        OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();

        const VarLabel* posLabel;
        OnDemandDataWarehouse* posDW;

        // the load balancer is used to determine where data was in the old dw on the prev timestep -
        // pass it in if the particle data is on the old dw
        LoadBalancer* lb = 0;

        if (!reloc_new_posLabel_ && parentScheduler_) {
          posDW =
              dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
          posLabel = parentScheduler_->reloc_new_posLabel_;
        } else {
          // on an output task (and only on one) we require particle variables from the NewDW
          if (req->toTasks.front()->getTask()->getType() == Task::Output) {
            posDW =
                dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
          } else {
            posDW =
                dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
            lb = getLoadBalancer();
          }
          posLabel = reloc_new_posLabel_;
        }
        //Invoke Kernel to copy this range out of the GPU.
        prepareGpuDependencies(deviceVars, taskVars, ghostVars, task, batch,
            posLabel, dw, posDW, req, lb,
            DeviceGhostCells::anotherDeviceSameMpiRank);
      }
    }  // end for (DependencyBatch * batch = task->getInteranlComputes() )

    // Prepare external dependencies.  The only thing that needs to be prepared is
    // getting ghost cell data from a GPU into a flat array and copied to host memory
    // so that the MPI engine can treat it normally.
    // That means this handles GPU->other node GPU and GPU->other node CPU.
    //
    for (DependencyBatch* batch = task->getComputes(); batch != 0;
        batch = batch->comp_next) {
      for (DetailedDep* req = batch->head; req != 0; req = req->next) {
        if ((req->condition == DetailedDep::FirstIteration && iteration > 0)
            || (req->condition == DetailedDep::SubsequentIterations
                && iteration == 0)
            || (notCopyDataVars_.count(req->req->var->getName()) > 0)) {
          // See comment in DetailedDep about CommCondition
          if (gpu_stats.active()) {
            gpu_stats << myRankThread()
                << "   Preparing GPU dependencies, ignoring conditional send for "
                << *req << std::endl;
          }
          continue;
        }
        // if we send/recv to an output task, don't send/recv if not an output timestep
        if (req->toTasks.front()->getTask()->getType() == Task::Output
            && !oport_->isOutputTimestep() && !oport_->isCheckpointTimestep()) {
          if (gpu_stats.active()) {
            gpu_stats << myRankThread()
                << "   Preparing GPU dependencies, ignoring non-output-timestep send for "
                << *req << std::endl;
          }
          continue;
        }
        OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();

        if (gpu_stats.active()) {
          gpu_stats << myRankThread()
              << " --> Preparing GPU dependencies for sending " << *req
              << ", ghost: " << req->req->gtype << ", "
              << req->req->numGhostCells << " from dw " << dw->getID() << '\n';
        }
        const VarLabel* posLabel;
        OnDemandDataWarehouse* posDW;

        // the load balancer is used to determine where data was in the old dw on the prev timestep -
        // pass it in if the particle data is on the old dw
        LoadBalancer* lb = 0;

        if (!reloc_new_posLabel_ && parentScheduler_) {
          posDW =
              dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
          posLabel = parentScheduler_->reloc_new_posLabel_;
        } else {
          // on an output task (and only on one) we require particle variables from the NewDW
          if (req->toTasks.front()->getTask()->getType() == Task::Output) {
            posDW =
                dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
          } else {
            posDW =
                dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
            lb = getLoadBalancer();
          }
          posLabel = reloc_new_posLabel_;
        }
        //Invoke Kernel to copy this range out of the GPU.
        prepareGpuDependencies(deviceVars, taskVars, ghostVars, task, batch,
            posLabel, dw, posDW, req, lb, DeviceGhostCells::anotherMpiRank);
      }
    }  // end for (DependencyBatch * batch = task->getComputes() )
  }
}

//______________________________________________________________________
//
void UnifiedScheduler::copyGpuInternalDependencies(DetailedTask * task,
    int iteration, int t_id) {
  MALLOC_TRACE_TAG_SCOPE("UnifiedScheduler::copyGpuInternalDependencies");

  for (int i = 0; i < (int) dws.size(); i++) {
    if (task->getInternalComputes() != NULL) {
      dws[i]->copyGPUGhostCellsBetweenDevices(task,
          OnDemandDataWarehouse::getNumDevices());
    }
  }
}


//______________________________________________________________________
//
void UnifiedScheduler::syncTaskGpuDWs(DetailedTask* dtask) {

  //For each GPU datawarehouse, see if there are ghost cells listed to be copied
  //if so, launch a kernel that copies them.

  if (dtask->getTaskGpuDataWarehouse(Task::OldDW) != NULL) {
    dtask->getTaskGpuDataWarehouse(Task::OldDW)->syncto_device(dtask->getCUDAStream());
  }
  if (dtask->getTaskGpuDataWarehouse(Task::NewDW) != NULL) {
    dtask->getTaskGpuDataWarehouse(Task::NewDW)->syncto_device(dtask->getCUDAStream());
  }
}


//______________________________________________________________________
//
void UnifiedScheduler::performInternalGhostCellCopies(DetailedTask* dtask) {

  //For each GPU datawarehouse, see if there are ghost cells listed to be copied
  //if so, launch a kernel that copies them.

  if (dtask->getTaskGpuDataWarehouse(Task::OldDW) != NULL
      && dtask->getTaskGpuDataWarehouse(Task::OldDW)->ghostCellCopiesNeeded()) {
    dtask->getTaskGpuDataWarehouse(Task::OldDW)->copyGpuGhostCellsToGpuVarsInvoker(
        dtask->getCUDAStream());
  } else {
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << myRankThread()
            << " No internal ghost cell copies needed for this task \""
            << dtask->getName() << "\"\'s old DW"<< endl;
      }
      cerrLock.unlock();
    }
  }
  if (dtask->getTaskGpuDataWarehouse(Task::NewDW) != NULL
      && dtask->getTaskGpuDataWarehouse(Task::NewDW)->ghostCellCopiesNeeded()) {
    dtask->getTaskGpuDataWarehouse(Task::NewDW)->copyGpuGhostCellsToGpuVarsInvoker(
        dtask->getCUDAStream());
  } else {
    if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << myRankThread()
                << " No internal ghost cell copies needed for this task \""
                << dtask->getName() << "\"\'s new DW"<< endl;
          }
          cerrLock.unlock();
        }
  }
}

//______________________________________________________________________
//
void UnifiedScheduler::copyAllExtGpuDependenciesToHost(const DetailedTask* dtask,
    const DeviceGridVariables& deviceVars,
    const DeviceGhostCells& ghostVars) {

  bool copiesExist = false;
  //If we put it in ghostVars, then we copied it to an array on the GPU.  Go through the ones that indicate
  //they are going to another MPI rank.  Copy them out to the host.  To make the engine cleaner for now,
  //we'll then do another copy step into the variable.  In the future, to be more efficient, we should skip the
  //host to host copy and instead have sendMPI() send the array we get from the device instead.
  for (unsigned int i = 0; i < ghostVars.numItems(); i++) {

    //TODO: Needs a particle section
    if (ghostVars.getDestination(i) == DeviceGhostCells::anotherMpiRank) {
      void* host_ptr    = NULL;    // host base pointer to raw data
      void* device_ptr  = NULL;  // device base pointer to raw data
      size_t host_bytes = 0;
      IntVector host_low, host_high, host_offset, host_size, host_strides;
      int3 device_offset;
      int3 device_size;

      //We created a temporary host variable for this earlier,
      //and the deviceVars collection knows about it.  (We also created it in the "open" destination patch).
      DeviceGridVariableInfo item = deviceVars.getItem(ghostVars.getLabel(i),
                 ghostVars.getDestPatchPointer(i),
                 ghostVars.getMatlIndx(i),
                 ghostVars.getLevelIndx(i),
                 (const int)ghostVars.getDwIndex(i));
      GridVariableBase* tempGhostVar = (GridVariableBase*)item.var;
      //make a spot on the host for it.
      //GridVariableBase* tempGhostVar =
      //    dynamic_cast<GridVariableBase*>(ghostVars.getLabel(i)->typeDescription()->createInstance());
      //tempGhostVar->allocate(ghostVarscopyHostDataToDevice.getLow(i), ghostVars.getHigh(i)); //TODO, clean up.
      tempGhostVar->getSizes(host_low, host_high, host_offset, host_size,
         host_strides);

      host_ptr = tempGhostVar->getBasePointer();
      host_bytes = tempGhostVar->getDataSize();

      // copy the computes data back to the host
      //d2hComputesLock_.writeLock();
      //{

        GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(host_strides.x());
        OnDemandDataWarehouseP dw = dws[(int)ghostVars.getDwIndex(i)];
        GPUDataWarehouse* gpudw = dw->getGPUDW(ghostVars.getSourceDeviceNum(i));

        gpudw->get(*device_var, ghostVars.getLabel(i)->getName().c_str(), ghostVars.getDestPatchPointer(i)->getID(), ghostVars.getMatlIndx(i));
        device_var->getArray3(device_offset, device_size, device_ptr);

        // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
        if (device_offset.x == host_low.x()
            && device_offset.y == host_low.y()
            && device_offset.z == host_low.z()
            && device_size.x == host_size.x()
            && device_size.y == host_size.y()
            && device_size.z == host_size.z()) {

          //Since we know we need a stream, obtain one.

          cudaStream_t* stream = dtask->getCUDAStream(ghostVars.getSourceDeviceNum(i));

          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread()
                  << " copyAllExtGpuDependenciesToHost()  - \""
                  << ghostVars.getLabel(i)->getName() << "\", size = "
                  << std::dec << host_bytes << " to " << std::hex
                  << host_ptr << " from " << std::hex << device_ptr
                  << ", using stream " << std::hex
                  << dtask->getCUDAStream() << std::dec << endl;
            }
            cerrLock.unlock();
          }

          CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes,
                  cudaMemcpyDeviceToHost, *stream));
          copiesExist = true;
        }

      //}
      //d2hComputesLock_.writeUnlock();
      delete device_var;
    }
  }

  if (copiesExist) {
    //Wait until all streams are done
    //Further optimization could be to check each stream one by one and make copies before waiting for other streams to complete.

    while (!dtask->checkAllCUDAStreamsDone()) {
      //sleep?
      printf("Sleeping\n");
    }


    for (unsigned int i = 0; i < ghostVars.numItems(); i++) {

      if (ghostVars.getDestination(i) == DeviceGhostCells::anotherMpiRank) {
        //TODO: Needs a particle section

        IntVector host_low, host_high, host_offset, host_size, host_strides;

        //We created a temporary host variable for this earlier,
        //and the deviceVars collection knows about it.
        DeviceGridVariableInfo item = deviceVars.getItem(ghostVars.getLabel(i),
            ghostVars.getDestPatchPointer(i),
            ghostVars.getMatlIndx(i),
            ghostVars.getLevelIndx(i),
            (const int)ghostVars.getDwIndex(i));

        GridVariableBase* tempGhostVar = (GridVariableBase*)item.var;

        OnDemandDataWarehouseP dw = dws[(int)ghostVars.getDwIndex(i)];


        //Also get the existing host copy
        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(ghostVars.getLabel(i)->typeDescription()->createInstance());

        //Get variable size. Scratch computes means we need to factor that in when computing the size.
        Patch::VariableBasis basis = Patch::translateTypeToBasis(ghostVars.getLabel(i)->typeDescription()->getType(), false);
        IntVector lowIndex, highIndex;
        IntVector lowOffset, highOffset;

        Patch::getGhostOffsets(
            gridVar->virtualGetTypeDescription()->getType(),
            item.dep->gtype, item.dep->numGhostCells, lowOffset,
            highOffset);
        ghostVars.getSourcePatchPointer(i)->computeExtents(basis,
            item.dep->var->getBoundaryLayer(), lowOffset, highOffset,
            lowIndex, highIndex);

        //size_t memSize = (highIndex.x() - lowIndex.x())
        //    * (highIndex.y() - lowIndex.y())
        //    * (highIndex.z() - lowIndex.z()) * host_strides.x();

        //If it doesn't exist yet on the host, create it.  If it does exist,
        //nuke the old contents and create a new one.  Either way gridVar will
        //hold the var for the source patch for us.
        dw->allocateAndPut(*gridVar, item.dep->var, ghostVars.getMatlIndx(i),
            ghostVars.getSourcePatchPointer(i), item.dep->gtype,
            item.dep->numGhostCells);
        //printf("grid vars sizes are %d, %d, %d to %d, %d, %d at data dize is %d\n", gridVar->getLow().x(), gridVar->getLow().y(), gridVar->getLow().z(), gridVar->getHigh().x(), gridVar->getHigh().y(),gridVar->getHigh().z(), gridVar->getDataSize());
        gridVar->copyPatch(tempGhostVar, ghostVars.getLow(i), ghostVars.getHigh(i) );
        /*constNCVariable<double> phi;
        dw. get(phi, ghostVars.getLabel(i),
            ghostVars.getMatlIndx(i),
            ghostVars.getSourcePatchPointer(i),
            item.dep->gtype, ghostLayers, item.dep->numGhostCells);
        double*** phi_data = (double***)gridVar->getWindow()->getData()->get3DPointer();
        for (int i = ghostVars.getLow(i).z(); i < ghostVars.getHigh(i).z(); i++) {
          for (int j = ghostVars.getLow(i).y(); j < ghostVars.getHigh(i).y(); j++) {
            for (int k = ghostVars.getLow(i).x(); k < ghostVars.getHigh(i).x(); k++) {
               printf("At phi_data[%d][%d][%d], value is %1.6lf\n",i, j, k, phi_data[i][j][k])
            }
          }
        }*/

        //double * dst = new double[tempGhostVar->getDataSize()/sizeof(double)];
        //tempGhostVar->copyOut(dst);

        //for (int i = 0; i < tempGhostVar->getDataSize()/sizeof(double); i++) {
        //  printf("At phi_data[%d], value is %1.6lf\n",i, dst[i]);
        //}


        //let go of our reference counters.
        delete gridVar;
        delete tempGhostVar;
      }
    }
  }
}

void UnifiedScheduler::clearForeignGpuVars( DeviceGridVariables& deviceVars){
  map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = deviceVars.getMap();
  for (map<DeviceGridVariableInfo::LabelPatchMatlLevelDw,
         DeviceGridVariableInfo>::iterator it = varMap.begin();
         it != varMap.end(); ++it) {
    if (it->second.foreign) {

      int dwIndex = it->second.dep->mapDataWarehouse();
      dws[dwIndex]->getGPUDW(it->second.whichGPU)->remove(
          it->first.label.c_str(),
          it->first.patchID,
          it->first.matlIndx,
          it->first.levelIndx);
    }
  }
}
#endif

//______________________________________________________________________
//  generate string   <MPI rank>.<Thread ID>
//  useful to see who running what    
std::string UnifiedScheduler::myRankThread() {
  std::ostringstream out;
  out << Uintah::Parallel::getMPIRank() << "." << Thread::self()->myid();
  return out.str();
}

//------------------------------------------
// UnifiedSchedulerWorker Thread Methods
//------------------------------------------
UnifiedSchedulerWorker::UnifiedSchedulerWorker(UnifiedScheduler* scheduler,
    int thread_id) :
    d_scheduler(scheduler), d_runsignal("run condition"), d_runmutex(
        "run mutex"), d_quit(false), d_idle(true), d_thread_id(thread_id + 1), d_rank(
        scheduler->getProcessorGroup()->myrank()), d_waittime(0.0), d_waitstart(
        0.0) {
  d_runmutex.lock();
}

//______________________________________________________________________
//

void UnifiedSchedulerWorker::run() {
  Thread::self()->set_myid(d_thread_id);

  // Set affinity
  if (unified_compactaffinity.active()) {
    if ((unified_threaddbg.active()) && (Uintah::Parallel::getMPIRank() == 0)) {
      cerrLock.lock();
      std::string threadType =
          (d_scheduler->parentScheduler_) ? " subscheduler " : " ";
      unified_threaddbg << "Binding" << threadType << "thread ID "
          << d_thread_id << " to core " << d_thread_id << "\n";
      cerrLock.unlock();
    }
    Thread::self()->set_affinity(d_thread_id);
  }

  while (true) {
    d_runsignal.wait(d_runmutex); // wait for main thread signal.
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds() - d_waitstart;

    if (d_quit) {
      if (taskdbg.active()) {
        cerrLock.lock();
        unified_threaddbg << "Worker " << d_rank << "-" << d_thread_id
            << " quitting" << "\n";
        cerrLock.unlock();
      }
      return;
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      unified_threaddbg << "Worker " << d_rank << "-" << d_thread_id
          << ": executing tasks \n";
      cerrLock.unlock();
    }

    try {
      d_scheduler->runTasks(d_thread_id);
    } catch (Exception& e) {
      cerrLock.lock();
      std::cerr << "Worker " << d_rank << "-" << d_thread_id
          << ": Caught exception: " << e.message() << "\n";
      if (e.stackTrace()) {
        std::cerr << "Stack trace: " << e.stackTrace() << '\n';
      }
      cerrLock.unlock();
    }

    if (taskdbg.active()) {
      cerrLock.lock();
      unified_threaddbg << "Worker " << d_rank << "-" << d_thread_id
          << ": finished executing tasks   \n";
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

double UnifiedSchedulerWorker::getWaittime() {
  return d_waittime;
}

//______________________________________________________________________
//

void UnifiedSchedulerWorker::resetWaittime(double start) {
  d_waitstart = start;
  d_waittime = 0.0;
}

