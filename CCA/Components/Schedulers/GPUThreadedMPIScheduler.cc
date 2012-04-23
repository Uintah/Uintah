/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/GPUThreadedMPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Ports/Output.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

#include <cstring>

#include <sci_defs/cuda_defs.h>

#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex cerrLock;
extern DebugStream taskdbg;
extern DebugStream mpidbg;
extern map<string, double> waittimes;
extern map<string, double> exectimes;
extern DebugStream waitout;
extern DebugStream execout;

static double CurrentWaitTime = 0;

static DebugStream dbg("GPUThreadedMPIScheduler", false);
static DebugStream timeout("GPUThreadedMPIScheduler.timings", false);
static DebugStream queuelength("QueueLength", false);
static DebugStream threaddbg("ThreadDBG", false);
static DebugStream affinity("CPUAffinity", true);

GPUThreadedMPIScheduler::GPUThreadedMPIScheduler(const ProcessorGroup* myworld,
                                                 Output* oport,
                                                 GPUThreadedMPIScheduler* parentScheduler) :
    MPIScheduler(myworld, oport, parentScheduler), d_nextsignal("next condition"),
      d_nextmutex("next mutex"), dlbLock("loadbalancer lock") {

  if (Parallel::usingGPU()) {
    gpuInitialize();

    // we need one of these for each GPU, as each device will have it's own CUDA context
    for (int i = 0; i < numGPUs_; i++) {
      idleStreams.push_back(std::queue<cudaStream_t*>());
      idleEvents.push_back(std::queue<cudaEvent_t*>());
    }
  }

  // disable memory windowing on variables.  This will ensure that
  // each variable is allocated its own memory on each patch,
  // precluding memory blocks being defined across multiple patches.
  Uintah::OnDemandDataWarehouse::d_combineMemory = false;
}

GPUThreadedMPIScheduler::~GPUThreadedMPIScheduler() {
  for (int i = 0; i < numThreads_; i++) {
    t_worker[i]->d_runmutex.lock();
    t_worker[i]->quit();
    t_worker[i]->d_runsignal.conditionSignal();
    t_worker[i]->d_runmutex.unlock();
    t_thread[i]->setCleanupFunction(NULL);
    t_thread[i]->join();
  }

  if (timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      avgStats.close();
      maxStats.close();
    }
  }

  if (Parallel::usingGPU()) {
    // cleanup CUDA stream and event handles
    // TODO need to fix clearCudaStreams() bug
    idleStreams.clear();
    idleEvents.clear();
  }
}

void GPUThreadedMPIScheduler::problemSetup(const ProblemSpecP& prob_spec, SimulationStateP& state) {
  //default taskReadyQueueAlg
  taskQueueAlg_ = MostMessages;
  string taskQueueAlg = "MostMessages";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);

    if (taskQueueAlg == "FCFS")
      taskQueueAlg_ = FCFS;
    else if (taskQueueAlg == "Random")
      taskQueueAlg_ = Random;
    else if (taskQueueAlg == "Stack")
      taskQueueAlg_ = Stack;
    else if (taskQueueAlg == "MostChildren")
      taskQueueAlg_ = MostChildren;
    else if (taskQueueAlg == "LeastChildren")
      taskQueueAlg_ = LeastChildren;
    else if (taskQueueAlg == "MostAllChildren")
      taskQueueAlg_ = MostChildren;
    else if (taskQueueAlg == "LeastAllChildren")
      taskQueueAlg_ = LeastChildren;
    else if (taskQueueAlg == "MostL2Children")
      taskQueueAlg_ = MostL2Children;
    else if (taskQueueAlg == "LeastL2Children")
      taskQueueAlg_ = LeastL2Children;
    else if (taskQueueAlg == "MostMessages")
      taskQueueAlg_ = MostMessages;
    else if (taskQueueAlg == "LeastMessages")
      taskQueueAlg_ = LeastMessages;
    else if (taskQueueAlg == "PatchOrder")
      taskQueueAlg_ = PatchOrder;
    else if (taskQueueAlg == "PatchOrderRandom")
      taskQueueAlg_ = PatchOrderRandom;
  }
  if (d_myworld->myrank() == 0) {
    cout << "   Using \"" << taskQueueAlg << "\" Algorithm" << endl;
  }

  numThreads_ = Uintah::Parallel::getNumThreads() - 1;
  if (numThreads_ < 1) {
    if (d_myworld->myrank() == 0) {
      cerr << "Error: no thread number specified" << endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads > 1, use  -nthreads <num> and -gpu ", __FILE__,
          __LINE__);
    }
  } else if (numThreads_ > MAX_THREADS) {
    if (d_myworld->myrank() == 0) {
      cerr << "Error: thread number too large" << endl;
      throw ProblemSetupException(
          "Too many number of threads. Try to increase MAX_THREADS and recompile", __FILE__,
          __LINE__);
    }
  }

  if (d_myworld->myrank() == 0) {
    cout << "\tWARNING: " << (Uintah::Parallel::usingGPU() ? "GPU " : "")
         << "Multi-thread/MPI hybrid scheduler is EXPERIMENTAL "
         << "not all tasks are thread safe yet." << endl << "\tUsing 1 thread for scheduling, "
         << numThreads_ << " threads for task execution." << endl;
  }

  /* d_nextsignal = scinew ConditionVariable("NextCondition");
   d_nextmutex = scinew Mutex("NextMutex");*/

  char name[1024];

  for (int i = 0; i < numThreads_; i++) {
    GPUTaskWorker * worker = scinew GPUTaskWorker(this, i);
    t_worker[i] = worker;
    sprintf(name, "Computing Worker %d-%d", Parallel::getRootProcessorGroup()->myrank(), i);
    Thread * t = scinew Thread(worker, name);
    t_thread[i] = t;
    //t->detach();
  }

//  WAIT_FOR_DEBUGGER();
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
  if (affinity.active()) Thread::self()->set_affinity(0);  //bind main thread to cpu 0

}

SchedulerP GPUThreadedMPIScheduler::createSubScheduler() {
  GPUThreadedMPIScheduler* newsched = scinew GPUThreadedMPIScheduler(d_myworld, m_outPort, this);
  newsched->d_sharedState = d_sharedState;
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState = d_sharedState;
  return newsched;
}

void GPUThreadedMPIScheduler::runTask(DetailedTask* task, int iteration, int t_id /*=0*/) {
  TAU_PROFILE("GPUThreadedMPIScheduler::runTask()", " ", TAU_USER);

  if (waitout.active()) {
    waittimes[task->getTask()->getName()] += CurrentWaitTime;
    CurrentWaitTime = 0;
  }

  double taskstart = Time::currentSeconds();

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }

  vector<DataWarehouseP> plain_old_dws(dws.size());
  for (int i = 0; i < (int)dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
    //const char* tag = AllocatorSetDefaultTag(task->getTask()->getName());
  }

  task->doit(d_myworld, dws, plain_old_dws);
  //AllocatorSetDefaultTag(tag);

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  double dtask = Time::currentSeconds() - taskstart;

  dlbLock.lock();
  if (execout.active()) {
    exectimes[task->getTask()->getName()] += dtask;
  }

  //if i do not have a sub scheduler 
  if (!task->getTask()->getHasSubScheduler()) {
    //add my task time to the total time
    mpi_info_.totaltask += dtask;
    //if(d_myworld->myrank()==0)
    //  cout << "adding: " << dtask << " to counters, new total: " << mpi_info_.totaltask << endl;
    if (!d_sharedState->isCopyDataTimestep() && task->getTask()->getType() != Task::Output) {
      //if(d_myworld->myrank()==0 && task->getPatches()!=0)
      //  cout << d_myworld->myrank() << " adding: " << task->getTask()->getName() << " to profile:" << dtask << " on patches:" << *(task->getPatches()) << endl;
      //add contribution for patchlist
      getLoadBalancer()->addContribution(task, dtask);
    }
  }
  dlbLock.unlock();

  postMPISends(task, iteration, t_id);
  task->done(dws);  // should this be timed with taskstart? - BJW
  double teststart = Time::currentSeconds();

  // sendsLock.lock(); // Dd... could do better?
  sends_[t_id].testsome(d_myworld);
  // sendsLock.unlock(); // Dd... could do better?

  mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;

  if (parentScheduler) {  //add my timings to the parent scheduler
    //  if(d_myworld->myrank()==0)
    //    cout << "adding: " << mpi_info_.totaltask << " to parent counters, new total: " << parentScheduler->mpi_info_.totaltask << endl;
    parentScheduler->mpi_info_.totaltask += mpi_info_.totaltask;
    parentScheduler->mpi_info_.totaltestmpi += mpi_info_.totaltestmpi;
    parentScheduler->mpi_info_.totalrecv += mpi_info_.totalrecv;
    parentScheduler->mpi_info_.totalsend += mpi_info_.totalsend;
    parentScheduler->mpi_info_.totalwaitmpi += mpi_info_.totalwaitmpi;
    parentScheduler->mpi_info_.totalreduce += mpi_info_.totalreduce;
  }

}  // end runTask()

void GPUThreadedMPIScheduler::runGPUTask(DetailedTask* task, int iteration, int t_id /*=0*/) {

  TAU_PROFILE("GPUThreadedMPIScheduler::runGPUTask()", " ", TAU_USER);

  if (waitout.active()) {
    waittimes[task->getTask()->getName()] += CurrentWaitTime;
    CurrentWaitTime = 0;
  }

  double taskstart = Time::currentSeconds();

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }

  vector<DataWarehouseP> plain_old_dws(dws.size());
  for (int i = 0; i < (int)dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
    //const char* tag = AllocatorSetDefaultTag(task->getTask()->getName());
  }

  task->doit(d_myworld, dws, plain_old_dws);

  //AllocatorSetDefaultTag(tag);

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  double dtask = Time::currentSeconds() - taskstart;

  dlbLock.lock();
  if (execout.active()) {
    exectimes[task->getTask()->getName()] += dtask;
  }

  //if i do not have a sub scheduler
  if (!task->getTask()->getHasSubScheduler()) {
    // add my task time to the total time
    mpi_info_.totaltask += dtask;
    //if(d_myworld->myrank()==0)
    //  cout << "adding: " << dtask << " to counters, new total: " << mpi_info_.totaltask << endl;
    if (!d_sharedState->isCopyDataTimestep() && task->getTask()->getType() != Task::Output) {
      // if(d_myworld->myrank()==0 && task->getPatches()!=0)
      //   cout << d_myworld->myrank() << " adding: " << task->getTask()->getName() << " to profile:" << dtask << " on patches:" << *(task->getPatches()) << endl;
      // add contribution for patchlist
      getLoadBalancer()->addContribution(task, dtask);
    }
  }
  dlbLock.unlock();

  // postMPISends() and done() is called here for GPUTaskWorker threads... for GPU tasks using the control
  // thread, postMPISends() and done() are called in GPUThreadedMPIScheduler::checkD2HCopyDependencies

  double teststart = Time::currentSeconds();

  // sendsLock.lock(); // Dd... could do better?
  sends_[t_id].testsome(d_myworld);
  // sendsLock.unlock(); // Dd... could do better?

  mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;

  if (parentScheduler) {  //add my timings to the parent scheduler
    //  if(d_myworld->myrank()==0)
    //    cout << "adding: " << mpi_info_.totaltask << " to parent counters, new total: " << parentScheduler->mpi_info_.totaltask << endl;
    parentScheduler->mpi_info_.totaltask += mpi_info_.totaltask;
    parentScheduler->mpi_info_.totaltestmpi += mpi_info_.totaltestmpi;
    parentScheduler->mpi_info_.totalrecv += mpi_info_.totalrecv;
    parentScheduler->mpi_info_.totalsend += mpi_info_.totalsend;
    parentScheduler->mpi_info_.totalwaitmpi += mpi_info_.totalwaitmpi;
    parentScheduler->mpi_info_.totalreduce += mpi_info_.totalreduce;
  }

}  // end runGPUTask()

void GPUThreadedMPIScheduler::execute(int tgnum /*=0*/, int iteration /*=0*/) {

  if (d_sharedState->isCopyDataTimestep()) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::execute");
  TAU_PROFILE("GPUThreadedMPIScheduler::execute()", " ", TAU_USER);

  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[GPUThreadedMPIScheduler::execute()] " , TAU_USER); TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[GPUThreadedMPIScheduler::execute()] " , TAU_USER); TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[GPUThreadedMPIScheduler::execute()] " , TAU_USER); TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[GPUThreadedMPIScheduler::execute()] ",
      TAU_USER); TAU_PROFILE_TIMER(testsometimer, "Test Some", "[GPUThreadedMPIScheduler::execute()] ",
      TAU_USER); TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[GPUThreadedMPIScheduler::execute()] ",
      TAU_USER); TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[GPUThreadedMPIScheduler::execute()] ",
      TAU_USER); TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[GPUThreadedMPIScheduler::execute()] ",
      TAU_USER);

  ASSERTRANGE(tgnum, 0, (int)graphs.size());
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
    if (d_myworld->myrank() == 0) {
      cerr << "GPUThreadedMPIScheduler skipping execute, no tasks\n";
    }
    return;
  }

  //ASSERT(pg_ == 0);
  //pg_ = pg;

  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  if (timeout.active()) {
    d_labels.clear();
    d_times.clear();
    //emitTime("time since last execute");
  }

  int me = d_myworld->myrank();
  makeTaskGraphDoc(dts, me);

  //if(timeout.active())
  //emitTime("taskGraph output");

  mpi_info_.totalreduce = 0;
  mpi_info_.totalsend = 0;
  mpi_info_.totalrecv = 0;
  mpi_info_.totaltask = 0;
  mpi_info_.totalreducempi = 0;
  mpi_info_.totalsendmpi = 0;
  mpi_info_.totalrecvmpi = 0;
  mpi_info_.totaltestmpi = 0;
  mpi_info_.totalwaitmpi = 0;

  int numTasksDone = 0;

  bool abort = false;
  int abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  TAU_PROFILE_TIMER(doittimer, "Task execution",
      "[GPUThreadedMPIScheduler::execute() loop] ", TAU_USER); TAU_PROFILE_START(doittimer);

  int currphase = 0;
  int currcomm = 0;
  map<int, int> phaseTasks;
  map<int, int> phaseTasksDone;
  map<int, DetailedTask *> phaseSyncTask;
  dts->setTaskPriorityAlg(taskQueueAlg_);
  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }

  if (dbg.active()) {
    cerrLock.lock();
    dbg << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)" << endl;
    cerrLock.unlock();
  }

  static vector<int> histogram;
  static int totaltasks;
  set<DetailedTask*> pending_tasks;

  taskdbg << d_myworld->myrank() << " Switched to Task Phase " << currphase << " , total task  "
          << phaseTasks[currphase] << endl;
  for (int i = 0; i < numThreads_; i++) {
    t_worker[i]->resetWaittime(Time::currentSeconds());
  }

  /*control loop for all tasks of task graph*/
//  WAIT_FOR_DEBUGGER();
  while (numTasksDone < ntasks) {

    if (phaseTasks[currphase] == phaseTasksDone[currphase]) {  // this phase done, goto next phase
      currphase++;
      taskdbg << d_myworld->myrank() << " Switched to Task Phase " << currphase << " , total task  "
              << phaseTasks[currphase] << endl;
    }

    /*
     * (1)
     *
     * If we have an internally-ready CPU task, initiate its MPI recvs, preparing it for
     * CPU external ready queue. The task is moved to the CPU external-ready queue in the
     * call to task->checkExternalDepCount().
     *
     */
    else if (dts->numInternalReadyTasks() > 0) {

      DetailedTask* task = dts->getNextInternalReadyTask();

      //save the reduction task and once per proc task for later execution
      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {
        phaseSyncTask[task->getTask()->d_phase] = task;
        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Task Reduction/OPP ready " << *task
                  << " deps needed: " << task->getExternalDepCount() << endl;
          cerrLock.unlock();
        }
      } else {
        // post this task's MPI recvs
        initiateTask(task, abort, abort_point, iteration);
        task->markInitiated();

        // put task in CPU external-ready queue if MPI recvs are complete
        task->checkExternalDepCount();

        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Task internal ready " << *task << " deps needed: " << task->getExternalDepCount() << endl;
          cerrLock.unlock();
          pending_tasks.insert(task);
        }
      }
    }

    /*
     * (2)
     *
     * If it is time to run reduction task, do so.
     *
     */
    else if ((phaseSyncTask.find(currphase) != phaseSyncTask.end()) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {

      if (queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }

      DetailedTask *reducetask = phaseSyncTask[currphase];

      taskdbg << d_myworld->myrank() << " Ready Reduce/OPP task " << reducetask->getTask()->getName() << endl;
      if (reducetask->getTask()->getType() == Task::Reduction) {
        if (!abort) {
          currcomm++;
          taskdbg << d_myworld->myrank() << " Running Reduce task " << reducetask->getTask()->getName()
                  << " with communicator " << currcomm << endl;
          assignTask(reducetask, currcomm);
        }
      } else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->usesMPI());
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();
        ASSERT(reducetask->getExternalDepCount() == 0) ;
        assignTask(reducetask, iteration);
        taskdbg << d_myworld->myrank() << " Running OPP task:  \t";
        printTask(taskdbg, reducetask);
        taskdbg << '\n';
      }ASSERT(reducetask->getTask()->d_phase==currphase);
      numTasksDone++;
      phaseTasksDone[reducetask->getTask()->d_phase]++;
    }

    /*
     * (3)
     *
     * Run a CPU task that has its MPI communication complete. These tasks get in the external
     * ready queue automatically when their receive count hits 0 in DependencyBatch::received,
     * which is called when a MPI message is delivered.
     *
     * NOTE: This is also where a GPU-enabled task gets into the GPU initially-ready queue
     *
     */
    else if (dts->numExternalReadyTasks() > 0) {

      // this is for debugging
      if (queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }

      DetailedTask* dtask = dts->getNextExternalReadyTask();

      /*
       * If it's a GPU-enabled task, assign it to a device and initiate it's H2D computes
       * and requires data copies. This is where each GPU task's execution cycle begins.
       */
      if (dtask->getTask()->usesGPU()) {

        // assigning devices round robin fashion for now
        dtask->assignDevice(currentGPU_);
        currentGPU_++;
        currentGPU_ %= this->numGPUs_;

        // initiate H2D mem copies for this task's computes and requires
        int timeStep = d_sharedState->getCurrentTopLevelTimeStep();
        if (timeStep > 0) {
          initiateH2DRequiresCopies(dtask, iteration);
          initiateH2DComputesCopies(dtask, iteration);
        }

        dtask->markInitiated();
        dts->addInitiallyReadyGPUTask(dtask);
        continue;
      }  // end first stage of GPU task execution cycle

      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << d_myworld->myrank() << " Dispatching task " << *dtask << "(" << dts->numExternalReadyTasks() << "/"
                << pending_tasks.size() << " tasks in queue)" << endl;
        cerrLock.unlock();
        pending_tasks.erase(pending_tasks.find(dtask));
      }

      ASSERTEQ(dtask->getExternalDepCount(), 0);
      assignTask(dtask, iteration);
      numTasksDone++;
      phaseTasksDone[dtask->getTask()->d_phase]++;
    }

    /*
     * (4)
     *
     * Check if highest priority GPU task's async H2D copies are completed. If so, then
     * reclaim the streams and events it used for these operations, execute the task and
     * then put it into the GPU completion-pending queue.
     */
    else if (dts->numInitiallyReadyGPUTasks() > 0) {

      DetailedTask* dtask = dts->peekNextInitiallyReadyGPUTask();
      cudaError_t retVal = dtask->checkH2DCopyDependencies();
      if (retVal == cudaSuccess) {
        // all work associated with this task's h2d copies is complete
        dtask = dts->getNextInitiallyReadyGPUTask();

        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Dispatching task " << *dtask << "("
                  << dts->numInitiallyReadyGPUTasks() << "/" << pending_tasks.size() << " tasks in queue)"
                  << endl;
          cerrLock.unlock();
          pending_tasks.erase(pending_tasks.find(dtask));
        }

        // recycle this task's H2D copies streams and events
        reclaimStreams(dtask, H2D);
        reclaimEvents(dtask, H2D);

        // TODO - create the GPU analog to getExternalDepCount()
        // ASSERTEQ(task->getExternalDepCount(), 0);
        runGPUTask(dtask, iteration);
        phaseTasksDone[dtask->getTask()->d_phase]++;
        dts->addCompletionPendingGPUTask(dtask);
      }
    }

    /*
     * (5)
     *
     * Check to see if any GPU tasks have their D2H copies completed. This means the kernel(s)
     * have executed and all teh results are back on the host in the DataWarehouse. This task's
     * MPI sends can then be posted and done() can be called.
     */
    else if (dts->numCompletionPendingGPUTasks() > 0) {

      DetailedTask* dtask = dts->peekNextCompletionPendingGPUTask();
      cudaError_t retVal = dtask->checkD2HCopyDependencies();
      if (retVal == cudaSuccess) {
        dtask = dts->getNextCompletionPendingGPUTask();
        postMPISends(dtask, iteration, 0);  // t_id 0 (the control thread) for centralized threaded scheduler

        // recycle streams and events used by the DetailedTask for kernel execution and H2D copies
        reclaimStreams(dtask, D2H);
        reclaimEvents(dtask, D2H);

        // finish up the gpu task
        dtask->done(dws);
        numTasksDone++;
      }
    }

    /*
     * (6)
     *
     * Otherwise there's nothing to do but process MPI recvs.
     */
    else {
      processMPIRecvs(TEST);
    }

  }  // end while( numTasksDone < ntasks )


  // Free up all the pointer maps for device and pinned host pointers
  if (d_sharedState->getCurrentTopLevelTimeStep() != 0) {
    freeDeviceRequiresMem();         // call cudaFree on all device memory for task->requires
    freeDeviceComputesMem();         // call cudaFree on all device memory for task->computes
//    unregisterPageLockedHostMem();   // unregister all registered, page-locked host memory
    clearMaps();
  }


  TAU_PROFILE_STOP(doittimer);

  // wait for all tasks to finish
  wait_till_all_done();
  //if any thread is busy, conditional wait here
  d_nextmutex.lock();
  while (getAviableThreadNum() < numThreads_) {
    d_nextsignal.wait(d_nextmutex);
  }
  d_nextmutex.unlock();

  //if (me==0)
  //  cout <<"AviableThreads : " << getAviableThreadNum()  << ", task worked: " << numTasksDone << endl;

  //if (d_generation > 2)
  //dws[dws.size()-2]->printParticleSubsets();

  if (queuelength.active()) {
    float lengthsum = 0;
    totaltasks += ntasks;
    // if (me == 0) cout << d_myworld->myrank() << " queue length histogram: ";
    for (unsigned int i = 1; i < histogram.size(); i++) {
      // if (me == 0)cout << histogram[i] << " ";
      //cout << iter->first << ":" << iter->second << " ";
      lengthsum = lengthsum + i * histogram[i];
    }
    // if (me==0) cout << endl;
    float queuelength = lengthsum / totaltasks;
    float allqueuelength = 0;
    MPI_Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());
    if (me == 0)
      cout << "average queue length:" << allqueuelength / d_myworld->size() << endl;
  }

  if (timeout.active()) {
    emitTime("MPI send time", mpi_info_.totalsendmpi);
    //emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
    emitTime("Total send time", mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time", mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    //emitTime("Total reduction time", mpi_info_.totalreduce - mpi_info_.totalreducempi);
    emitTime("Total comm time", mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

    double time = Time::currentSeconds();
    double totalexec = time - d_lasttime;

    d_lasttime = time;

    emitTime(
        "Other excution time",
        totalexec - mpi_info_.totalsend - mpi_info_.totalrecv - mpi_info_.totaltask
        - mpi_info_.totalreduce);
  }

  if (d_sharedState != 0) {  // subschedulers don't have a sharedState
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime;  // don't count output time...
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
      if (dws[0])
        dws[0]->setRestarted();
    }
  }

  finalizeTimestep();

  log.finishTimestep();
  if (timeout.active() && !parentScheduler) {  // only do on toplevel scheduler
    //emitTime("finalize");

    // add number of cells, patches, and particles
    int numCells = 0, numParticles = 0;
    OnDemandDataWarehouseP dw = dws[dws.size() - 1];
    const GridP grid(const_cast<Grid*>(dw->getGrid()));
    const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(grid)->getSubset(
        d_myworld->myrank());
    for (int p = 0; p < myPatches->size(); p++) {
      const Patch* patch = myPatches->get(p);
      IntVector range = patch->getExtraCellHighIndex() - patch->getExtraCellLowIndex();
      numCells += range.x() * range.y() * range.z();

      // go through all materials since getting an MPMMaterial correctly would depend on MPM
      for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
        if (dw->haveParticleSubset(m, patch))
          numParticles += dw->getParticleSubset(m, patch)->numParticles();
      }
    }
    d_labels.push_back("NumPatches");
    d_labels.push_back("NumCells");
    d_labels.push_back("NumParticles");
    d_times.push_back(myPatches->size());
    d_times.push_back(numCells);
    d_times.push_back(numParticles);
    vector<double> d_totaltimes(d_times.size());
    vector<double> d_maxtimes(d_times.size());
    vector<double> d_avgtimes(d_times.size());
    double maxTask = -1, avgTask = -1;
    double maxComm = -1, avgComm = -1;
    double maxCell = -1, avgCell = -1;

    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());
    MPI_Reduce(&d_times[0], &d_maxtimes[0], (int)d_times.size(), MPI_DOUBLE, MPI_MAX, 0, d_myworld->getComm());

    double total = 0, avgTotal = 0, maxTotal = 0;
    for (int i = 0; i < (int)d_totaltimes.size(); i++) {
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
      avgTotal += d_avgtimes[i];
      maxTotal += d_maxtimes[i];
    }

    // to not duplicate the code
    vector<ofstream*> files;
    vector<vector<double>*> data;
    files.push_back(&timingStats);
    data.push_back(&d_times);

    if (me == 0) {
      files.push_back(&avgStats);
      files.push_back(&maxStats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (unsigned file = 0; file < files.size(); file++) {
      ofstream& out = *files[file];
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep() << endl;
      for (int i = 0; i < (int)(*data[file]).size(); i++) {
        out << "GPUThreadedMPIScheduler: " << d_labels[i] << ": ";
        int len = (int)(strlen(d_labels[i]) + strlen("GPUThreadedMPIScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++)
          out << ' ';
        double percent;
        if (strncmp(d_labels[i], "Num", 3) == 0)
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i] / d_totaltimes[i] * 100;
        else
          percent = (*data[file])[i] / total * 100;
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << endl << endl;
    }

    if (me == 0) {
      timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = "
              << (1 - avgTask / maxTask) * 100 << " load imbalance (exec)%\n";
      timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = "
              << (1 - avgComm / maxComm) * 100 << " load imbalance (comm)%\n";
      timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = "
              << (1 - avgCell / maxCell) * 100 << " load imbalance (theoretical)%\n";
    }
    double time = Time::currentSeconds();
    //double rtime=time-d_lasttime;
    d_lasttime = time;
    //timeout << "GPUThreadedMPIScheduler: TOTAL << total << '\n';
    //timeout << "GPUThreadedMPIScheduler: time sum reduction (one processor only): " << rtime << '\n';
  }

  if (execout.active()) {
    static int count = 0;

    if (++count % 10 == 0) {
      ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);

      for (map<string, double>::iterator iter = exectimes.begin(); iter != exectimes.end(); iter++) {
        fout << fixed << d_myworld->myrank() << ": TaskExecTime: " << iter->second << " Task:" << iter->first << endl;
      }
      fout.close();
      //exectimes.clear();
    }
  }
  if (waitout.active()) {
    static int count = 0;

    //only output the wait times every so many timesteps
    if (++count % 100 == 0) {
      for (map<string, double>::iterator iter = waittimes.begin(); iter != waittimes.end(); iter++) {
        waitout << fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second
                << " Task:" << iter->first << endl;
      }

      for (map<string, double>::iterator iter = DependencyBatch::waittimes.begin();
          iter != DependencyBatch::waittimes.end(); iter++) {
        waitout << fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second
                << " Task:" << iter->first << endl;
      }

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }

  if (dbg.active()) {
    dbg << me << " GPUThreadedMPIScheduler finished\n";
  }
  //pg_ = 0;

} // end GPUTHreadedMPIScheduler::execute


void GPUThreadedMPIScheduler::postMPISends(DetailedTask * task,
                                           int iteration,
                                           int t_id) {
  MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::postMPISends");
  double sendstart = Time::currentSeconds();
  if (dbg.active()) {
    cerrLock.lock();
    dbg << d_myworld->myrank() << " postMPISends - task " << *task << '\n';
    cerrLock.unlock();
  }

  // Send data to dependendents
  for (DependencyBatch* batch = task->getComputes(); batch != 0; batch = batch->comp_next) {

    // Prepare to send a message
#ifdef USE_PACKING
    PackBufferInfo mpibuff;
#else
    BufferInfo mpibuff;
#endif
    // Create the MPI type
    int to = batch->toTasks.front()->getAssignedResourceIndex();
    ASSERTRANGE(to, 0, d_myworld->size());
    ostringstream ostr;
    ostr.clear();
    for (DetailedDep* req = batch->head; req != 0; req = req->next) {
      if ((req->condition == DetailedDep::FirstIteration && iteration > 0)
          || (req->condition == DetailedDep::SubsequentIterations && iteration == 0)) {
        // See comment in DetailedDep about CommCondition
        if (dbg.active()) {
          dbg << d_myworld->myrank() << "   Ignoring conditional send for " << *req << endl;
        }
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep()
          && !oport_->isCheckpointTimestep()) {
        if (dbg.active()) {
          dbg << d_myworld->myrank() << "   Ignoring non-output-timestep send for " << *req << endl;
        }
        continue;
      }
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();

      //dbg.setActive(req->req->lookInOldTG);
      if (dbg.active()) {
        ostr << *req << ' ';
        //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
        dbg << d_myworld->myrank() << " --> sending " << *req << ", ghost: " << req->req->gtype
            << ", " << req->req->numGhostCells << " from dw " << dw->getID() << '\n';
      }
      const VarLabel* posLabel;
      OnDemandDataWarehouse* posDW;

      // the load balancer is used to determine where data was in the old dw on the prev timestep -
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;

      if (!reloc_new_posLabel_ && parentScheduler) {
        posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
        posLabel = parentScheduler->reloc_new_posLabel_;
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output) {
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        } else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
        posLabel = reloc_new_posLabel_;
      }
      MPIScheduler* top = this;

      while (top->parentScheduler) {
        top = top->parentScheduler;
      }

      dw->sendMPI(batch, posLabel, mpibuff, posDW, req, lb);
    }
    // Post the send
    if (mpibuff.count() > 0) {
      ASSERT(batch->messageTag > 0);
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
      mpibuff.pack(d_myworld->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      //only send message if size is greather than zero
      //we need this empty message to enforce modify after read dependencies 
      //if(count>0)
      //{
      if (dbg.active()) {
        cerrLock.lock();
        //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
        dbg << d_myworld->myrank() << " Sending message number " << batch->messageTag << " to "
            << to << ": " << ostr.str() << "\n";
        cerrLock.unlock();
        //dbg.setActive(false);
      }
      //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
      if (mpidbg.active()) {
        mpidbg << d_myworld->myrank() << " Sending message number " << batch->messageTag << ", to "
               << to << ", length: " << count << "\n";
      }

      numMessages_++;
      int typeSize;

      MPI_Type_size(datatype, &typeSize);
      messageVolume_ += count * typeSize;

      MPI_Request requestid;
      MPI_Isend(buf, count, datatype, to, batch->messageTag, d_myworld->getComm(), &requestid);
      int bytes = count;

      //sendsLock.lock(); // Dd: ??
      sends_[t_id].add(requestid, bytes, mpibuff.takeSendlist(), ostr.str(), batch->messageTag);
      //sendsLock.unlock(); // Dd: ??
      mpi_info_.totalsendmpi += Time::currentSeconds() - start;
      //}
    }
  }  // end for (DependencyBatch * batch = task->getComputes() )
  double dsend = Time::currentSeconds() - sendstart;
  mpi_info_.totalsend += dsend;

}  // end postMPISends();

int GPUThreadedMPIScheduler::getAviableThreadNum() {
  int num = 0;
  for (int i = 0; i < numThreads_; i++) {
    if (t_worker[i]->d_task == NULL) {
      num++;
    }
  }
  return num;
}

void GPUThreadedMPIScheduler::assignTask(DetailedTask* task,
                                         int iteration) {
  d_nextmutex.lock();
  if (getAviableThreadNum() == 0) {
    d_nextsignal.wait(d_nextmutex);
  }
  //find an idle thread and assign task
  int targetThread = -1;
  for (int i = 0; i < numThreads_; i++) {
    if (t_worker[i]->d_task == NULL) {
      targetThread = i;
      break;
    }
  }
  d_nextmutex.unlock();
  //send task and wake up worker
  ASSERT(targetThread>=0);
  t_worker[targetThread]->d_runmutex.lock();
  t_worker[targetThread]->d_task = task;
  t_worker[targetThread]->d_iteration = iteration;
  t_worker[targetThread]->d_runsignal.conditionSignal();
  t_worker[targetThread]->d_runmutex.unlock();
}

void GPUThreadedMPIScheduler::gpuInitialize() {
  cudaError_t retVal;
  CUDA_SAFE_CALL( retVal = cudaGetDeviceCount(&numGPUs_) );
  currentGPU_ = 0;
}

void GPUThreadedMPIScheduler::initiateH2DRequiresCopies(DetailedTask* dtask, int iteration) {

  MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::initiateH2DRequiresCopies");
  TAU_PROFILE("GPUThreadedMPIScheduler::initiateH2DRequiresCopies()", " ", TAU_USER);

  // determine which variables it will require
  const Task* task = dtask->getTask();
  for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->next) {
    constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());

    // for now, we're only interested in grid variables
    TypeDescription::Type type = req->var->typeDescription()->getType();
    if (type == TypeDescription::CCVariable   || type == TypeDescription::NCVariable || type == TypeDescription::SFCXVariable ||
        type == TypeDescription::SFCYVariable || type == TypeDescription::SFCZVariable) {

      // this is so we can allocate persistent events and streams to distribute when needed
      //   one stream and one event per variable per H2D copy (numPatches * numMatls)
      int numPatches = patches->size();
      int numMatls   = matls->size();
      int numStreams = numPatches * numMatls;
      int numEvents  = numStreams;
      int device = dtask->getDeviceNum();

      // knowing how many H2D "requires" copies we'll need, allocate streams and events for them
      createCudaStreams(numStreams, device);
      createCudaEvents(numEvents,  device);

      int dwIndex = req->mapDataWarehouse();
      OnDemandDataWarehouseP dw = dws[dwIndex];
      IntVector size;
      double* h_reqData = NULL;

      for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numMatls; j++) {
          // a fix for when INF ghost cells are requested such as in RMCRT
          //  e.g. tsk->requires(abskg_dw, d_abskgLabel, gac, SHRT_MAX);
          bool uses_SHRT_MAX = (req->numGhostCells == SHRT_MAX);
          if (type == TypeDescription::CCVariable) {
            constCCVariable<double> ccVar;
            if (uses_SHRT_MAX) {
              const Level* level = getLevel(dtask->getPatches());
              IntVector domainLo_EC, domainHi_EC;
              level->findCellIndexRange(domainLo_EC, domainHi_EC); // including extraCells
              dw->getRegion(ccVar, req->var, matls->get(j), level, domainLo_EC, domainHi_EC, true);
            } else {
              dw->get(ccVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            }
            ccVar.getWindow()->getData()->addReference();
            h_reqData = (double*)ccVar.getWindow()->getData()->getPointer();
            size = ccVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::NCVariable) {
            constNCVariable<double> ncVar;
            dw->get(ncVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            ncVar.getWindow()->getData()->addReference();
            h_reqData = (double*)ncVar.getWindow()->getData()->getPointer();
            size = ncVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::SFCXVariable) {
            constSFCXVariable<double> sfcxVar;
            dw->get(sfcxVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_reqData = (double*)sfcxVar.getWindow()->getData()->getPointer();
            size = sfcxVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::SFCYVariable) {
            constSFCYVariable<double> sfcyVar;
            dw->get(sfcyVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_reqData = (double*)sfcyVar.getWindow()->getData()->getPointer();
            size = sfcyVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::SFCZVariable) {
            constSFCZVariable<double> sfczVar;
            dw->get(sfczVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_reqData = (double*)sfczVar.getWindow()->getData()->getPointer();
            size = sfczVar.getWindow()->getData()->size();
          }

          // copy the requires variable data to the device
          h2dRequiresCopy(dtask, req->var, matls->get(j), patches->get(i), size, h_reqData);
        }
      }
    }
  }
}

void GPUThreadedMPIScheduler::initiateH2DComputesCopies(DetailedTask* dtask, int iteration) {

  MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::initiateH2DComputesCopies");
  TAU_PROFILE("GPUThreadedMPIScheduler::initiateH2DComputesCopies()", " ", TAU_USER);

  // determine which variables it will require
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(dtask->getMaterials());

    // for now, we're only interested in grid variables
    TypeDescription::Type type = comp->var->typeDescription()->getType();
    if (type == TypeDescription::CCVariable   || type == TypeDescription::NCVariable || type == TypeDescription::SFCXVariable ||
        type == TypeDescription::SFCYVariable || type == TypeDescription::SFCZVariable) {

      // this is so we can allocate persistent events and streams to distribute when needed
      //   one stream and one event per variable per H2D copy (numPatches * numMatls)
      int numPatches = patches->size();
      int numMatls   = matls->size();
      int numStreams = numPatches * numMatls;
      int numEvents  = numPatches * numMatls;
      int device = dtask->getDeviceNum();

      // knowing how many H2D "computes" copies we'll need, allocate streams and events for them
      createCudaStreams(numStreams, device);
      createCudaEvents(numEvents, device);

      int dwIndex = comp->mapDataWarehouse();
      OnDemandDataWarehouseP dw = dws[dwIndex];
      IntVector size;
      double* h_compData = NULL;

      for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numMatls; j++) {

          // get the host memory that will be copied to the device
          if (type == TypeDescription::CCVariable) {
            CCVariable<double> ccVar;
            dw->allocateAndPut(ccVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            ccVar.getWindow()->getData()->addReference();
            h_compData = (double*)ccVar.getWindow()->getData()->getPointer();
            size = ccVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::NCVariable) {
            NCVariable<double> ncVar;
            dw->allocateAndPut(ncVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            ncVar.getWindow()->getData()->addReference();
            h_compData = (double*)ncVar.getWindow()->getData()->getPointer();
            size = ncVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::SFCXVariable) {
            SFCXVariable<double> sfcxVar;
            dw->allocateAndPut(sfcxVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_compData = (double*)sfcxVar.getWindow()->getData()->getPointer();
            size = sfcxVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::SFCYVariable) {
            SFCYVariable<double> sfcyVar;
            dw->allocateAndPut(sfcyVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_compData = (double*)sfcyVar.getWindow()->getData()->getPointer();
            size = sfcyVar.getWindow()->getData()->size();

          } else if (type == TypeDescription::SFCZVariable) {
            SFCZVariable<double> sfczVar;
            dw->allocateAndPut(sfczVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_compData = (double*)sfczVar.getWindow()->getData()->getPointer();
            size = sfczVar.getWindow()->getData()->size();
          }

          // copy the computes  data to the device
          h2dComputesCopy(dtask, comp->var, matls->get(j), patches->get(i), size, h_compData);
        }
      }
    }
  }
}

void GPUThreadedMPIScheduler::h2dRequiresCopy(DetailedTask* dtask, const VarLabel* label, int matlIndex, const Patch* patch, IntVector size, double* h_reqData)
{
  // set the device and CUDA context
  cudaError_t retVal;
  int device = dtask->getDeviceNum();
  CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );

  double* d_reqData;
  size_t nbytes = size.x() * size.y() * size.z() * sizeof(double);
  VarLabelMatl<Patch> var(label, matlIndex, patch);

//  const bool pinned = ( *(pinnedHostPtrs.find(h_reqData)) == h_reqData );
//  if (!pinned) {
//    // page-lock (pin) host memory for async copy to device
//    // cudaHostRegisterPortable flag is used so returned memory will be considered pinned by all CUDA contexts
//    retVal = cudaHostRegister(h_reqData, nbytes, cudaHostRegisterPortable);
//    if(retVal == cudaSuccess) {
//      pinnedHostPtrs.insert(h_reqData);
//    }
//  }

  CUDA_SAFE_CALL( retVal = cudaMalloc(&d_reqData, nbytes) );
  hostRequiresPtrs.insert(pair<VarLabelMatl<Patch>, GPUGridVariable>(var, GPUGridVariable(dtask, h_reqData, size, device)));
  deviceRequiresPtrs.insert(pair<VarLabelMatl<Patch>, GPUGridVariable>(var, GPUGridVariable(dtask, d_reqData, size, device)));

  // get a stream and an event from the appropriate queues
  cudaStream_t* stream = getCudaStream(device);
  cudaEvent_t* event = getCudaEvent(device);
  dtask->addH2DStream(stream);
  dtask->addH2DCopyEvent(event);

  // set up the host2device memcopy and follow it with an event added to the stream
  CUDA_SAFE_CALL( retVal = cudaMemcpyAsync(d_reqData, h_reqData, nbytes, cudaMemcpyDefault, *stream) );
  CUDA_SAFE_CALL( retVal = cudaEventRecord(*event, *stream) );

  dtask->incrementH2DCopyCount();
}

void GPUThreadedMPIScheduler::h2dComputesCopy (DetailedTask* dtask, const VarLabel* label, int matlIndex, const Patch* patch, IntVector size, double* h_compData)
{
  // set the device and CUDA context
  cudaError_t retVal;
  int device = dtask->getDeviceNum();
  CUDA_SAFE_CALL( retVal = cudaSetDevice(device));

  double* d_compData;
  size_t nbytes = size.x() * size.y() * size.z() * sizeof(double);
  VarLabelMatl<Patch> var(label, matlIndex, patch);

//  const bool pinned = ( *(pinnedHostPtrs.find(h_compData)) == h_compData );
//  if (!pinned) {
//    // page-lock (pin) host memory for async copy to device
//    // cudaHostRegisterPortable flag is used so returned memory will be considered pinned by all CUDA contexts
//    retVal = cudaHostRegister(h_compData, nbytes, cudaHostRegisterPortable);
//    if(retVal == cudaSuccess) {
//      pinnedHostPtrs.insert(h_compData);
//    }
//  }

  CUDA_SAFE_CALL( retVal = cudaMalloc(&d_compData, nbytes) );
  hostComputesPtrs.insert(pair<VarLabelMatl<Patch>, GPUGridVariable>(var, GPUGridVariable(dtask, h_compData, size, device)));
  deviceComputesPtrs.insert(pair<VarLabelMatl<Patch>, GPUGridVariable>(var, GPUGridVariable(dtask, d_compData, size, device)));

  // get a stream and an event from the appropriate queues
  cudaStream_t* stream = getCudaStream(device);
  cudaEvent_t* event = getCudaEvent(device);
  dtask->addH2DStream(stream);
  dtask->addH2DCopyEvent(event);

  // set up the host2device memcopy and follow it with an event added to the stream
  CUDA_SAFE_CALL( retVal = cudaMemcpyAsync(d_compData, h_compData, nbytes, cudaMemcpyDefault, *stream) );
  CUDA_SAFE_CALL( retVal = cudaEventRecord(*event, *stream) );

  dtask->incrementH2DCopyCount();
}

void GPUThreadedMPIScheduler::createCudaStreams(int numStreams, int device)
{
  cudaError_t retVal;

  for (int j = 0; j < numStreams; j++) {
    cutilSafeCall( retVal = cudaSetDevice(device) );
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    cutilSafeCall( retVal = cudaStreamCreate(&(*stream)) );
    idleStreams[device].push(stream);
  }
}

void GPUThreadedMPIScheduler::createCudaEvents(int numEvents, int device)
{
  cudaError_t retVal;

  for (int j = 0; j < numEvents; j++) {
    cutilSafeCall( retVal = cudaSetDevice(device) );
    cudaEvent_t* event = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
    cutilSafeCall( retVal = cudaEventCreate(&(*event)) );
    idleEvents[device].push(event);
  }
}

void GPUThreadedMPIScheduler::clearCudaStreams()
{
  cudaError_t retVal;
  int numQueues = idleStreams.size();

  for (int i = 0; i < numQueues; i++) {
    cutilSafeCall( retVal = cudaSetDevice(i) );
    while (!idleStreams[i].empty()) {
      cudaStream_t* stream = idleStreams[i].front();
      idleStreams[i].pop();
      cutilSafeCall( retVal = cudaStreamDestroy(*stream) );
    }
    idleStreams.clear();
  }
}

void GPUThreadedMPIScheduler::clearCudaEvents()
{
  cudaError_t retVal;
  int numQueues = idleEvents.size();

  for (int i = 0; i < numQueues; i++) {
    CUDA_SAFE_CALL( retVal = cudaSetDevice(i) );
    while (!idleEvents[i].empty()) {
      cudaEvent_t* event = idleEvents[i].front();
      idleEvents[i].pop();
      CUDA_SAFE_CALL( retVal = cudaEventDestroy(*event) );
    }
    idleEvents.clear();
  }
}

cudaStream_t* GPUThreadedMPIScheduler::getCudaStream(int device)
{
  cudaError_t retVal;
  cudaStream_t* stream;

  if (idleStreams[device].size() > 0) {
    stream = idleStreams[device].front();
    idleStreams[device].pop();
  } else { // shouldn't need any more than the queue capacity, but in case
    CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );
    // this will get put into idle stream queue and properly disposed of later
    stream = ((cudaStream_t*) malloc(sizeof(cudaStream_t)));
  }
  return stream;
}

cudaEvent_t* GPUThreadedMPIScheduler::getCudaEvent(int device)
{
  cudaError_t retVal;
  cudaEvent_t* event;

  if (idleEvents[device].size() > 0) {
    event = idleEvents[device].front();
    idleEvents[device].pop();
  } else { // shouldn't need any more than the queue capacity, but in case
    CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );
    // this will get put into idle event queue and properly disposed of later
    event = ((cudaEvent_t*)malloc(sizeof(cudaEvent_t)));
  }
  return event;
}

void GPUThreadedMPIScheduler::addCudaStream(cudaStream_t* stream, int device)
{
  idleStreams[device].push(stream);
}

void GPUThreadedMPIScheduler::addCudaEvent(cudaEvent_t* event, int device)
{
  idleEvents[device].push(event);
}

double* GPUThreadedMPIScheduler::getDeviceRequiresPtr(const VarLabel* label, int matlIndex, const Patch* patch)
{
  VarLabelMatl<Patch> var(label, matlIndex, patch);
  double* d_reqPtr = deviceRequiresPtrs.find(var)->second.ptr;
  return  d_reqPtr;
}

double* GPUThreadedMPIScheduler::getDeviceComputesPtr(const VarLabel* label, int matlIndex, const Patch* patch)
{
  VarLabelMatl<Patch> var(label, matlIndex, patch);
  double* d_compPtr = deviceComputesPtrs.find(var)->second.ptr;
  return  d_compPtr;
}

double* GPUThreadedMPIScheduler::getHostRequiresPtr(const VarLabel* label, int matlIndex, const Patch* patch)
{
  VarLabelMatl<Patch> var(label, matlIndex, patch);
  double* h_reqPtr = hostRequiresPtrs.find(var)->second.ptr;
  return  h_reqPtr;
}

double* GPUThreadedMPIScheduler::getHostComputesPtr(const VarLabel* label, int matlIndex, const Patch* patch)
{
  VarLabelMatl<Patch> var(label, matlIndex, patch);
  double* h_compPtr = hostComputesPtrs.find(var)->second.ptr;
  return h_compPtr;
}

IntVector GPUThreadedMPIScheduler::getDeviceRequiresSize(const VarLabel* label, int matlIndex, const Patch* patch)
{
  VarLabelMatl<Patch> var(label, matlIndex, patch);
  IntVector size = deviceRequiresPtrs.find(var)->second.size;
  return size;
}

IntVector GPUThreadedMPIScheduler::getDeviceComputesSize(const VarLabel* label, int matlIndex, const Patch* patch)
{
  VarLabelMatl<Patch> var(label, matlIndex, patch);
  IntVector size = deviceComputesPtrs.find(var)->second.size;
  return size;
}

void GPUThreadedMPIScheduler::requestD2HCopy(const VarLabel* label,
                                             int matlIndex,
                                             const Patch* patch,
                                             cudaStream_t* stream,
                                             cudaEvent_t* event)
{
  cudaError_t retVal;
  VarLabelMatl<Patch> var(label, matlIndex, patch);

  // set the CUDA context
  DetailedTask* dtask = hostComputesPtrs.find(var)->second.dtask;
  int device = dtask->getDeviceNum();
  CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );

  // collect arguments and setup the async d2h copy
  double* d_compData = deviceComputesPtrs.find(var)->second.ptr;
  double* h_compData = hostComputesPtrs.find(var)->second.ptr;
  IntVector size = hostComputesPtrs.find(var)->second.size;
  size_t nbytes = size.x() * size.y() * size.z() * sizeof(double);

  // event and stream were already added to the task in getCudaEvent(...) and getCudaStream(...)
  CUDA_SAFE_CALL( retVal = cudaMemcpyAsync(h_compData, d_compData, nbytes, cudaMemcpyDefault, *stream) );
  CUDA_SAFE_CALL( retVal = cudaEventRecord(*event, *stream) );

  dtask->incrementD2HCopyCount();
}

cudaError_t GPUThreadedMPIScheduler::freeDeviceRequiresMem()
{
  cudaError_t retVal;
  std::map<VarLabelMatl<Patch>, GPUGridVariable>::iterator iter;

  for(iter = deviceRequiresPtrs.begin(); iter != deviceRequiresPtrs.end(); iter++) {

    // set the device & CUDA context
    int device = iter->second.device;
    CUDA_SAFE_CALL( retVal = cudaSetDevice(device) ); // set the CUDA context so the free() works

    // free the device requires memory
    double* d_reqPtr = iter->second.ptr;
    CUDA_SAFE_CALL( retVal = cudaFree(d_reqPtr) );
  }
  return retVal;
}

cudaError_t GPUThreadedMPIScheduler::freeDeviceComputesMem()
{
  cudaError_t retVal;
  std::map<VarLabelMatl<Patch>, GPUGridVariable>::iterator iter;

  for(iter=deviceComputesPtrs.begin(); iter != deviceComputesPtrs.end(); iter++) {

    // set the device & CUDA context
    int device = iter->second.device;
    CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );

    // free the device computes memory
    double* d_compPtr = iter->second.ptr;
    CUDA_SAFE_CALL( retVal = cudaFree(d_compPtr) );
  }
  return retVal;
}

cudaError_t GPUThreadedMPIScheduler::unregisterPageLockedHostMem()
{
  cudaError_t retVal;
  std::set<double*>::iterator iter;

  for(iter = pinnedHostPtrs.begin(); iter != pinnedHostPtrs.end(); iter++) {

    // unregister the page-locked host requires memory
    double* ptr = *iter;
    CUDA_SAFE_CALL( retVal = cudaHostUnregister(ptr) );
  }
  return retVal;
}

void GPUThreadedMPIScheduler::reclaimStreams(DetailedTask* dtask, CopyType type)
{
  std::vector<cudaStream_t*>* dtaskStreams;
  std::vector<cudaStream_t*>::iterator iter;
  int device = dtask->getDeviceNum();
  dtaskStreams = ( (type == H2D) ? dtask->getH2DStreams() : dtask->getD2HStreams() );

  // reclaim DetailedTask streams
  for (iter = dtaskStreams->begin(); iter != dtaskStreams->end(); iter++) {
    cudaStream_t* stream = *iter;
    this->idleStreams[device].push(stream);
  }
  dtaskStreams->clear();
}

void GPUThreadedMPIScheduler::reclaimEvents(DetailedTask* dtask, CopyType type)
{
  std::vector<cudaEvent_t*>* dtaskEvents;
  std::vector<cudaEvent_t*>::iterator iter;
  int device = dtask->getDeviceNum();
  dtaskEvents = ( (type == H2D) ? dtask->getH2DCopyEvents() : dtask->getD2HCopyEvents() );

  // reclaim DetailedTask events
  for (iter = dtaskEvents->begin(); iter != dtaskEvents->end(); iter++) {
    cudaEvent_t* event = *iter;
    this->idleEvents[device].push(event);
  }
  dtaskEvents->clear();
}

void GPUThreadedMPIScheduler::clearMaps()
{
  deviceRequiresPtrs.clear();
  deviceComputesPtrs.clear();
  hostRequiresPtrs.clear();
  hostComputesPtrs.clear();
  pinnedHostPtrs.clear();
}


/** Computing Thread Methods***/
GPUTaskWorker::GPUTaskWorker(GPUThreadedMPIScheduler* scheduler, int id) :
   d_id(id), d_schedulergpu(scheduler), d_task(NULL), d_iteration(0),
   d_runmutex("run mutex"),  d_runsignal("run condition"), d_quit(false),
   d_waittime(0.0), d_waitstart(0.0), d_rank(scheduler->getProcessorGroup()->myrank())
{
  d_runmutex.lock();
}

GPUTaskWorker::~GPUTaskWorker()
{
}

void GPUTaskWorker::run()
{
//  WAIT_FOR_DEBUGGER();
  threaddbg << "Binding thread id " << d_id+1 << " to cpu " << d_id+1 << endl;
  Thread::self()->set_myid(d_id+1);
  if (affinity.active()) Thread::self()->set_affinity(d_id+1);

  while(true) {
    //wait for main thread signal
    d_runsignal.wait(d_runmutex);
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds()-d_waitstart;

    if (d_quit) {
      if(taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Worker " << d_rank  << "-" << d_id << " quitting   " << "\n";
        cerrLock.unlock();
      }
      return;
    }

    if(taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank  << "-" << d_id << ": executeTask:   " << *d_task << "\n";
      cerrLock.unlock();
    }
    ASSERT(d_task!=NULL);

    try {
      if (d_task->getTask()->getType() == Task::Reduction) {
    	d_schedulergpu->initiateReduction(d_task);
      } else {
          d_schedulergpu->runTask(d_task, d_iteration, d_id);
      }
    } catch (Exception& e) {
      cerrLock.lock();
      cerr << "Worker " << d_rank << "-" << d_id << ": Caught exception: " << e.message() << "\n";
      if(e.stackTrace()) {
        cerr << "Stack trace: " << e.stackTrace() << '\n';
      }
      cerrLock.unlock();
    }

    if(taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank << "-" << d_id << ": finishTask:   " << *d_task << "\n";
      cerrLock.unlock();
    }

    // signal main thread for next task
    d_schedulergpu->d_nextmutex.lock();
    d_runmutex.lock();
    d_task = NULL;
    d_iteration = 0;
    d_waitstart = Time::currentSeconds();
    d_schedulergpu->d_nextsignal.conditionSignal();
    d_schedulergpu->d_nextmutex.unlock();
  }
}

double GPUTaskWorker::getWaittime()
{
    return  d_waittime;
}

void GPUTaskWorker::resetWaittime(double start)
{
    d_waitstart  = start;
    d_waittime = 0.0;
}

