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
#include <CCA/Components/Schedulers/TaskWorker.h>
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

#include   <cstring>

#include <sci_defs/cuda_defs.h>

#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllimport)
#else
#define UINTAHSHARE
#endif

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern UINTAHSHARE SCIRun::Mutex cerrLock;
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

GPUThreadedMPIScheduler::GPUThreadedMPIScheduler(const ProcessorGroup* myworld,
                                                 Output* oport,
                                                 GPUThreadedMPIScheduler* parentScheduler) :
    MPIScheduler(myworld, oport, parentScheduler), d_nextsignal("next condition"),
      d_nextmutex("next mutex"), dlbLock("loadbalancer lock") {

  initializeGPUVars();
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
}

void GPUThreadedMPIScheduler::problemSetup(const ProblemSpecP& prob_spec,
                                           SimulationStateP& state) {
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

  numThreads_ = Uintah::Parallel::getMaxThreads() - 1;
  if (numThreads_ < 1) {
    if (d_myworld->myrank() == 0) {
      cerr << "Error: no thread number specified" << endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads > 1, use  -nthreads <num> ", __FILE__,
          __LINE__);
    }
  } else if (numThreads_ > 32) {
    if (d_myworld->myrank() == 0) {
      cerr << "Error: thread number too large" << endl;
      throw ProblemSetupException(
          "Too many number of threads. This scheduler only supports up to 32 threads", __FILE__,
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
    TaskWorker * worker = scinew TaskWorker(this, i);
    t_worker[i] = worker;
    sprintf(name, "Computing Worker %d-%d", Parallel::getRootProcessorGroup()->myrank(), i);
    Thread * t = scinew Thread(worker, name);
    t_thread[i] = t;
    //t->detach();
  }

//  WAIT_FOR_DEBUGGER();
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
  Thread::self()->set_affinity(0);  //bind main thread to cpu 0

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

  // postMPISends() and done() is called here for TaskWorker threads... for GPU tasks using the control
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

void GPUThreadedMPIScheduler::execute(int tgnum /*=0*/,
                                      int iteration /*=0*/) {
  if (d_sharedState->isCopyDataTimestep()) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::execute");
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
  for (int i = 0; i < numThreads_; i++)
    t_worker[i]->resetWaittime(Time::currentSeconds());

  /*control loop for all tasks of task graph*/
//  WAIT_FOR_DEBUGGER();
  while (numTasksDone < ntasks) {

    if (phaseTasks[currphase] == phaseTasksDone[currphase]) {  // this phase done, goto next phase
      currphase++;
      taskdbg << d_myworld->myrank() << " Switched to Task Phase " << currphase << " , total task  "
              << phaseTasks[currphase] << endl;
    }

    // 1.) if we have an internally-ready CPU task, initiate its recvs
    //       * meaning prepare the CPU task for the CPU external ready queue
    else if (dts->numInternalReadyTasks() > 0) {
      DetailedTask * task = dts->getNextInternalReadyTask();

      //save the reduction task and once per proc task for later execution
      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->getType() == Task::OncePerProc)) {
        phaseSyncTask[task->getTask()->d_phase] = task;
        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Task Reduction/OPP ready " << *task
                  << " deps needed: " << task->getExternalDepCount() << endl;
          cerrLock.unlock();
        }
      } else {
        initiateTask(task, abort, abort_point, iteration);
        task->markInitiated();

        // put task in CPU external-ready queue if MPI recvs are complete
        task->checkExternalDepCount();

        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Task internal ready " << *task << " deps needed: "
                  << task->getExternalDepCount() << endl;
          cerrLock.unlock();
          pending_tasks.insert(task);
        }
      }
    }

    // 2.) if we have a GPU task that has its MPI comm completed, then prepare device memory
    //       this is roughly the GPU analog to getting CPU tasks' MPI recvs underway,
    //       meaning prepare the GPU task for the GPU external ready queue
    else if (dts->numInternalReadyGPUTasks() > 0) {

      // check highest priority GPU task with H2D copies completed and put into the GPU external ready queue
      checkH2DCopyDependencies(dts); // may empty the queue, hence the check below

      // otherwise process another GridVariable
      if (dts->numInternalReadyGPUTasks() > 0) {
        DetailedTask* dtask = dts->getNextInternalReadyGPUTask();

        // declare the primary CUDA context for the specified device, "currentGPU_" this calls driverAPI -> cuCtxSetCurrent().
        CUDA_SAFE_CALL( cudaSetDevice(currentGPU_) );

        // assign a device to this GPU task, round robin fashion for now
        dtask->assignDevice(currentGPU_);

        currentGPU_++;
        currentGPU_ %= this->numGPUs_;

        // initiate H2D mem copies for this task's variables
        initiateH2DRequiresCopies(dtask, iteration);
        initiateH2DComputesCopies(dtask, iteration);
        dtask->markInitiated();

        if (taskdbg.active()) {
          cerrLock.lock();
          // taskdbg << d_myworld->myrank() << " GPU task internal ready " << *task << " deps needed: " << task->getExternalDepCount() << endl;
          cerrLock.unlock();
          pending_tasks.insert(dtask);
        }
      }
    }

    // 3.) if it is time to run reduction task, do so
    else if ((phaseSyncTask.find(currphase) != phaseSyncTask.end()) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {
      if (queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }
      DetailedTask *reducetask = phaseSyncTask[currphase];
      taskdbg << d_myworld->myrank() << " Ready Reduce/OPP task "
              << reducetask->getTask()->getName() << endl;
      if (reducetask->getTask()->getType() == Task::Reduction) {
        if (!abort) {
          currcomm++;
          taskdbg << d_myworld->myrank() << " Running Reduce task "
                  << reducetask->getTask()->getName() << " with communicator " << currcomm << endl;
          assignTask(reducetask, currcomm);
        }
      } else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->getType() == Task::OncePerProc);
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();

        while (reducetask->getExternalDepCount() > 0) {
          processMPIRecvs(WAIT_ONCE);
          reducetask->checkExternalDepCount();
        }

        assignTask(reducetask, iteration);
        taskdbg << d_myworld->myrank() << " Running OPP task:  \t";
        printTask(taskdbg, reducetask);
        taskdbg << '\n';
      }ASSERT(reducetask->getTask()->d_phase==currphase);
      numTasksDone++;
      phaseTasksDone[reducetask->getTask()->d_phase]++;
    }

    // 4.) Run a task that has its MPI communication complete. These tasks get in the external
    //     ready queue automatically when their receive count hits 0 in DependencyBatch::received,
    //     which is called when a message is delivered.
    else if (dts->numExternalReadyTasks() > 0) {

      // this is for debugging
      if (queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }

      DetailedTask* dtask = dts->getNextExternalReadyTask();

      // if it's a GPU-enabled task, place it in the GPU internal-ready queue
      if (dtask->getTask()->usesGPU()) {
        dts->addInitialReadyGPUTask(dtask);
        continue;
      }

      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << d_myworld->myrank() << " Dispatching task " << *dtask << "("
                << dts->numExternalReadyTasks() << "/" << pending_tasks.size() << " tasks in queue)"
                << endl;
        cerrLock.unlock();
        pending_tasks.erase(pending_tasks.find(dtask));
      }
      ASSERTEQ(dtask->getExternalDepCount(), 0);
      assignTask(dtask, iteration);
      numTasksDone++;
      phaseTasksDone[dtask->getTask()->d_phase]++;
    }

    // 5.) if we have a GPU task with its device memory prepared, execute the task
    else if (dts->numExternalReadyGPUTasks() > 0) {
      DetailedTask* dtask = dts->getNextExternalReadyGPUTask();

      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << d_myworld->myrank() << " Dispatching task " << *dtask << "("
                << dts->numExternalReadyTasks() << "/" << pending_tasks.size() << " tasks in queue)"
                << endl;
        cerrLock.unlock();
        pending_tasks.erase(pending_tasks.find(dtask));
      }
//      // TODO - create the GPU analog to getExternalDepCount()
//      ASSERTEQ(task->getExternalDepCount(), 0);
      runGPUTask(dtask, iteration);
      numTasksDone++;
      phaseTasksDone[dtask->getTask()->d_phase]++;
    }

    // 6.) otherwise there's nothing to do but process MPI recvs and also see if any GPU tasks
    //       have their D2H copies completed so that done() can be called
    else {
      if (dts->numExternalReadyGPUTasks() > 0) {
        checkGPUTaskCompletion(dts, iteration); // dtask->done(dws) called here
      }
      processMPIRecvs(TEST);
    }

  }  // end while( numTasksDone < ntasks )


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
    emitTime("Total send time",
             mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time",
             mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    //emitTime("Total reduction time",
    //         mpi_info_.totalreduce - mpi_info_.totalreducempi);
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

  freeDeviceRequiresMem(); // call cudaFree on all device allocated memory
  freeDeviceComputesMem(); // call cudaFree on all device allocated memory
  freePinnedHostMem();     // unregister all host memory that was page-locked for async mem copies

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
    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE, MPI_SUM, 0,
               d_myworld->getComm());
    MPI_Reduce(&d_times[0], &d_maxtimes[0], (int)d_times.size(), MPI_DOUBLE, MPI_MAX, 0,
               d_myworld->getComm());

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
    //timeout << "GPUThreadedMPIScheduler: TOTAL                                    "
    //        << total << '\n';
    //timeout << "GPUThreadedMPIScheduler: time sum reduction (one processor only): "
    //        << rtime << '\n';
  }

  if (execout.active()) {
    static int count = 0;

    if (++count % 10 == 0) {
      ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);

      for (map<string, double>::iterator iter = exectimes.begin(); iter != exectimes.end(); iter++)
        fout << fixed << d_myworld->myrank() << ": TaskExecTime: " << iter->second << " Task:"
             << iter->first << endl;
      fout.close();
      //exectimes.clear();
    }
  }
  if (waitout.active()) {
    static int count = 0;

    //only output the wait times every so many timesteps
    if (++count % 100 == 0) {
      for (map<string, double>::iterator iter = waittimes.begin(); iter != waittimes.end(); iter++)
        waitout << fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second
                << " Task:" << iter->first << endl;

      for (map<string, double>::iterator iter = DependencyBatch::waittimes.begin();
          iter != DependencyBatch::waittimes.end(); iter++)
        waitout << fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second
                << " Task:" << iter->first << endl;

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }

  if (dbg.active()) {
    dbg << me << " GPUThreadedMPIScheduler finished\n";
  }
  //pg_ = 0;
}

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

void GPUThreadedMPIScheduler::initializeGPUVars() {
  CUDA_SAFE_CALL( cudaGetDeviceCount(&numGPUs_) );
  currentGPU_ = 0;
}

void GPUThreadedMPIScheduler::initiateH2DRequiresCopies(DetailedTask* dtask, int iteration) {

  MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::initiateGPUTask");
  TAU_PROFILE("GPUThreadedMPIScheduler::initiateGPUTask()", " ", TAU_USER);

  // determine which variables it will require
  const Task* task = dtask->getTask();
  for (const Task::Dependency* req = task->getRequires(); req != 0; req = req->next) {
    constHandle<PatchSubset> patches = req->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls   = matls->size();
    int numStreams = numPatches * numMatls;
    int numEvents  = numStreams;
    createCudaStreams(numStreams);
    createCudaEvents(numEvents);

    // for now, we're only interested in grid variables
    TypeDescription::Type type = req->var->typeDescription()->getType();
    if (type == TypeDescription::CCVariable   || type == TypeDescription::NCVariable || type == TypeDescription::SFCXVariable ||
        type == TypeDescription::SFCYVariable || type == TypeDescription::SFCZVariable) {

      int dwIndex = req->mapDataWarehouse();
      OnDemandDataWarehouseP dw = dws[dwIndex];
      IntVector size;
      double* h_data = NULL;

      for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numMatls; j++) {

          // get the host memory that will be copied to the device
          if (type == TypeDescription::CCVariable) {
            constCCVariable<double> ccVar;
            dw->get(ccVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_data = (double*)ccVar.getWindow()->getData()->getPointer();
            size = ccVar.getWindow()->getData()->size();
            h2dRequiresCopy(dtask, req->var, size, h_data);

          } else if (type == TypeDescription::NCVariable) {
            constNCVariable<double> ncVar;
            dw->get(ncVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_data = (double*)ncVar.getWindow()->getData()->getPointer();
            size = ncVar.getWindow()->getData()->size();
            h2dRequiresCopy(dtask, req->var, size, h_data);

          } else if (type == TypeDescription::SFCXVariable) {
            constSFCXVariable<double> sfcxVar;
            dw->get(sfcxVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_data = (double*)sfcxVar.getWindow()->getData()->getPointer();
            size = sfcxVar.getWindow()->getData()->size();
            h2dRequiresCopy(dtask, req->var, size, h_data);

          } else if (type == TypeDescription::SFCYVariable) {
            constSFCYVariable<double> sfcyVar;
            dw->get(sfcyVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_data = (double*)sfcyVar.getWindow()->getData()->getPointer();
            size = sfcyVar.getWindow()->getData()->size();
            h2dRequiresCopy(dtask, req->var, size, h_data);

          } else if (type == TypeDescription::SFCZVariable) {
            constSFCZVariable<double> sfczVar;
            dw->get(sfczVar, req->var, matls->get(j), patches->get(i), req->gtype, req->numGhostCells);
            h_data = (double*)sfczVar.getWindow()->getData()->getPointer();
            size = sfczVar.getWindow()->getData()->size();
            h2dRequiresCopy(dtask, req->var, size, h_data);
          }
        }
      }
    }
  }
}

void GPUThreadedMPIScheduler::initiateH2DComputesCopies(DetailedTask* dtask, int iteration) {

  MALLOC_TRACE_TAG_SCOPE("GPUThreadedMPIScheduler::initiateGPUTask");
  TAU_PROFILE("GPUThreadedMPIScheduler::initiateGPUTask()", " ", TAU_USER);

  // determine which variables it will require
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls   = matls->size();
    int numStreams = numPatches * numMatls;
    int numEvents  = numPatches * numMatls;
    createCudaStreams(numStreams);
    createCudaEvents(numEvents);

    // for now, we're only interested in grid variables
    TypeDescription::Type type = comp->var->typeDescription()->getType();
    if (type == TypeDescription::CCVariable   || type == TypeDescription::NCVariable || type == TypeDescription::SFCXVariable ||
        type == TypeDescription::SFCYVariable || type == TypeDescription::SFCZVariable) {

      int dwIndex = comp->mapDataWarehouse();
      OnDemandDataWarehouseP dw = dws[dwIndex];
      IntVector size;
      double* h_data = NULL;

      for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numMatls; j++) {

          // get the host memory that will be copied to the device
          if (type == TypeDescription::CCVariable) {
            CCVariable<double> ccVar;
            dw->allocateAndPut(ccVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_data = (double*)ccVar.getWindow()->getData()->getPointer();
            size = ccVar.getWindow()->getData()->size();
            h2dComputesCopy(dtask, comp->var, size, h_data);

          } else if (type == TypeDescription::NCVariable) {
            NCVariable<double> ncVar;
            dw->allocateAndPut(ncVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_data = (double*)ncVar.getWindow()->getData()->getPointer();
            size = ncVar.getWindow()->getData()->size();
            h2dComputesCopy(dtask, comp->var, size, h_data);

          } else if (type == TypeDescription::SFCXVariable) {
            SFCXVariable<double> sfcxVar;
            dw->allocateAndPut(sfcxVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_data = (double*)sfcxVar.getWindow()->getData()->getPointer();
            size = sfcxVar.getWindow()->getData()->size();
            h2dComputesCopy(dtask, comp->var, size, h_data);

          } else if (type == TypeDescription::SFCYVariable) {
            SFCYVariable<double> sfcyVar;
            dw->allocateAndPut(sfcyVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_data = (double*)sfcyVar.getWindow()->getData()->getPointer();
            size = sfcyVar.getWindow()->getData()->size();
            h2dComputesCopy(dtask, comp->var, size, h_data);

          } else if (type == TypeDescription::SFCZVariable) {
            SFCZVariable<double> sfczVar;
            dw->allocateAndPut(sfczVar, comp->var, matls->get(j), patches->get(i), comp->gtype, comp->numGhostCells);
            h_data = (double*)sfczVar.getWindow()->getData()->getPointer();
            size = sfczVar.getWindow()->getData()->size();
            h2dComputesCopy(dtask, comp->var, size, h_data);
          }
        }
      }
    }
  }
}

void GPUThreadedMPIScheduler::h2dRequiresCopy(DetailedTask* dtask, const VarLabel* label, IntVector size, double* h_reqData) {

  // allocate device memory and add to map associating it to its variable
  double* d_reqData;
  int nbytes = size.x() * size.y() * size.z() * sizeof(double);

  // page-lock host memory for async copy to device
  // cudaHostRegisterPortable flag is used so returned memory will be considered pinned by all CUDA contexts
  CUDA_SAFE_CALL( cudaHostRegister((void*)&h_reqData, nbytes, cudaHostRegisterPortable) );

//  double* hostPinned;
//  cutilSafeCall( cudaMallocHost((void**)&hostPinned, nbytes) );
//  CUDA_SAFE_CALL( cudaMemcpy(hostPinned, h_varData, nbytes, cudaMemcpyHostToHost) );

  // set the device and CUDA context
  CUDA_SAFE_CALL( cudaSetDevice(dtask->getDeviceNum()) );
  CUDA_SAFE_CALL( cudaMalloc(&d_reqData, nbytes) );
  deviceRequiresPtrs.insert(pair<const VarLabel*, GPUGridVariable>(label, GPUGridVariable(d_reqData, size, dtask->getDeviceNum())));
  hostRequiresPtrs.insert(pair<const VarLabel*, GPUGridVariable>(label, GPUGridVariable(h_reqData, size, dtask->getDeviceNum())));

  // get a stream and an event from the appropriate queues
  cudaStream_t* stream = getCudaStream();
  cudaEvent_t* event = getCudaEvent();
  dtask->addH2DStream(stream);
  dtask->addH2DCopyEvent(event);

  // set up the host2device memcopy and follow it with an event added to the stream
  CUDA_SAFE_CALL( cudaMemcpyAsync(d_reqData, h_reqData, nbytes, cudaMemcpyDefault, *stream) );
  CUDA_SAFE_CALL( cudaEventRecord(*event, *stream) );
  dtask->incrementH2DCopyCount();
}

void GPUThreadedMPIScheduler::h2dComputesCopy (DetailedTask* dtask, const VarLabel* label, IntVector size, double* h_compData)
{
  // allocate device memory and add to map associating it to its variable
  double* d_compData;
  int nbytes = size.x() * size.y() * size.z() * sizeof(double);

  // page-lock host memory for async copy to device
  // cudaHostRegisterPortable flag is used so returned memory will be considered pinned by all CUDA contexts
  CUDA_SAFE_CALL( cudaHostRegister((void*)&h_compData, nbytes, cudaHostRegisterPortable) );

//  double* hostPinned;
//  cutilSafeCall( cudaMallocHost((void**)&hostPinned, nbytes) );
//  CUDA_SAFE_CALL( cudaMemcpy(hostPinned, h_varData, nbytes, cudaMemcpyHostToHost) );

  // set the device and CUDA context
  CUDA_SAFE_CALL( cudaSetDevice(dtask->getDeviceNum()) );
  CUDA_SAFE_CALL( cudaMalloc(&d_compData, nbytes) );
  deviceComputesPtrs.insert(pair<const VarLabel*, GPUGridVariable>(label, GPUGridVariable(d_compData, size, dtask->getDeviceNum())));
  hostComputesPtrs.insert(pair<const VarLabel*, GPUGridVariable>(label, GPUGridVariable(h_compData, size, dtask->getDeviceNum())));
  hostPtrToTasksMap.insert(pair<double*, DetailedTask*>(h_compData, dtask));

  // get a stream and an event from the appropriate queues
  cudaStream_t* stream = getCudaStream();
  cudaEvent_t* event = getCudaEvent();
  dtask->addH2DStream(stream);
  dtask->addH2DCopyEvent(event);

  // set up the host2device memcopy and follow it with an event added to the stream
  CUDA_SAFE_CALL( cudaMemcpyAsync(d_compData, h_compData, nbytes, cudaMemcpyDefault, *stream) );
  CUDA_SAFE_CALL( cudaEventRecord(*event, *stream) );
  dtask->incrementH2DCopyCount();
}

void GPUThreadedMPIScheduler::checkH2DCopyDependencies(DetailedTasks* dts)
{
  DetailedTask* dtask = dts->peekNextInternalReadyGPUTask();
  cudaError_t ret;

  // see if highest priority tasks has its H2D copies completed, if so, add to GPU external ready queue
  if (dtask->getH2DCopyCount() > 0) {
    if ((ret = dtask->checkH2DCopyDependencies()) == cudaSuccess) {
      // all work associated with this task's h2d copies is complete
      dtask = dts->getNextInternalReadyGPUTask();
      dts->addExternalReadyGPUTask(dtask);
      reclaimStreams(dtask, H2D);
      reclaimEvents(dtask, H2D);
      return;
    }
  }
}

void GPUThreadedMPIScheduler::checkGPUTaskCompletion(DetailedTasks* dts, int iteration)
{
  // see if highest priority tasks has its D2H copies completed, if so, done() can be called
  DetailedTask* dtask = dts->peekNextExternalReadyGPUTask();
  cudaError_t ret;

  if (dtask->getD2HCopyCount() > 0) {
    if ((ret = dtask->checkD2HCopyDependencies()) == cudaSuccess) {
      dtask = dts->getNextExternalReadyGPUTask();
      postMPISends(dtask, iteration, 0); // t_id 0 (the control thread) for centralized threaded scheduler
      reclaimStreams(dtask, D2H);
      reclaimEvents(dtask, D2H);
      dtask->done(dws);
    }
  }
}

void GPUThreadedMPIScheduler::createCudaStreams(int numStreams)
{
  for(int i = 0; i < numStreams; i++) {
    cudaStream_t* stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    cutilSafeCall( cudaStreamCreate(stream) );
    cudaStreams.push(stream);
  }
}

void GPUThreadedMPIScheduler::createCudaEvents(int numEvents)
{
  for(int i = 0; i < numEvents; i++) {
    cudaEvent_t* event = (cudaEvent_t*) malloc(sizeof(cudaEvent_t));
    cutilSafeCall( cudaEventCreate(event) );
    cudaEvents.push(event);
  }
}

void GPUThreadedMPIScheduler::clearCudaStreams()
{
  while (!cudaStreams.empty()) {
    cudaStream_t* stream = cudaStreams.front();
    cudaStreams.pop();
    cudaStreamDestroy(*stream);
  }
}

void GPUThreadedMPIScheduler::clearCudaEvents()
{
  while (!cudaEvents.empty()) {
    cudaEvent_t* event = cudaEvents.front();
    cudaEvents.pop();
    cudaEventDestroy(*event);
  }
}

cudaStream_t* GPUThreadedMPIScheduler::getCudaStream()
{
  if (cudaStreams.size() > 0) {
    cudaStream_t* stream = cudaStreams.front();
    cudaStreams.pop();
    return stream;
  } else { // shouldn't need any more than the queue capacity, but in case
    return ((cudaStream_t*) malloc(sizeof(cudaStream_t)));
  }
}

cudaEvent_t* GPUThreadedMPIScheduler::getCudaEvent()
{
  if (cudaEvents.size() > 0) {
    cudaEvent_t* event = cudaEvents.front();
    cudaEvents.pop();
    return event;
  } else { // shouldn't need any more than the queue capacity, but in case
    return ((cudaEvent_t*)malloc(sizeof(cudaEvent_t)));
  }
}
void GPUThreadedMPIScheduler::addCudaStream(cudaStream_t* stream)
{
  cudaStreams.push(stream);
}

void GPUThreadedMPIScheduler::addCudaEvent(cudaEvent_t* event)
{
  cudaEvents.push(event);
}

double* GPUThreadedMPIScheduler::getDeviceRequiresPtr(const VarLabel* label)
{
  return deviceRequiresPtrs.find(label)->second.ptr;
}

double* GPUThreadedMPIScheduler::getDeviceComputesPtr(const VarLabel* label)
{
  return deviceComputesPtrs.find(label)->second.ptr;
}

double* GPUThreadedMPIScheduler::getHostRequiresPtr(const VarLabel* label)
{
  return hostRequiresPtrs.find(label)->second.ptr;
}

double* GPUThreadedMPIScheduler::getHostComputesPtr(const VarLabel* label)
{
  return hostComputesPtrs.find(label)->second.ptr;
}

IntVector GPUThreadedMPIScheduler::getDeviceRequiresSize(const VarLabel* label)
{
  return deviceRequiresPtrs.find(label)->second.size;
}

IntVector GPUThreadedMPIScheduler::getDeviceComputesSize(const VarLabel* label)
{
  return deviceComputesPtrs.find(label)->second.size;
}

void GPUThreadedMPIScheduler::requestD2HCopy(const VarLabel* label,
                                             double* h_data,
                                             double* d_data,
                                             cudaStream_t* stream,
                                             cudaEvent_t* event)
{
  cudaSetDevice(hostComputesPtrs.find(label)->second.device);
  IntVector size = hostComputesPtrs.find(label)->second.size;
  int nbytes = size.x() * size.y() * size.z() * sizeof(double);
  CUDA_SAFE_CALL( cudaMemcpyAsync(h_data, d_data, nbytes, cudaMemcpyDefault, *stream));
  CUDA_SAFE_CALL( cudaEventRecord(*event, *stream) );
  DetailedTask* dtask = hostPtrToTasksMap.find(d_data)->second;
  dtask->addD2HCopyEvent(event);
}

void GPUThreadedMPIScheduler::freeDeviceRequiresMem()
{
  std::map<const VarLabel*, GPUGridVariable>::iterator iter;
  for(iter = deviceRequiresPtrs.begin(); iter != deviceRequiresPtrs.end(); iter++) {
    cudaSetDevice(iter->second.device); // set the CUDA context so the free() works
    cudaFree(iter->second.ptr);
  }
}

void GPUThreadedMPIScheduler::freeDeviceComputesMem()
{
  std::map<const VarLabel*, GPUGridVariable>::iterator iter;
  for(iter=deviceComputesPtrs.begin(); iter != deviceComputesPtrs.end(); iter++) {
    cudaSetDevice(iter->second.device); // set the CUDA context so the free() works
    cudaFree(iter->second.ptr);
  }
}

void GPUThreadedMPIScheduler::freePinnedHostMem()
{
  std::map<const VarLabel*, GPUGridVariable>::iterator iter;
  for(iter = deviceComputesPtrs.begin(); iter != deviceComputesPtrs.end(); iter++) {
    cudaSetDevice(iter->second.device); // set the CUDA context so the free() works
    cudaHostUnregister(iter->second.ptr);
  }
}

void GPUThreadedMPIScheduler::reclaimStreams(DetailedTask* dtask, CopyType type)
{
  std::vector<cudaStream_t*>* dtaskStreams;
  dtaskStreams = ((type == H2D) ? dtask->getH2DStreams() : dtask->getH2DStreams());

  std::vector<cudaStream_t*>::iterator iter;
  for (iter = dtaskStreams->begin(); iter != dtaskStreams->end(); iter++) {
    this->cudaStreams.push(*iter);
  }
  dtaskStreams->clear();
}

void GPUThreadedMPIScheduler::reclaimEvents(DetailedTask* dtask, CopyType type)
{
  std::vector<cudaEvent_t*>* dtaskEvents;
  dtaskEvents = ((type == H2D) ? dtask->getH2DCopyEvents() : dtask->getD2HCopyEvents());

  std::vector<cudaEvent_t*>::iterator iter;
  for (iter = dtaskEvents->begin(); iter != dtaskEvents->end(); iter++) {
    this->cudaEvents.push(*iter);
  }
  dtaskEvents->clear();

}
