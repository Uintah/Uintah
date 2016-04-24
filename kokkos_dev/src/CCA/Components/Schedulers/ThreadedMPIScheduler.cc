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
#include <Core/Util/DOUT.hpp>

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>

#include <sched.h>

#define USE_PACKING

using namespace Uintah;


//______________________________________________________________________
//
namespace {

std::condition_variable  g_next_signal;
std::mutex               g_next_mutex;  // conditional wait mutex
std::mutex               g_io_mutex;

Dout g_output_mpi_infos( "MPI_Reporting"  , false );

} // namespace


//______________________________________________________________________
//
namespace Uintah { namespace Impl {

namespace {

__thread int       t_tid = 0;

}

namespace {

enum class ThreadState : int
{
    Inactive
  , Active
  , Exit
};

TaskWorker           * g_runners[MAX_THREADS]        = {};
volatile ThreadState   g_thread_states[MAX_THREADS]  = {};
int                    g_cpu_affinities[MAX_THREADS] = {};
int                    g_num_threads                 = 0;

volatile int g_run_tasks{0};


//______________________________________________________________________
//
void set_affinity( const int proc_unit )
{
#ifndef __APPLE__
  //disable affinity on OSX since sched_setaffinity() is not available in OSX API
  cpu_set_t mask;
  unsigned int len = sizeof(mask);
  CPU_ZERO(&mask);
  CPU_SET(proc_unit, &mask);
  sched_setaffinity(0, len, &mask);
#endif
}


//______________________________________________________________________
//
void thread_driver( const int tid )
{
  // t_tid is __thread variable, unique to each std::thread spawned below
  t_tid = tid;

  // set each TaskWorker thread's affinity
  set_affinity( g_cpu_affinities[tid] );

  try {
    // wait until main thread sets function and changes states
    g_thread_states[tid] = ThreadState::Inactive;
    while (g_thread_states[tid] == ThreadState::Inactive) {
      std::this_thread::yield();
    }

    while (g_thread_states[tid] == ThreadState::Active) {

      // run the function and wait for main thread to reset state
      g_runners[tid]->run();

      g_thread_states[tid] = ThreadState::Inactive;
      while (g_thread_states[tid] == ThreadState::Inactive) {
        std::this_thread::yield();
      }
    }
  } catch (const std::exception & e) {
    std::cerr << "Exception thrown from worker thread: " << e.what() << std::endl;
    std::cerr.flush();
    std::abort();
  } catch (...) {
    std::cerr << "Unknown Exception thrown from worker thread" << std::endl;
    std::cerr.flush();
    std::abort();
  }
}


//______________________________________________________________________
// only called by thread 0 (main thread)
void thread_fence()
{
  // main thread tid is at [0]
  g_thread_states[0] = ThreadState::Inactive;

  // TaskRunner threads start at [1]
  for (int i = 1; i < g_num_threads; ++i) {
    while (g_thread_states[i] == ThreadState::Active) {
      std::this_thread::yield();
//      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
}


//______________________________________________________________________
// only called by main thread
void init_threads( ThreadedMPIScheduler * sched, int num_threads )
{
  g_num_threads = num_threads;
  for (int i = 0; i < g_num_threads; ++i) {
    g_thread_states[i]  = ThreadState::Active;
    g_cpu_affinities[i] = i;
  }

  // set main thread's affinity - core 0
  set_affinity(g_cpu_affinities[0]);
  t_tid = 0;

  // TaskRunner threads start at [1]
  for (int i = 1; i < g_num_threads; ++i) {
    g_runners[i] = new TaskWorker(sched);
  }

  // spawn worker threads
  // TaskRunner threads start at [1]
  for (int i = 1; i < g_num_threads; ++i) {
    std::thread(thread_driver, i).detach();
  }

  thread_fence();
}

} // namespace
}} // namespace Uintah::Impl




ThreadedMPIScheduler::ThreadedMPIScheduler( const ProcessorGroup        * myworld
                                          ,  const Output               * oport
                                          ,        ThreadedMPIScheduler * parentScheduler
                                          )
  : MPIScheduler(myworld, oport, parentScheduler)
{
  if (g_output_mpi_infos) {
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

  if (g_output_mpi_infos) {
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
  m_task_queue_alg = MostMessages;
  std::string taskQueueAlg = "MostMessages";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
    if (taskQueueAlg == "FCFS") {
      m_task_queue_alg = FCFS;
    }
    else if (taskQueueAlg == "Random") {
      m_task_queue_alg = Random;
    }
    else if (taskQueueAlg == "Stack") {
      m_task_queue_alg = Stack;
    }
    else if (taskQueueAlg == "MostChildren") {
      m_task_queue_alg = MostChildren;
    }
    else if (taskQueueAlg == "LeastChildren") {
      m_task_queue_alg = LeastChildren;
    }
    else if (taskQueueAlg == "MostAllChildren") {
      m_task_queue_alg = MostChildren;
    }
    else if (taskQueueAlg == "LeastAllChildren") {
      m_task_queue_alg = LeastChildren;
    }
    else if (taskQueueAlg == "MostL2Children") {
      m_task_queue_alg = MostL2Children;
    }
    else if (taskQueueAlg == "LeastL2Children") {
      m_task_queue_alg = LeastL2Children;
    }
    else if (taskQueueAlg == "MostMessages") {
      m_task_queue_alg = MostMessages;
    }
    else if (taskQueueAlg == "LeastMessages") {
      m_task_queue_alg = LeastMessages;
    }
    else if (taskQueueAlg == "PatchOrder") {
      m_task_queue_alg = PatchOrder;
    }
    else if (taskQueueAlg == "PatchOrderRandom") {
      m_task_queue_alg = PatchOrderRandom;
    }
  }

  proc0cout << "   Using \"" << taskQueueAlg << "\" task queue priority algorithm" << std::endl;

  m_num_threads = Uintah::Parallel::getNumThreads() - 1;
  if ((m_num_threads < 1) && Uintah::Parallel::usingMPI()) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: no thread number specified for ThreadedMPIScheduler" << std::endl;
      throw ProblemSetupException("This scheduler requires number of threads to be in the range [2, 64],\n.... please use -nthreads <num>", __FILE__, __LINE__);
      }
    }
  else if (m_num_threads > MAX_THREADS) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: Number of threads too large..." << std::endl;
      throw ProblemSetupException("Too many threads. Reduce MAX_THREADS and recompile.", __FILE__, __LINE__);
    }
  }

  if (d_myworld->myrank() == 0) {
    std::string plural = (m_num_threads == 1) ? " thread" : " threads";
    std::cout << "   WARNING: Component tasks must be thread safe.\n"
              << "   Using 1 thread for scheduling, and " << m_num_threads
              << plural + " for task execution." << std::endl;
  }

  // this spawns threads, sets affinity, etc
  Impl::init_threads(this, m_num_threads);

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
  subsched->m_num_threads = Uintah::Parallel::getNumThreads() - 1;

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

  if (g_output_mpi_infos) {
    d_labels.clear();
    d_times.clear();
	makeTaskGraphDoc(dts, d_myworld->myrank());
    emitTime("taskGraph output");
    emitTime("time since last execute");
  }

  mpi_info_.reset( 0 );

  int  numTasksDone = 0;
  bool abort        = false;
  int  abort_point  = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  int currphase = 0;
  int numPhases = tg->getNumTaskPhases();
  std::vector<int> phaseTasks(numPhases);
  std::vector<int> phaseTasksDone(numPhases);
  std::vector<DetailedTask*> phaseSyncTask(numPhases);

  dts->setTaskPriorityAlg(m_task_queue_alg);

  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }

//  for (int i = 0; i < numThreads_; i++) {
//    t_worker[i]->resetWaittime(Time::currentSeconds());
//  }

  //------------------------------------------------------------------------------------------------
  // activate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!d_sharedState->isCopyDataTimestep()) {
    Impl::g_run_tasks = 1;
    for (int i = 1; i < m_num_threads; ++i) {
      Impl::g_thread_states[i] = Impl::ThreadState::Active;
    }
  }
  //------------------------------------------------------------------------------------------------

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

  //------------------------------------------------------------------------------------------------
  // deactivate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!d_sharedState->isCopyDataTimestep()) {
    Impl::g_run_tasks = 0;

    Impl::thread_fence();

    for (int i = 1; i < m_num_threads; ++i) {
      Impl::g_thread_states[i] = Impl::ThreadState::Inactive;
    }
  }
  //------------------------------------------------------------------------------------------------


  if (g_output_mpi_infos) {
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

  if( g_output_mpi_infos && !parentScheduler_ ) {  // only do on toplevel scheduler
    outputTimingStats("ThreadedMPIScheduler");
  }
} // end execute()

//______________________________________________________________________
//

int ThreadedMPIScheduler::getAvailableThreadNum()
{
  int num = 0;
  for (int i = 0; i < m_num_threads; i++) {
    if (Impl::g_runners[i]->m_task == nullptr) {
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
  g_next_mutex.lock();
  if (getAvailableThreadNum() == 0) {
    std::unique_lock<std::mutex> next_lock(g_next_mutex);
    g_next_signal.wait(next_lock);
  }
  // find an idle thread and assign task
  int targetThread = -1;
  for (int i = 0; i < m_num_threads; i++) {
    if (Impl::g_runners[i]->m_task == nullptr) {
      targetThread = i;
      Impl::g_runners[i]->m_num_tasks++;
      break;
    }
  }
  g_next_mutex.unlock();

  ASSERT(targetThread >= 0);

  // send task and wake up worker
  Impl::g_runners[targetThread]->m_run_mutex.lock();
  Impl::g_runners[targetThread]->m_task = task;
  Impl::g_runners[targetThread]->m_iteration = iteration;
  Impl::g_runners[targetThread]->m_run_signal.notify_one();
  Impl::g_runners[targetThread]->m_run_mutex.unlock();
}

//------------------------------------------
// TaskWorker Thread Methods
//------------------------------------------
TaskWorker::TaskWorker( ThreadedMPIScheduler * scheduler )
  : m_scheduler{scheduler}
  , m_rank{ scheduler->getProcessorGroup()->myrank() }
{
  m_run_mutex.lock();
}

//______________________________________________________________________
//

void
TaskWorker::run()
{
  while (true) {
    std::unique_lock<std::mutex> next_lock(g_next_mutex);
    m_run_signal.wait(next_lock); // wait for main thread signal
    m_run_mutex.unlock();
    m_wait_time += Time::currentSeconds() - m_wait_start;

    if (m_quit) {
      return;
    }

    ASSERT(m_task != NULL);
    try {
      if (m_task->getTask()->getType() == Task::Reduction) {
        m_scheduler->initiateReduction(m_task);
      }
      else {
        m_scheduler->runTask(m_task, m_iteration, Impl::t_tid);
      }
    }
    catch (Exception& e) {
      std::lock_guard<std::mutex> lock(g_io_mutex);
      {
        std::cerr << "TaskWorker " << m_rank << "-" << Impl::t_tid << ": Caught exception: " << e.message() << "\n";
        if (e.stackTrace()) {
          std::cerr << "Stack trace: " << e.stackTrace() << '\n';
        }
      }
    }

    // Signal main thread for next task.
    g_next_mutex.lock();
    m_run_mutex.lock();
    m_task = NULL;
    m_iteration = 0;
    m_wait_start = Time::currentSeconds();
    g_next_signal.notify_one();
    g_next_mutex.unlock();
  }
}

//______________________________________________________________________
//

double TaskWorker::getWaittime()
{
    return  m_wait_time;
}

//______________________________________________________________________
//

void TaskWorker::resetWaittime( double start )
{
    m_wait_start = start;
    m_wait_time  = 0.0;
}

