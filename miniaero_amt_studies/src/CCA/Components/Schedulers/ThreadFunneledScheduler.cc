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

#include <CCA/Components/Schedulers/ThreadFunneledScheduler.h>

#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Parallel/CommunicationList.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DOUT.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include <sched.h>

#define USE_PACKING

using namespace Uintah;

extern DebugStream timeout;

//______________________________________________________________________
//
namespace {

DebugStream threaded_dbg(       "ThreadFunneled_DBG",        false);
DebugStream threaded_threaddbg( "ThreadFunneled_ThreadDBG",  false);

Timers::Simple  s_total_exec_time {};
std::mutex      s_io_mutex;
std::mutex      s_lb_mutex;

} // namespace


//______________________________________________________________________
//
namespace Uintah { namespace Impl {

namespace {
thread_local int       t_tid = 0;
}

namespace {

enum class ThreadState : int
{
    Inactive
  , Active
  , Exit
};

TaskRunner *           g_runners[MAX_THREADS]        = {};
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
  // t_tid is thread_local variable, unique to each std::thread spawned below
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
void init_threads( ThreadFunneledScheduler * sched, int num_threads )
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
    g_runners[i] = new TaskRunner(sched);
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


//______________________________________________________________________
//
void ThreadFunneledScheduler::problemSetup(  const ProblemSpecP     & prob_spec
                                           ,       SimulationStateP & state
                                          )
{
  std::string taskQueueAlg = "MostMessages"; // default taskReadyQueueAlg

  proc0cout << "   Using \"" << taskQueueAlg << "\" task queue priority algorithm" << std::endl;

  m_num_threads = Uintah::Parallel::getNumThreads();

  m_task_pool        = TaskPool{ static_cast<size_t>(m_num_threads) };

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
  init_threads(this, m_num_threads);

  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}


//______________________________________________________________________
//
SchedulerP ThreadFunneledScheduler::createSubScheduler()
{
  ThreadFunneledScheduler* newsched = new ThreadFunneledScheduler(d_myworld, m_outPort, this);
  newsched->d_sharedState = d_sharedState;
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState = d_sharedState;
  return newsched;
}


//______________________________________________________________________
//
void ThreadFunneledScheduler::execute(  int tgnum /*=0*/ , int iteration /*=0*/ )
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

  // for multi TG model, where each graph is going to need to have its dwmap reset here (even with the same tgnum)
  if (graphs.size() > 1) {
    tg->remapTaskDWs(dwmap);
  }

  m_detailed_tasks = tg->getDetailedTasks();
  m_detailed_tasks->initializeScrubs(dws, dwmap);
  m_detailed_tasks->initTimestep();
  m_num_tasks = m_detailed_tasks->numLocalTasks();

  for (int i = 0; i < m_num_tasks; ++i) {
    m_detailed_tasks->localTask(i)->resetDependencyCounts();
  }

  makeTaskGraphDoc(m_detailed_tasks, d_myworld->myrank());

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(m_detailed_tasks, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  // clear & resize task phase bookkeeping data structures
  mpi_info_.reset( 0 );
  m_num_tasks_done.store(0, std::memory_order_relaxed);
  m_abort          = false;
  m_abort_point    = 987654;
  m_current_iteration = iteration;
  m_current_phase = 0;
  m_num_phases = tg->getNumTaskPhases();
  m_phase_tasks.clear();
  m_phase_tasks.resize(m_num_phases, 0);
  m_phase_sync_tasks.clear();
  m_phase_sync_tasks.resize(m_num_phases, nullptr);

  m_phase_tasks_done.release();
  m_phase_tasks_done = atomic_int_array(new std::atomic<int>[m_num_phases]{});

  // count the number of tasks in each task-phase
  //   each task is assigned a task-phase in TaskGraph::createDetailedDependencies()
  for (int i = 0; i < m_num_tasks; ++i) {
    ++m_phase_tasks[m_detailed_tasks->localTask(i)->getTask()->d_phase];
  }

  //------------------------------------------------------------------------------------------------
  // activate TaskRunners
  //------------------------------------------------------------------------------------------------
  Impl::g_run_tasks = 1;

  // reset per-thread wait times and activate
  for (int i = 1; i < m_num_threads; ++i) {
    Impl::g_runners[i]->m_task_wait_time.reset();
    Impl::g_thread_states[i] = Impl::ThreadState::Active;
  }
  //------------------------------------------------------------------------------------------------


  // The main task loop
  while (m_num_tasks_done.load(std::memory_order_relaxed) < m_num_tasks) {

    if (m_phase_tasks[m_current_phase] == m_phase_tasks_done[m_current_phase].load(std::memory_order_relaxed)) {  // this phase done, goto next phase
      ++m_current_phase;
    }
    // if we have an internally-ready task, initiate its recvs
    else if (m_detailed_tasks->numInternalReadyTasks() > 0) {
      DetailedTask* task = m_detailed_tasks->getNextInternalReadyTask();
      // save the reduction task and once per proc task for later execution
      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {
        m_phase_sync_tasks[task->getTask()->d_phase] = task;
        ASSERT(task->getRequires().size() == 0)
      }
      else {
        MPIScheduler::initiateTask(task, m_abort, m_abort_point, iteration);
        task->markInitiated();
        task->checkExternalDepCount();
      }
    }
    // if it is time to run reduction or once-per-proc task
    else if ((m_phase_sync_tasks[m_current_phase] != nullptr) && (m_phase_tasks_done[m_current_phase].load(std::memory_order_relaxed) == m_phase_tasks[m_current_phase] - 1)) {
      DetailedTask* sync_task = m_phase_sync_tasks[m_current_phase];
      if (sync_task->getTask()->getType() == Task::Reduction) {
        MPIScheduler::initiateReduction(sync_task);
      }
      else {  // Task::OncePerProc task
        ASSERT(sync_task->getTask()->usesMPI());
        MPIScheduler::initiateTask(sync_task, m_abort, m_abort_point, iteration);
        sync_task->markInitiated();
        ASSERT(sync_task->getExternalDepCount() == 0)
        MPIScheduler::runTask(sync_task, iteration, Impl::t_tid);
      }
      ASSERT(sync_task->getTask()->d_phase == m_current_phase);
      m_num_tasks_done.fetch_add(1, std::memory_order_relaxed);
      m_phase_tasks_done[sync_task->getTask()->d_phase].fetch_add(1, std::memory_order_relaxed);
    }
    else if (m_detailed_tasks->numExternalReadyTasks() > 0) {
      DetailedTask* task = m_detailed_tasks->getNextExternalReadyTask();
      ASSERTEQ(task->getExternalDepCount(), 0);
      m_task_pool.insert(task);
    }
    else { // nothing to do process MPI
      processMPIRecvs(TEST);
    }
  }


  //------------------------------------------------------------------------------------------------
  // deactivate TaskRunners
  //------------------------------------------------------------------------------------------------
  Impl::g_run_tasks = 0;

  Impl::thread_fence();

  for (int i = 1; i < m_num_threads; ++i) {
    Impl::g_thread_states[i] = Impl::ThreadState::Inactive;
  }
  //------------------------------------------------------------------------------------------------

  emitNetMPIStats();

  // compute the net timings and add in wait times for all TaskRunner threads
  if (d_sharedState != 0) {
    computeNetRunTimeStats(d_sharedState->d_runTimeStats);
    double thread_wait_time = 0.0;
    for (int i = 1; i < m_num_threads; ++i) {
      thread_wait_time += Impl::g_runners[i]->m_task_wait_time.nanoseconds();
    }
    d_sharedState->d_runTimeStats[SimulationState::TaskWaitThreadTime] += (thread_wait_time / (m_num_threads - 1) );
  }

  // Copy the restart flag to all processors
  reduceRestartFlag(tgnum);

  finalizeTimestep();

  log.finishTimestep();

  if (!parentScheduler_) {  // only do on toplevel scheduler
    outputTimingStats("ThreadFunnledScheduler");
  }
} // end execute()


//______________________________________________________________________
//
void ThreadFunneledScheduler::select_tasks()
{
  while ( Impl::g_run_tasks ) {
    TaskPool::iterator iter = m_task_pool.find_any();
    if (iter) {
      DetailedTask* ready_task = *iter;
      MPIScheduler::runTask(ready_task, Impl::t_tid);
      m_task_pool.erase(iter);
      m_num_tasks_done.fetch_add(1, std::memory_order_relaxed);
      m_phase_tasks_done[ready_task->getTask()->d_phase].fetch_add(1, std::memory_order_relaxed);
    }
  }
}


//______________________________________________________________________
//
void ThreadFunneledScheduler::set_runner( TaskRunner * runner, int tid )
{
  Impl::g_runners[tid] = runner;
  std::atomic_thread_fence(std::memory_order_seq_cst);
}


//______________________________________________________________________
//
void ThreadFunneledScheduler::init_threads(ThreadFunneledScheduler * sched, int num_threads )
{
  Impl::init_threads(sched, num_threads);
}


//______________________________________________________________________
//
void TaskRunner::run() const
{
  m_scheduler->select_tasks();
}

