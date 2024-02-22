/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Parallel/CommunicationList.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Timers/Timers.hpp>

#include <atomic>
#include <cstring>
#include <iomanip>
#include <thread>


using namespace Uintah;

//______________________________________________________________________
//
namespace Uintah {

  extern Dout g_task_dbg;
  extern Dout g_task_run;
  extern Dout g_task_order;
  extern Dout g_exec_out;

  extern Dout do_task_exec_stats;
}


namespace {

  Dout g_dbg(         "Unified_DBG"        , "UnifiedScheduler", "general debugging info for the UnifiedScheduler"      , false );
  Dout g_queuelength( "Unified_QueueLength", "UnifiedScheduler", "report the task queue length for the UnifiedScheduler", false );

  Dout g_thread_stats     ( "Unified_ThreadStats",    "UnifiedScheduler", "Aggregated MPI thread stats for the UnifiedScheduler", false );
  Dout g_thread_indv_stats( "Unified_IndvThreadStats","UnifiedScheduler", "Individual MPI thread stats for the UnifiedScheduler", false );

  Uintah::MasterLock g_scheduler_mutex{};           // main scheduler lock for multi-threaded task selection
  Uintah::MasterLock g_mark_task_consumed_mutex{};  // allow only one task at a time to enter the task consumed section
  Uintah::MasterLock g_lb_mutex{};                  // load balancer lock

} // namespace


//______________________________________________________________________
//
namespace Uintah { namespace Impl {

namespace {

thread_local  int  t_tid = 0;   // unique ID assigned in thread_driver()

}

namespace {

enum class ThreadState : int
{
    Inactive
  , Active
  , Exit
};

UnifiedSchedulerWorker   * g_runners[MAX_THREADS]        = {};
std::vector<std::thread>   g_threads                       {};
volatile ThreadState       g_thread_states[MAX_THREADS]  = {};
int                        g_cpu_affinities[MAX_THREADS] = {};
int                        g_num_threads                 = 0;

std::atomic<int> g_run_tasks{0};


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
  // t_tid is a thread_local variable, unique to each std::thread spawned below
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
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
}


//______________________________________________________________________
// only called by main thread
void init_threads( UnifiedScheduler * sched, int num_threads )
{
  // we now need to refer to the total number of active threads (num_threads + 1)
  g_num_threads = num_threads + 1;

  for (int i = 0; i < g_num_threads; ++i) {
    g_thread_states[i]  = ThreadState::Active;
    g_cpu_affinities[i] = i;
  }

  // set main thread's affinity and tid, core-0 and tid-0, respectively
  set_affinity(g_cpu_affinities[0]);
  t_tid = 0;

  // TaskRunner threads start at g_runners[1], and std::threads start at g_threads[1]
  for (int i = 1; i < g_num_threads; ++i) {
    g_runners[i] = new UnifiedSchedulerWorker(sched, i, g_cpu_affinities[i]);
    Impl::g_threads.push_back(std::thread(thread_driver, i));
  }

  for (auto& t : Impl::g_threads) {
    if (t.joinable()) {
      t.detach();
    }
  }

  thread_fence();
}

} // namespace
}} // namespace Uintah::Impl


//______________________________________________________________________
//
UnifiedScheduler::UnifiedScheduler( const ProcessorGroup   * myworld
                                  ,       UnifiedScheduler * parentScheduler
                                  )
  : MPIScheduler(myworld, parentScheduler)
{
}


//______________________________________________________________________
//
UnifiedScheduler::~UnifiedScheduler()
{
}


//______________________________________________________________________
//
void
UnifiedScheduler::problemSetup( const ProblemSpecP     & prob_spec
                              , const MaterialManagerP & materialManager
                              )
{
  // Default taskReadyQueueAlg
  std::string taskQueueAlg = "";
  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
    if (taskQueueAlg == "") {
      taskQueueAlg = "MostMessages";  //default taskReadyQueueAlg
    }
    if (taskQueueAlg == "FCFS") {
      m_task_queue_alg = FCFS;
    }
    else if (taskQueueAlg == "Stack") {
      m_task_queue_alg = Stack;
    }
    else if (taskQueueAlg == "Random") {
      m_task_queue_alg = Random;
    }
    else if (taskQueueAlg == "MostChildren") {
      m_task_queue_alg = MostChildren;
    }
    else if (taskQueueAlg == "LeastChildren") {
      m_task_queue_alg = LeastChildren;
    }
    else if (taskQueueAlg == "MostAllChildren") {
      m_task_queue_alg = MostAllChildren;
    }
    else if (taskQueueAlg == "LeastAllChildren") {
      m_task_queue_alg = LeastAllChildren;
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
    else {
      throw ProblemSetupException("Unknown task ready queue algorithm", __FILE__, __LINE__);
    }
  }
  proc0cout << "Using \"" << taskQueueAlg << "\" task queue priority algorithm" << std::endl;

  int num_threads = Uintah::Parallel::getNumThreads() - 1;

  if ( (num_threads < 0) &&  Uintah::Parallel::usingDevice() ) {
    if (d_myworld->myRank() == 0) {
      std::cerr << "Error: no thread number specified for Unified Scheduler"
          << std::endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads to be in the range [1, 64],\n.... please use -nthreads <num>, and -gpu if using GPUs",
          __FILE__, __LINE__);
    }
  } else if (num_threads > MAX_THREADS) {
    if (d_myworld->myRank() == 0) {
      std::cerr << "Error: Number of threads too large..." << std::endl;
      throw ProblemSetupException(
          "Too many threads. Reduce MAX_THREADS and recompile.", __FILE__,
          __LINE__);
    }
  }

  if (d_myworld->myRank() == 0) {
    std::string plural = (num_threads == 1) ? " thread" : " threads";
    std::cout
        << "\nWARNING: Multi-threaded Unified scheduler is EXPERIMENTAL, not all tasks are thread safe yet.\n"
        << "Creating " << num_threads << " additional "
        << plural + " for task execution (total task execution threads = "
        << num_threads + 1 << ").\n" << std::endl;
  }

  SchedulerCommon::problemSetup(prob_spec, materialManager);

  // This spawns threads, sets affinity, etc
  init_threads(this, num_threads);

  // Setup the thread info mapper
  if( g_thread_stats || g_thread_indv_stats ) {
    m_thread_info.resize( Impl::g_num_threads );
    m_thread_info.setIndexName( "Thread" );
    m_thread_info.insert( WaitTime  , std::string("WaitTime")  , "seconds" );
    m_thread_info.insert( LocalTID  , std::string("LocalTID")  , "Index"   );
    m_thread_info.insert( Affinity  , std::string("Affinity")  , "CPU"     );
    m_thread_info.insert( NumTasks  , std::string("NumTasks")  , "tasks"   );
    m_thread_info.insert( NumPatches, std::string("NumPatches"), "patches" );

    m_thread_info.calculateMinimum(true);
    m_thread_info.calculateStdDev (true);
  }
}

//______________________________________________________________________
//
SchedulerP
UnifiedScheduler::createSubScheduler()
{
  return MPIScheduler::createSubScheduler();
}


//______________________________________________________________________
//
void
UnifiedScheduler::runTask( DetailedTask  * dtask
                         , int             iteration
                         , int             thread_id /* = 0 */
                         , CallBackEvent   event
                         )
{
  // end of per-thread wait time - how long has a thread waited before executing another task
  if (thread_id > 0) {
    Impl::g_runners[thread_id]->stopWaitTime();

    if( g_thread_stats || g_thread_indv_stats ) {
      m_thread_info[thread_id][NumTasks] += 1;

      const PatchSubset *patches = dtask->getPatches();
      if (patches)
        m_thread_info[thread_id][NumPatches] += patches->size();
    }
  }

  // Only execute CPU or GPU tasks.  Don't execute postGPU tasks a second time.
  if ( event == CallBackEvent::CPU || event == CallBackEvent::GPU) {

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_BEFORE_EXEC) {
      printTrackedVars(dtask, SchedulerCommon::PRINT_BEFORE_EXEC);
    }

    std::vector<DataWarehouseP> plain_old_dws(m_dws.size());
    for (int i = 0; i < static_cast<int>(m_dws.size()); i++) {
      plain_old_dws[i] = m_dws[i].get_rep();
    }

    DOUT(g_task_run, myRankThread() << " Running task:   " << *dtask);

    dtask->doit(d_myworld, m_dws, plain_old_dws, event);

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_AFTER_EXEC) {
      printTrackedVars(dtask, SchedulerCommon::PRINT_AFTER_EXEC);
    }
  }

  // For CPU task runs, post MPI sends and call task->done;
  if (event == CallBackEvent::CPU || event == CallBackEvent::postGPU) {

    MPIScheduler::postMPISends(dtask, iteration);

    dtask->done(m_dws);

    g_lb_mutex.lock();
    {
      // Do the global and local per task monitoring
      sumTaskMonitoringValues( dtask );

      double total_task_time = dtask->task_exec_time();
      if (g_exec_out || do_task_exec_stats) {
        m_task_info[dtask->getTask()->getName()][TaskStatsEnum::ExecTime] += total_task_time;
        m_task_info[dtask->getTask()->getName()][TaskStatsEnum::WaitTime] += dtask->task_wait_time();
      }

      // if I do not have a sub scheduler
      if (!dtask->getTask()->getHasSubScheduler()) {
        //add my task time to the total time
        m_mpi_info[TotalTask] += total_task_time;
        if (!m_is_copy_data_timestep &&
            dtask->getTask()->getType() != Task::Output) {
          //add contribution for patchlist
          m_loadBalancer->addContribution(dtask, total_task_time);
        }
      }
    }
    g_lb_mutex.unlock();

    //---------------------------------------------------------------------------
    // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
    //---------------------------------------------------------------------------
    // test a pending request
    auto ready_request = [](CommRequest const& r)->bool {return r.test();};
    CommRequestPool::handle find_handle;
    CommRequestPool::iterator comm_sends_iter = m_sends.find_any(find_handle, ready_request);
    if (comm_sends_iter) {
      MPI_Status status;
      comm_sends_iter->finishedCommunication(d_myworld, status);
      find_handle = comm_sends_iter;
      m_sends.erase(comm_sends_iter);
    }
    //---------------------------------------------------------------------------


    // Add subscheduler timings to the parent scheduler and reset subscheduler timings
    if (m_parent_scheduler != nullptr) {
      for (size_t i = 0; i < m_mpi_info.size(); ++i) {
        m_parent_scheduler->m_mpi_info[i] += m_mpi_info[i];
      }
      m_mpi_info.reset(0);
      m_thread_info.reset( 0 );
    }
  }

  // beginning of per-thread wait time... until executing another task
  if (thread_id > 0) {
    Impl::g_runners[thread_id]->startWaitTime();
  }

}  // end runTask()

//______________________________________________________________________
//
void
UnifiedScheduler::execute( int tgnum       /* = 0 */
                         , int iteration   /* = 0 */
                         )
{
  // copy data timestep must be single threaded for now and
  //  also needs to run deterministically, in a static order
  if (m_is_copy_data_timestep) {
    MPIScheduler::execute( tgnum, iteration );
    return;
  }

  // track total scheduler execution time across timesteps
  m_exec_timer.reset(true);

  // If doing in situ monitoring clear the times before each time step
  // otherwise the times are accumulated over N time steps.
  if (do_task_exec_stats) {
    m_task_info.reset(0);
  }

  RuntimeStats::initialize_timestep( m_num_schedulers, m_task_graphs );

  ASSERTRANGE(tgnum, 0, static_cast<int>(m_task_graphs.size()));
  TaskGraph* tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;

  if (m_task_graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(m_dwmap);
  }

  m_detailed_tasks = tg->getDetailedTasks();

  if (m_detailed_tasks == nullptr) {
    proc0cout << "UnifiedScheduler skipping execute, no tasks\n";
    return;
  }

  m_detailed_tasks->initializeScrubs(m_dws, m_dwmap);
  m_detailed_tasks->initTimestep();

  m_num_tasks = m_detailed_tasks->numLocalTasks();

  if( m_runtimeStats )
    (*m_runtimeStats)[RuntimeStatsEnum::NumTasks] += m_num_tasks;

  for (int i = 0; i < m_num_tasks; i++) {
    m_detailed_tasks->localTask(i)->resetDependencyCounts();
  }

  int my_rank = d_myworld->myRank();

  // This only happens if "-emit_taskgraphs" is passed to sus
  makeTaskGraphDoc(m_detailed_tasks, my_rank);

  m_mpi_info.reset( 0 );
  m_thread_info.reset( 0 );

  m_num_tasks_done = 0;
  m_abort = false;
  m_abort_point = 987654;

  if( m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr ) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(m_detailed_tasks, m_loadBalancer, m_reloc_new_pos_label, iteration);
  }

  m_curr_iteration = iteration;
  m_curr_phase = 0;
  m_num_phases = tg->getNumTaskPhases();
  m_phase_tasks.clear();
  m_phase_tasks.resize(m_num_phases, 0);
  m_phase_tasks_done.clear();
  m_phase_tasks_done.resize(m_num_phases, 0);
  m_phase_sync_task.clear();
  m_phase_sync_task.resize(m_num_phases, nullptr);
  m_detailed_tasks->setTaskPriorityAlg(m_task_queue_alg);

  // get the number of tasks in each task phase
  for (int i = 0; i < m_num_tasks; i++) {
    m_phase_tasks[m_detailed_tasks->localTask(i)->getTask()->m_phase]++;
  }

  if (g_dbg) {
    std::ostringstream message;
    message << "\n" << "Rank-" << my_rank << " Executing " << m_detailed_tasks->numTasks() << " tasks (" << m_num_tasks
            << " local)\n" << "Total task phases: " << m_num_phases << "\n";
    for (size_t phase = 0; phase < m_phase_tasks.size(); ++phase) {
      message << "Phase: " << phase << " has " << m_phase_tasks[phase] << " total tasks\n";
    }
    DOUT(true, message.str());
  }

  static int totaltasks;

  Impl::thread_fence();

  //------------------------------------------------------------------------------------------------
  // activate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!m_is_copy_data_timestep) {
    Impl::g_run_tasks.store(1, std::memory_order_relaxed);
    for (int i = 1; i < Impl::g_num_threads; ++i) {
      Impl::g_thread_states[i] = Impl::ThreadState::Active;
    }
  }
  //------------------------------------------------------------------------------------------------


  // main thread also executes tasks
  runTasks( Impl::t_tid );


  //------------------------------------------------------------------------------------------------
  // deactivate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!m_is_copy_data_timestep) {
    Impl::g_run_tasks.store(0, std::memory_order_relaxed);

    Impl::thread_fence();

    for (int i = 1; i < Impl::g_num_threads; ++i) {
      Impl::g_thread_states[i] = Impl::ThreadState::Inactive;
    }
  }
  //------------------------------------------------------------------------------------------------


  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
  //---------------------------------------------------------------------------
  // wait on all pending requests
  auto ready_request = [](CommRequest const& r)->bool { return r.wait(); };
  CommRequestPool::handle find_handle;
  while ( m_sends.size() != 0u ) {
    CommRequestPool::iterator comm_sends_iter;
    if ( (comm_sends_iter = m_sends.find_any(find_handle, ready_request)) ) {
      find_handle = comm_sends_iter;
      m_sends.erase(comm_sends_iter);
    } else {
      // TODO - make this a sleep? APH 07/20/16
    }
  }
  //---------------------------------------------------------------------------

  ASSERT(m_sends.size() == 0u);
  ASSERT(m_recvs.size() == 0u);


  if (g_queuelength) {
    float lengthsum = 0;
    totaltasks += m_num_tasks;
    for (unsigned int i = 1; i < m_histogram.size(); i++) {
      lengthsum = lengthsum + i * m_histogram[i];
    }

    float queuelength = lengthsum / totaltasks;
    float allqueuelength = 0;
    Uintah::MPI::Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());

    proc0cout << "average queue length:" << allqueuelength / d_myworld->nRanks() << std::endl;
  }

  finalizeTimestep();

  m_exec_timer.stop();

  // compute the net timings
  if( m_runtimeStats ) {

    // Stats specific to this threaded scheduler - TaskRunner threads start at g_runners[1]
    for (int i = 1; i < Impl::g_num_threads; ++i) {
      (*m_runtimeStats)[TaskWaitThreadTime] += Impl::g_runners[i]->getWaitTime();

//      DOUT(true, "ThreadID: " << Impl::g_runners[i]->getLocalTID() << ", bound to core: " << Impl::g_runners[i]->getAffinity());

      if( g_thread_stats || g_thread_indv_stats ) {
        m_thread_info[i][WaitTime] = Impl::g_runners[i]->getWaitTime();
        m_thread_info[i][LocalTID] = Impl::g_runners[i]->getLocalTID();
        m_thread_info[i][Affinity] = Impl::g_runners[i]->getAffinity();
      }
    }

    MPIScheduler::computeNetRuntimeStats();
  }

  // Thread average runtime performance stats.
  if (g_thread_stats ) {
    m_thread_info.reduce( false ); // true == skip the first entry.

    m_thread_info.reportSummaryStats( "Thread", "",
                                      d_myworld->myRank(),
                                      d_myworld->nRanks(),
                                      m_application->getTimeStep(),
                                      m_application->getSimTime(),
                                      BaseInfoMapper::Dout, false );
  }

  // Per thread runtime performance stats
  if (g_thread_indv_stats) {
    m_thread_info.reportIndividualStats( "Thread", "",
                                         d_myworld->myRank(),
                                         d_myworld->nRanks(),
                                         m_application->getTimeStep(),
                                         m_application->getSimTime(),
                                         BaseInfoMapper::Dout );
  }

  // only do on toplevel scheduler
  if (m_parent_scheduler == nullptr) {
    MPIScheduler::outputTimingStats("UnifiedScheduler");
  }

  RuntimeStats::report(d_myworld->getComm());

} // end execute()


//______________________________________________________________________
//
void
UnifiedScheduler::markTaskConsumed( int          & numTasksDone
                                  , int          & currphase
                                  , int            numPhases
                                  , DetailedTask * dtask
                                  )
{
  std::lock_guard<Uintah::MasterLock> task_consumed_guard(g_mark_task_consumed_mutex);

  // Update the count of tasks consumed by the scheduler.
  numTasksDone++;

  // task ordering debug info - please keep this here, APH 05/30/18
  if (g_task_order && d_myworld->myRank() == d_myworld->nRanks() / 2) {
    std::ostringstream task_name;
    task_name << "  Running task: \"" << dtask->getTask()->getName() << "\" ";

    std::ostringstream task_type;
    task_type << "(" << dtask->getTask()->getType() << ") ";

    // task ordering debug info - please keep this here, APH 05/30/18
    DOUT(true, "Rank-" << d_myworld->myRank()
                       << std::setw(60) << std::left << task_name.str()
                       << std::setw(14) << std::left << task_type.str()
                       << std::setw(15) << " static order: "    << std::setw(3) << std::left << dtask->getStaticOrder()
                       << std::setw(18) << " scheduled order: " << std::setw(3) << std::left << numTasksDone);
  }

  // Update the count of this phase consumed.
  m_phase_tasks_done[dtask->getTask()->m_phase]++;

  // See if we've consumed all tasks on this phase, if so, go to the next phase.
  while (m_phase_tasks[currphase] == m_phase_tasks_done[currphase] && currphase + 1 < numPhases) {
    currphase++;
    DOUT(g_task_dbg, myRankThread() << " switched to task phase " << currphase
                                    << ", total phase " << currphase << " tasks = " << m_phase_tasks[currphase]);
  }
}

//______________________________________________________________________
//
void
UnifiedScheduler::runTasks( int thread_id )
{

  while( m_num_tasks_done < m_num_tasks ) {

    DetailedTask* readyTask = nullptr;
    DetailedTask* initTask  = nullptr;

    bool havework = false;

    // ----------------------------------------------------------------------------------
    // Part 1:
    //    Check if anything this thread can do concurrently.
    //    If so, then update the various scheduler counters.
    // ----------------------------------------------------------------------------------
    //g_scheduler_mutex.lock();
    while (!havework) {
      /*
       * (1.1)
       *
       * If it is time to setup for a reduction task, then do so.
       *
       */

      if ((m_phase_sync_task[m_curr_phase] != nullptr) && (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
        g_scheduler_mutex.lock();
        if ((m_phase_sync_task[m_curr_phase] != nullptr) && (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
          readyTask = m_phase_sync_task[m_curr_phase];
          havework = true;
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
        }
        g_scheduler_mutex.unlock();
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
      else if ((readyTask = m_detailed_tasks->getNextExternalReadyTask())) {
        havework = true;
        // A CPU task and we can mark the task consumed
        markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
        break;
      }

      /*
       * (1.3)
       *
       * If we have an internally-ready CPU task, initiate its MPI receives, preparing it for
       * CPU external ready queue. The task is moved to the CPU external-ready queue in the
       * call to task->checkExternalDepCount().
       *
       */
      else if ((initTask = m_detailed_tasks->getNextInternalReadyTask())) {
        if (initTask->getTask()->getType() == Task::Reduction || initTask->getTask()->usesMPI()) {
          DOUT(g_task_dbg, myRankThread() <<  " Task internal ready 1 " << *initTask);
          m_phase_sync_task[initTask->getTask()->m_phase] = initTask;
          ASSERT(initTask->getRequires().size() == 0)
          initTask = nullptr;
        }
        else if (initTask->getRequires().size() == 0) {  // no ext. dependencies, then skip MPI sends
          initTask->markInitiated();
          initTask->checkExternalDepCount();  // where tasks get added to external ready queue (no ext deps though)
          initTask = nullptr;
        }
        else {
          havework = true;
          break;
         }
      }
      /*
       * (1.6)
       *
       * Otherwise there's nothing to do but process MPI recvs.
       */
      if (!havework) {
        if (m_recvs.size() != 0u) {
          havework = true;
          break;
        }
      }
      if (m_num_tasks_done == m_num_tasks) {
        break;
      }
    } // end while (!havework)
    //g_scheduler_mutex.unlock();


    // ----------------------------------------------------------------------------------
    // Part 2
    //    Concurrent Part:
    //      Each thread does its own thing here... modify this code with caution
    // ----------------------------------------------------------------------------------

    if (initTask != nullptr) {
      MPIScheduler::initiateTask(initTask, m_abort, m_abort_point, m_curr_iteration);

      DOUT(g_task_dbg, myRankThread() << " Task internal ready 2 " << *initTask << " deps needed: " << initTask->getExternalDepCount());

      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask) {

      DOUT(g_task_dbg, myRankThread() << " Task external ready " << *readyTask)

      if (readyTask->getTask()->getType() == Task::Reduction) {
        MPIScheduler::initiateReduction(readyTask);
      }
      else {
        // run CPU task.
        runTask(readyTask, m_curr_iteration, thread_id, CallBackEvent::CPU);
      }
    }
    else {
      if (m_recvs.size() != 0u) {
        MPIScheduler::processMPIRecvs(TEST);
      }
    }
  }  //end while (numTasksDone < ntasks)
  ASSERT(m_num_tasks_done == m_num_tasks);
}


//______________________________________________________________________
//  generate string   <MPI_rank>.<Thread_ID>
std::string
UnifiedScheduler::myRankThread()
{
  std::ostringstream out;
  out << Uintah::Parallel::getMPIRank() << "." << Impl::t_tid;
  return out.str();
}


//______________________________________________________________________
//
void
UnifiedScheduler::init_threads( UnifiedScheduler * sched, int num_threads )
{
  Impl::init_threads(sched, num_threads);
}


//------------------------------------------
// UnifiedSchedulerWorker Thread Methods
//------------------------------------------
UnifiedSchedulerWorker::UnifiedSchedulerWorker( UnifiedScheduler * scheduler, int tid, int affinity )
  : m_scheduler{ scheduler }
  , m_rank{ scheduler->d_myworld->myRank() }
  , m_tid{ tid }
  , m_affinity{ affinity }
{
}


//______________________________________________________________________
//
void
UnifiedSchedulerWorker::run()
{
  while( Impl::g_run_tasks.load(std::memory_order_relaxed) == 1 ) {
    try {
      resetWaitTime();
      m_scheduler->runTasks(Impl::t_tid);
    }
    catch (Exception& e) {
      DOUT(true, "Worker " << m_rank << "-" << Impl::t_tid << ": Caught exception: " << e.message());
      if (e.stackTrace()) {
        DOUT(true, "Stack trace: " << e.stackTrace());
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedSchedulerWorker::resetWaitTime()
{
  m_wait_timer.reset( true );
  m_wait_time = 0.0;
}


//______________________________________________________________________
//
void
UnifiedSchedulerWorker::startWaitTime()
{
  m_wait_timer.start();
}


//______________________________________________________________________
//
void
UnifiedSchedulerWorker::stopWaitTime()
{
  m_wait_timer.stop();
  m_wait_time += m_wait_timer().seconds();
}


//______________________________________________________________________
//
const double
UnifiedSchedulerWorker::getWaitTime() const
{
  return m_wait_time;
}

//______________________________________________________________________
//
const int
UnifiedSchedulerWorker::getAffinity() const
{
  return m_affinity;
}

//______________________________________________________________________
//
const int
UnifiedSchedulerWorker::getLocalTID() const
{
  return m_tid;
}
