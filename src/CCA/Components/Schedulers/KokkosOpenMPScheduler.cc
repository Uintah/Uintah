/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Schedulers/KokkosOpenMPScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
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

#include <sci_defs/kokkos_defs.h>

#if defined( KOKKOS_ENABLE_OPENMP )
  #include <Kokkos_Core.hpp>
#endif //KOKKOS_ENABLE_OPENMP

#include <atomic>
#include <cstring>
#include <iomanip>


using namespace Uintah;


//______________________________________________________________________
//
namespace Uintah {
  extern Dout g_task_dbg;
}


//______________________________________________________________________
//
namespace {

  Dout g_dbg(         "KokkosOMP_DBG"        , "KokkosOpenMPScheduler", "general debugging info for KokkosOpenMPScheduler"  , false );
  Dout g_queuelength( "KokkosOMP_QueueLength", "KokkosOpenMPScheduler", "report task queue length for KokkosOpenMPScheduler", false );

  Uintah::MasterLock g_scheduler_mutex{}; // main scheduler lock for multi-threaded task selection
  Uintah::MasterLock g_mark_task_consumed_mutex{};  // allow only one task at a time to enter the task consumed section

  volatile int  g_num_tasks_done{0};

  bool g_have_hypre_task{false};
  DetailedTask* g_HypreTask;

}


//______________________________________________________________________
//
KokkosOpenMPScheduler::KokkosOpenMPScheduler( const ProcessorGroup   * myworld
                                            ,       KokkosOpenMPScheduler * parentScheduler
                                            )
  : MPIScheduler(myworld, parentScheduler)
{
}


//______________________________________________________________________
//
void
KokkosOpenMPScheduler::problemSetup( const ProblemSpecP     & prob_spec
                                   , const MaterialManagerP & materialManager
                                   )
{

  m_num_partitions        = Uintah::Parallel::getNumPartitions();
  m_threads_per_partition = Uintah::Parallel::getThreadsPerPartition();

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

  if (d_myworld->myRank() == 0) {
    std::cout << "   WARNING: Kokkos-OpenMP Scheduler is EXPERIMENTAL, not all tasks are Kokkos-enabled yet." << std::endl;
  }

  SchedulerCommon::problemSetup(prob_spec, materialManager);
}


//______________________________________________________________________
//
SchedulerP
KokkosOpenMPScheduler::createSubScheduler()
{
  return MPIScheduler::createSubScheduler();
}


//______________________________________________________________________
//
void
KokkosOpenMPScheduler::execute( int tgnum       /* = 0 */
                              , int iteration   /* = 0 */
                              )
{
  // copy data timestep must be single threaded for now and
  //  also needs to run deterministically, in a static order
  if (m_is_copy_data_timestep) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  // track total scheduler execution time across timesteps
  m_exec_timer.reset(true);

  RuntimeStats::initialize_timestep(m_task_graphs);

  ASSERTRANGE(tgnum, 0, static_cast<int>(m_task_graphs.size()));
  TaskGraph* tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;

  if (m_task_graphs.size() > 1) {
    // TG model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(m_dwmap);
  }

  m_detailed_tasks = tg->getDetailedTasks();

  if (m_detailed_tasks == 0) {
    proc0cout << "KokkosOpenMPScheduler skipping execute, no tasks\n";
    return;
  }

  m_detailed_tasks->initializeScrubs(m_dws, m_dwmap);
  m_detailed_tasks->initTimestep();

  m_num_tasks = m_detailed_tasks->numLocalTasks();
  for (int i = 0; i < m_num_tasks; i++) {
    m_detailed_tasks->localTask(i)->resetDependencyCounts();
  }

  int my_rank = d_myworld->myRank();

  // This only happens if "-emit_taskgraphs" is passed to sus
  makeTaskGraphDoc(m_detailed_tasks, my_rank);

  m_mpi_info.reset( 0 );

  g_num_tasks_done = 0;

  if( m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr ) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(m_detailed_tasks, m_loadBalancer, m_reloc_new_pos_label, iteration);
  }

  m_curr_iteration.store(iteration, std::memory_order_relaxed);
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


//---------------------------------------------------------------------------

  while ( g_num_tasks_done < m_num_tasks ) {

#if defined( KOKKOS_ENABLE_OPENMP )

    auto task_worker = [&] ( int partition_id, int num_partitions ) {

      // Each partition created executes this block of code
      // A task_worker can run either a serial task, e.g. threads_per_partition == 1
      //       or a Kokkos-based data parallel task, e.g. threads_per_partition > 1

      this->runTasks();

    }; //end task_worker

    // Executes task_workers
    Kokkos::OpenMP::partition_master( task_worker
                                    , m_num_partitions
                                    , m_threads_per_partition );

#else //KOKKOS_ENABLE_OPENMP

    this->runTasks();

#endif // UINTAH_ENABLE_KOKKOS

    if ( g_have_hypre_task ) {
      DOUT( g_dbg, " Exited runTasks to run a " << g_HypreTask->getTask()->getType() << " task" );
      MPIScheduler::runTask( g_HypreTask, m_curr_iteration.load(std::memory_order_relaxed) );
      g_have_hypre_task = false;
    }

  } // end while ( g_num_tasks_done < m_num_tasks )

//---------------------------------------------------------------------------


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
  MPIScheduler::computeNetRuntimeStats();

  // only do on toplevel scheduler
  if (m_parent_scheduler == nullptr) {
    MPIScheduler::outputTimingStats("KokkosOpenMPScheduler");
  }

  RuntimeStats::report(d_myworld->getComm());

} // end execute()


//______________________________________________________________________
//
void
KokkosOpenMPScheduler::markTaskConsumed( volatile int          * numTasksDone
                                       ,          int          & currphase
                                       ,          int            numPhases
                                       ,          DetailedTask * dtask
                                       )
{

  std::lock_guard<Uintah::MasterLock> task_consumed_guard(g_mark_task_consumed_mutex);

  // Update the count of tasks consumed by the scheduler.
  (*numTasksDone)++;

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
KokkosOpenMPScheduler::runReadyTask( DetailedTask* readyTask )
{
  if (readyTask->getTask()->getType() == Task::Reduction) {
    MPIScheduler::initiateReduction(readyTask);
  }
  else {
    MPIScheduler::runTask(readyTask, m_curr_iteration.load(std::memory_order_relaxed));
  }
}


//______________________________________________________________________
//
void
KokkosOpenMPScheduler::runTasks()
{
  while( g_num_tasks_done < m_num_tasks && !g_have_hypre_task ) {

    DetailedTask* readyTask = nullptr;
    DetailedTask* initTask  = nullptr;

    bool havework = false;

    // ----------------------------------------------------------------------------------
    // Part 1 (serial): Task selection
    // ----------------------------------------------------------------------------------
    {
      std::lock_guard<Uintah::MasterLock> scheduler_mutex_guard(g_scheduler_mutex);

      while ( !havework ) {

        /*
         * (1.0)
         *
         * If it is time for a Hypre task, exit partitions.
         *
         */
        if ( g_have_hypre_task ) {
          return;
        }

        /*
         * (1.1)
         *
         * If it is time to setup for a reduction task, then do so.
         *
         */
        if ((m_phase_sync_task[m_curr_phase] != nullptr) && (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
          readyTask = m_phase_sync_task[m_curr_phase];
          havework = true;
          markTaskConsumed(&g_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
          break;
        }

        /*
         * (1.2)
         *
         * Run a CPU task that has its MPI communication complete. These tasks get in the external
         * ready queue automatically when their receive count hits 0 in DependencyBatch::received,
         * which is called when a MPI message is delivered.
         *
         */
        else if (m_detailed_tasks->numExternalReadyTasks() > 0) {
          readyTask = m_detailed_tasks->getNextExternalReadyTask();
          if (readyTask != nullptr) {
            havework = true;
            markTaskConsumed(&g_num_tasks_done, m_curr_phase, m_num_phases, readyTask);

            if ( readyTask->getTask()->getType() == Task::Hypre ) {
              g_HypreTask = readyTask;
              g_have_hypre_task = true;
              return;
            }

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
        else if (m_detailed_tasks->numInternalReadyTasks() > 0) {
          initTask = m_detailed_tasks->getNextInternalReadyTask();
          if (initTask != nullptr) {
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
        }

        /*
         * (1.4)
         *
         * Otherwise there's nothing to do but process MPI recvs.
         */
        else {
          if (m_recvs.size() != 0u) {
            havework = true;
            break;
          }
        }

        if (g_num_tasks_done == m_num_tasks) {
          break;
        }
      }  // end while (!havework)
    }  // end lock_guard



    // ----------------------------------------------------------------------------------
    // Part 2 (concurrent): Task execution
    // ----------------------------------------------------------------------------------

    if (initTask != nullptr) {
      MPIScheduler::initiateTask(initTask, m_abort, m_abort_point, m_curr_iteration.load(std::memory_order_relaxed));
      DOUT(g_task_dbg, myRankThread() << " Task internal ready 2 " << *initTask << " deps needed: " << initTask->getExternalDepCount());
      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask) {

      DOUT(g_task_dbg, myRankThread() << " Task external ready " << *readyTask);

      if (readyTask->getTask()->getType() == Task::Reduction) {
        MPIScheduler::initiateReduction(readyTask);
      }
      else {
        MPIScheduler::runTask(readyTask, m_curr_iteration.load(std::memory_order_relaxed));
      }
    }
    else {
      if (m_recvs.size() != 0u) {
        MPIScheduler::processMPIRecvs(TEST);
      }
    }
  }  // end while (numTasksDone < ntasks)
  ASSERT(g_num_tasks_done == m_num_tasks);
}


//______________________________________________________________________
//  generate string   <MPI_rank>.<Thread_ID>
std::string
KokkosOpenMPScheduler::myRankThread()
{
  std::ostringstream out;

#if defined( KOKKOS_ENABLE_OPENMP ) && defined( USING_LATEST_KOKKOS )
  out << Uintah::Parallel::getMPIRank()<< "." << Kokkos::OpenMP::impl_hardware_thread_id();
#elif defined( KOKKOS_ENABLE_OPENMP ) && !defined( USING_LATEST_KOKKOS )
  out << Uintah::Parallel::getMPIRank()<< "." << Kokkos::OpenMP::hardware_thread_id();
#else
  out << Uintah::Parallel::getMPIRank();
#endif

  return out.str();
}
