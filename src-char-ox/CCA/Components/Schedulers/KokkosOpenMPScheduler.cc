/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
                                   , const SimulationStateP & state
                                   )
{

  m_num_partitions        = Uintah::Parallel::getNumPartitions();
  m_threads_per_partition = Uintah::Parallel::getThreadsPerPartition();

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

  proc0cout << "Using \"" << taskQueueAlg << "\" task queue priority algorithm" << std::endl;

  if (d_myworld->myRank() == 0) {
    std::cout << "   WARNING: Kokkos-OpenMP Scheduler is EXPERIMENTAL, not all tasks are Kokkos-enabled yet." << std::endl;
  }

  SchedulerCommon::problemSetup(prob_spec, state);
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

  mpi_info_.reset( 0 );

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

  Timers::Simple upTimer;
  Timers::Simple downTimer;

//---------------------------------------------------------------------------

  while ( g_num_tasks_done < m_num_tasks ) {

#if defined( KOKKOS_ENABLE_OPENMP )

    auto task_worker = [&] ( int partition_id, int num_partitions ) {

      upTimer.stop();

      // Each partition created executes this block of code
      // A task_worker can run either a serial task, e.g. threads_per_partition == 1
      //       or a Kokkos-based data parallel task, e.g. threads_per_partition > 1

      this->runTasks();

      downTimer.start();

    }; //end task_worker

    upTimer.start();

    // Executes task_workers
    Kokkos::OpenMP::partition_master( task_worker
                                    , m_num_partitions
                                    , m_threads_per_partition );

    downTimer.stop();

    printf( " upTime: %g\n", upTimer().microseconds() );
    printf( " downTime: %g\n", downTimer().microseconds() );

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

  if( m_restartable && tgnum == static_cast<int>(m_task_graphs.size()) - 1 ) {
    // Copy the restart flag to all processors
    int myrestart = m_dws[m_dws.size() - 1]->timestepRestarted();
    int netrestart;

    Uintah::MPI::Allreduce( &myrestart, &netrestart, 1, MPI_INT, MPI_LOR, d_myworld->getComm() );

    if( netrestart ) {
      m_dws[m_dws.size() - 1]->restartTimestep();
      if( m_dws[0] ) {
        m_dws[0]->setRestarted();
      }
    }
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
    DOUT(g_dbg, " switched to task phase " << currphase << ", total phase " << currphase << " tasks = " << m_phase_tasks[currphase]);
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


#ifdef BRADS_NEW_DWDATABASE

//______________________________________________________________________
//
void
KokkosOpenMPScheduler::runTasks()
{

// This runTasks loop manages tasks that vary on these parameters:
// Are task variables in a homogeneous only environment (e.g. host memory) or heterogeneous (host memory + GPU memory)?
// Is the task computing on a CPU or a GPU?
// Did the task request to preload variables prior to task execution (new) or load them at task runtime (legacy)?
// The above decision tree would create 8 scenarios.  However, GPU tasks don't have a decision tree, they only support
// heterogeneous memory spaces and simulation variable preloading.  Therefore, the decision tree creates 5 scenarios.
// The chart below describes all 5 scenarios.  It also describes all possible phases the task proceeds through.
// ---------------------------------------------------------------------------------------------------------------------------
//                        CPU Task                         |   GPU Task    |
//         Homogeneous        |         Heterogeneous      | Heterogeneous | Scheduler Phase
// No Preloading | Preloading | No Preloading | Preloading |  Preloading   |
//   (Legacy)    | (New)      |   (Legacy)    | (New)      |               |
// -----------------------------------------------------------------------------------------------------------------------------------
//                                    ALL                                  | (1.1) Reduction Task.  (Run 1.5 *EXECUTE TASK*)
// -----------------------------------------------------------------------------------------------------------------------------------
//                                    ALL                                  | (1.3) Internally ready.  If it's a reduction, send it to (1.1)
//                                                                         |       Otherwise, initiate MPI receives, when complete
//                                                                         |       these tasks go into the externally ready queue (1.2)
// -----------------------------------------------------------------------------------------------------------------------------------
//                                    ALL                                  | (1.2) Externally ready. All MPI received have arrived.
//                                                                         |       From here tasks proceed according to the chart.
// -----------------------------------------------------------------------------------------------------------------------------------
//               |     X      |               |      X     |       X       | (1.4.0) Preallocate space for all computes for the task.
//                                                                         |         States that can be involved: (ALLOCATING, ALLOCATED, BECOMING_VALID)
// -----------------------------------------------------------------------------------------------------------------------------------
//               |            |        X      |      X     |       X       | (1.4.1) Copy task variables into the task's memory space.
//               |            |               |            |               |         This includes all halo data as well.
//               |            |               |            |               |         Assign GPU streams to the task.
//               |            |               |            |               |         States that can be involved: (ALLOCATING, ALLOCATED, BECOMING_VALID)
// -----------------------------------------------------------------------------------------------------------------------------------
//               |            |        X      |      X     |       X       | (1.4.2) Check the stream if GPU.
//               |            |               |            |               |         Mark all vars it copies as (VALID)
// -----------------------------------------------------------------------------------------------------------------------------------
//               |     X      |               |      X     |       X       | (1.4.3) Perform all ghost cell copies if possible and go to (1.4.4)
//               |            |               |            |               |         This step would also perform variable resizing and superpatching.
//               |            |               |            |               |         Otherwise, put back in (1.4.3) and check later.
//               |            |               |            |               |         Many states can be involved here.
// -----------------------------------------------------------------------------------------------------------------------------------
//               |     X      |               |      X     |       X       | (1.4.4) Check the stream if GPU.
//               |            |               |            |               |         Mark all vars it managed as (GHOST_VALID)
// -----------------------------------------------------------------------------------------------------------------------------------
//               |     X      |        X      |      X     |       X       | (1.4.5) If task vars not ready, put back in (1.4.5) and check later
//               |            |               |            |               |         If vars ready, go to (1.5)
// -----------------------------------------------------------------------------------------------------------------------------------
//                                    ALL                                  | (1.5)   Mark all modifies as no longer valid but as BECOMING_VALID
//                                                                         |         Increment usage counter of all requires.
//                                                                         |         *** EXECUTE TASK ***
//                                                                         |         Go to (1.5.1)
// -----------------------------------------------------------------------------------------------------------------------------------
//                                    ALL                                  | (1.5.1) Check the stream if GPU.
//                                                                         |         Mark task computes and modifies as VALID.
//                                                                         |         Decrement usage counter of all requires.
//                                                                         |         Mark task as done.  Clean up any temporary/pool variables.
// -----------------------------------------------------------------------------------------------------------------------------------
//                                    ALL                                  | (1.6)   Process MPI receives

//TODO: Processing MPI sends.  Task Data Warehouses.   Changing state of modifies variables.

  while( g_num_tasks_done < m_num_tasks && !g_have_hypre_task ) {

    DetailedTask* readyTask = nullptr;
    DetailedTask* initTask  = nullptr;  //Internally ready task

    bool havework = false;

    bool allocateComputes = false;
    bool runReady = false;
#ifdef HAVE_CUDA
    bool inHeterogeneousEnvironment = Uintah::Parallel::usingDevice();  // If the -gpu flag was passed in.
                                                                        // Not all tasks have to run on the GPU, but some could
#endif

    // ---------------------------------------------------------------------------------------------------------
    // The main work loop.  Modify with caution, as you can easily create concurrency problems/race conditions
    // ---------------------------------------------------------------------------------------------------------

    /*
     * (1.1)
     *
     * If it is time to setup for a reduction task, then do so.
     */
    if ((m_phase_sync_task[m_curr_phase] != nullptr) && (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
      {
        std::lock_guard<Uintah::MasterLock> scheduler_mutex_guard(g_scheduler_mutex);
        //Now that it's locked, do a second look just to make sure this is correct.
        if ((m_phase_sync_task[m_curr_phase] != nullptr) && (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
          readyTask = m_phase_sync_task[m_curr_phase];
        }
      }
      if (readyTask) {
        printf("1.1 The task is %s\n", readyTask->getTask()->getName().c_str());

        markTaskConsumed(&g_num_tasks_done, m_curr_phase, m_num_phases, readyTask);

        //TODO: Are Reduction tasks only phase tasks?
        // ** (1.5) **
        runReadyTask(readyTask);
      }
    }

    /*
     * (1.2)
     *
     * A task has its MPI communication complete. These tasks get in the external
     * ready queue automatically when their receive count hits 0 in DependencyBatch::received,
     * which is called when a MPI message is delivered.
     */
    else if ((readyTask = m_detailed_tasks->getNextExternalReadyTask())) {
      printf("1.2 The task is %s\n", readyTask->getTask()->getName().c_str());

      //bool usesKokkosOpenMP = readyTask->getTask()->usesKokkosCuda();
      bool usesSimVarPreloading = readyTask->getTask()->usesSimVarPreloading();

#ifdef HAVE_CUDA
      if (inHeterogeneousEnvironment == false || readyTask->getPatches() == nullptr) {
        // Uintah is running in a homogeneous/CPU environment or
        // this is a task that can only run on CPUs (no patches).
#endif
        if (usesSimVarPreloading) {
          allocateTaskComputesVariables(readyTask);
          //TODO: Put into a new queue.
        } else {
          markTaskConsumed(&g_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
          if ( readyTask->getTask()->getType() == Task::Hypre ) {
            // hypre tasks need to be run using all OpenMP threads.  Set global g_HypreTask to true so
            // so all threads in this scheduler work loop exit to run the hypre task.
            g_HypreTask = readyTask;
            g_have_hypre_task = true;
            return;
          } else {
            // ** (1.5) **
            runReadyTask(ReadyTask);
          }
        }
#ifdef HAVE_CUDA
      } else {
        //
      }
#endif
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
        std::lock_guard<Uintah::MasterLock> scheduler_mutex_guard(g_scheduler_mutex);
        m_phase_sync_task[initTask->getTask()->m_phase] = initTask;
        ASSERT(initTask->getRequires().size() == 0)
        initTask = nullptr;
      }
      else if (initTask->getRequires().size() == 0) {  // No ext. dependencies, then skip MPI receives
        initTask->markInitiated();
        initTask->checkExternalDepCount();  // Add task to external ready queue (no ext deps though)
        initTask = nullptr;
      }
      else {   //Posts the MPI receives
        MPIScheduler::initiateTask(initTask, m_abort, m_abort_point, m_curr_iteration.load(std::memory_order_relaxed));
        initTask->markInitiated();
        initTask->checkExternalDepCount();  // Add task to external ready queue
      }
    }

    /*
     * (1.4)
     *
     *
     *
     *
     *
     */


    /*
     * (1.6)
     *
     * Otherwise there's nothing to do but process MPI recvs.
     */
    else {
      if (m_recvs.size() != 0u) {
        MPIScheduler::processMPIRecvs(TEST);
      }
    }

  }  // end while (numTasksDone < ntasks)
  ASSERT(g_num_tasks_done == m_num_tasks);
}


//_____________________________________________________________________________________________________________
//
void
KokkosOpenMPScheduler::allocateTaskComputesVariables( DetailedTask * dtask )
{

  const Task* task = dtask->getTask();

  for (const Task::Dependency* computesVar = task->getComputes(); computesVar != 0; computesVar = computesVar->m_next) {
    constHandle<PatchSubset> patches = computesVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = computesVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {

        const TypeDescription::Type type = computesVar->m_var->typeDescription()->getType();

        if (type == TypeDescription::CCVariable
            || type == TypeDescription::NCVariable
            || type == TypeDescription::SFCXVariable
            || type == TypeDescription::SFCYVariable
            || type == TypeDescription::SFCZVariable
            || type == TypeDescription::PerPatch
            || type == TypeDescription::ReductionVariable) {

          //Get patch, level, and material information
          const Patch * patch = patches->get(i);
          if (!patch) {
            printf("ERROR:\nKokkosOpenMPScheduler::allocateTaskComputesVariables() patch not found.\n");
            SCI_THROW( InternalError("Patch not found.", __FILE__, __LINE__));
          }
          const int patchID = patch->getID();
          const Level* level = patch->getLevel();
          int levelID = level->getID();
          if (computesVar->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
            levelID = -1;
          }
          const int matlID = matls->get(j);

          //Determine the variable's cell size and required memory size.
          // a fix for when INF ghost cells are requested such as in RMCRT e.g. tsk->requires(abskg_dw, d_abskgLabel, gac, SHRT_MAX);
          bool uses_SHRT_MAX = (computesVar->m_num_ghost_cells == SHRT_MAX);
          //Get all size information about this dependency.
          IntVector low, high; // lowOffset, highOffset;
          if (uses_SHRT_MAX) {
            level->computeVariableExtents(type, low, high);
          } else {
            Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
            patch->computeVariableExtents(basis, computesVar->m_var->getBoundaryLayer(), computesVar->m_gtype, computesVar->m_num_ghost_cells, low, high);
          }
          const IntVector host_size = high - low;
          const size_t elementDataSize = computesVar->m_var->typeDescription()->getSubType()->getElementDataSize();
          size_t memSize = 0;
          if (type == TypeDescription::PerPatch
              || type == TypeDescription::ReductionVariable) {
            memSize = elementDataSize;
          } else {
            memSize = host_size.x() * host_size.y() * host_size.z() * elementDataSize;
          }

          //Attempt to allocate it, if not already done so.
          const int dwIndex = computesVar->mapDataWarehouse();
          OnDemandDataWarehouseP dw = m_dws[dwIndex];

          MemorySpace ms = dtask->getTask()->getMemorySpace();
          dw->allocateAndPutIfPossible(computesVar->m_var, matlID, patch, low, high, ms);

        } else {
          std::cerr << "KokkosOpenMPScheduler::allocateTaskComputesVariables(), unsupported variable type for computes variable "
                    << computesVar->m_var->getName() << std::endl;
          SCI_THROW(InternalError("Unsupported variable type for computes variable \n",__FILE__, __LINE__));
        }

      }
    }
  }
}

#else  //#ifdef BRADS_NEW_DWDATABASE

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
      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask) {

      DOUT(g_dbg, " Task now external ready: " << *readyTask);

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

#endif  //#ifdef BRADS_NEW_DWDATABASE
