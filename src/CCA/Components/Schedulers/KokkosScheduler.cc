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

#include <CCA/Components/Schedulers/KokkosScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Components/Schedulers/DetailedTask.h>
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

#include <sci_defs/gpu_defs.h>

#include <atomic>
#include <cstring>
#include <iomanip>
#include <thread>


#if defined(HAVE_KOKKOS)
#else
  namespace Kokkos {
    namespace Profiling {
      void pushRegion(const std::string& kName) {};
      void popRegion () {};
    };
};
#endif

/*______________________________________________________________________
  TO DO:
  - Add special handling for TaskType Hypre.
  - Add wait time and thread stats
  - Toggle setDebug calls with a Dout for GPUDW debugging output?
  - Update bulletproofing in problemSetup
  - Update myRankThread with partition equivalent
______________________________________________________________________*/


using namespace Uintah;

//______________________________________________________________________
//
namespace Uintah {

  extern Dout g_task_dbg;
  extern Dout g_task_run;
  extern Dout g_task_order;
  extern Dout g_exec_out;

}


namespace {

  Dout g_dbg(         "Kokkos_DBG"        , "KokkosScheduler", "general debugging info for the KokkosScheduler"      , false );
  Dout g_queuelength( "Kokkos_QueueLength", "KokkosScheduler", "report the task queue length for the KokkosScheduler", false );

  Uintah::MasterLock g_scheduler_mutex{};           // main scheduler lock for multi-threaded task selection
  Uintah::MasterLock g_mark_task_consumed_mutex{};  // allow only one task at a time to enter the task consumed section
  Uintah::MasterLock g_lb_mutex{};                  // load balancer lock

  bool g_have_hypre_task{false};
  DetailedTask* g_HypreTask;
  CallBackEvent g_hypre_task_event;

} // namespace


#if defined(UINTAH_USING_GPU)
extern Uintah::MasterLock cerrLock;

namespace {
#if defined(HAVE_CUDA_NOT_NEEDED)
  Dout g_gpu_ids( "Kokkos_GPU_IDs", "KokkosScheduler", "detailed information to uniquely identify GPUs on a node", false );
#endif
}
#endif


//______________________________________________________________________
//
namespace Uintah { namespace Impl {

namespace {

thread_local  int  t_tid = 0;   // unique ID assigned in thread_driver()

}

} } // namespace Uintah::Impl

#  define CUDA_RT_SAFE_CALL( call ) {                                          \
    cudaError err = call;                                                      \
    if(err != cudaSuccess) {                                                   \
        fprintf(stderr, "\nCUDA error %i in file '%s', on line %i : %s.\n\n",  \
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

//______________________________________________________________________
//
KokkosScheduler::KokkosScheduler( const ProcessorGroup  * myworld
                                ,       KokkosScheduler * parentScheduler
                                )
  : MPIScheduler(myworld, parentScheduler)
{

  if ( Uintah::Parallel::usingDevice() ) {

    // Disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::s_combine_memory = false;
  }

#if defined(HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA)
  // ARS - true if cuda or kokkos??
  //__________________________________
  //
  if ( Uintah::Parallel::usingDevice() ) {
    // ARS - This call resets each device  - not needed???
    // gpuInitialize();

    // ARS - Here is a basic check to make sure only GPUs on the same
    // NUMA can be seen.
    // KOKKOS equivalent - not needed??

    //get the true numDevices (in case we have the simulation turned on)
    int numDevices = 0;
    CUDA_RT_SAFE_CALL(cudaGetDeviceCount(&numDevices));
    int can_access = 0;
    for (int i = 0; i < numDevices; i++) {
      CUDA_RT_SAFE_CALL( cudaSetDevice(i) );
      for (int j = 0; j < numDevices; j++) {
        if (i != j) {
          cudaDeviceCanAccessPeer(&can_access, i, j);
          if (can_access) {
            printf("GOOD\n GPU device #%d can access GPU device #%d\n", i, j);
            cudaDeviceEnablePeerAccess(j, 0);
          } else {
            printf("ERROR\n GPU device #%d cannot access GPU device #%d\n.  "
                   "Uintah is not yet configured to work with multiple GPUs "
                   "in different NUMA regions.  For now, use the environment "
                   "variable CUDA_VISIBLE_DEVICES and don't list GPU device "
                   "#%d\n." , i, j, j);
            SCI_THROW( InternalError("** GPUs in multiple NUMA regions are currently unsupported.", __FILE__, __LINE__));
          }
        }
      }
    }
  }  // using Device
#endif
}


//______________________________________________________________________
//
KokkosScheduler::~KokkosScheduler()
{
#if defined(USE_KOKKOS_INSTANCE)
#if defined(USE_KOKKOS_MALLOC)
  // The data warehouses have not been cleared so Kokkos pointers are
  // still valid as they are reference counted.
  GPUMemoryPool::freeCudaMemoryFromPool();
#else // if defined(USE_KOKKOS_VIEW)
  GPUMemoryPool::freeViewsFromPool();
#endif
#elif(HAVE_CUDA)
  // The data warehouses have not been cleared.
  GPUMemoryPool::freeCudaMemoryFromPool();
#endif
}


//______________________________________________________________________
//
int
KokkosScheduler::verifyAnyGpuActive()
{
#if defined(HAVE_CUDA_NOT_NEEDED)
  // ARS - true if cuda or kokkos??

  // ARS called from sus as a check (no further execution). Normally
  // this call would exit out but sus should exit out.

  // CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));

  // Attempt to access the zeroth GPU
  cudaError_t errorCode = cudaSetDevice(0);
  if (errorCode == cudaSuccess) {
    return 1;  // let 1 be a good error code
  }
#endif

  return 2;
}


//______________________________________________________________________
//
void
KokkosScheduler::problemSetup( const ProblemSpecP     & prob_spec
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

  if (d_myworld->myRank() == 0) {
    std::cout << "\nWARNING: Multi-threaded Kokkos scheduler is EXPERIMENTAL, not all tasks are thread safe or Kokkos-enabled yet.\n" << std::endl;

#if defined(HAVE_CUDA_NOT_NEEDED)
    // ARS Prefunctory check of properties. Seems to me that this call
    // should have been combined with the check in the constructor.
    // KOKKOS equivalent - not needed??
    if ( !g_gpu_ids && Uintah::Parallel::usingDevice() ) {
      cudaError_t retVal;
      int availableDevices;
      // Note: m_num_devices == availableDevices
      CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&availableDevices));
      std::cout << "   Using " << m_num_devices << "/" << availableDevices
                << " available GPU(s)" << std::endl;

      for (int device_id = 0; device_id < availableDevices; device_id++) {
        cudaDeviceProp device_prop;
        CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceProperties(&device_prop, device_id));
        printf("   GPU Device %d: \"%s\" with compute capability %d.%d\n",
               device_id, device_prop.name, device_prop.major, device_prop.minor);
      }
    }
#endif
  }

#if defined(HAVE_CUDA_NOT_NEEDED)
  // ARS Prefunctory check of properties. Seems to me that this call
  // should have been combined with the check in the constructor.
  // KOKKOS equivalent - not needed??
  if ( g_gpu_ids && Uintah::Parallel::usingDevice() ) {
    cudaError_t retVal;
    int availableDevices;
    // Note: m_num_devices == availableDevices
    CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&availableDevices));
    std::ostringstream message;
    message << "   Rank-" << d_myworld->myRank()
            << " using " << m_num_devices << "/" << availableDevices
            << " available GPU(s)\n";

    for ( int device_id = 0; device_id < availableDevices; device_id++ ) {
      cudaDeviceProp device_prop;
      CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceProperties(&device_prop, device_id));
      message << "   Rank-" << d_myworld->myRank()
              << " using GPU Device " << device_id
              << ": \"" << device_prop.name << "\""
              << " with compute capability " << device_prop.major << "." << device_prop.minor
              << " on PCI " << device_prop.pciDomainID << ":" << device_prop.pciBusID << ":" << device_prop.pciDeviceID << "\n";
    }
    DOUT(true, message.str());
  }
#endif

  SchedulerCommon::problemSetup(prob_spec, materialManager);
}

//______________________________________________________________________
//
SchedulerP
KokkosScheduler::createSubScheduler()
{
  return MPIScheduler::createSubScheduler();
}


//______________________________________________________________________
//
void
KokkosScheduler::execute( int tgnum       /* = 0 */
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

  RuntimeStats::initialize_timestep(m_num_schedulers, m_task_graphs);

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
    proc0cout << "KokkosScheduler skipping execute, no tasks\n";
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

  // Get the number of tasks in each task phase and initiate each
  // task. Intiation will post MPI recvs and wont conflict with hypre later.
  for (int i = 0; i < m_num_tasks; i++) {
    m_phase_tasks[m_detailed_tasks->localTask(i)->getTask()->m_phase]++;
    MPIScheduler::initiateTask(m_detailed_tasks->localTask(i), m_abort, m_abort_point, m_curr_iteration);
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
  // A bit of mess here. Four options:
  // 1. Not compiled with OpenMP - serial dispatch
  // 2. Compiled with OpenMP but no Kokkos OpenMP front end.
  // 3. Compiled with OpenMP and Kokkos OpenMP front end,   no depreciated code.
  // 4. Compiled with OpenMP and Kokkos OpenMP front end, with depreciated code.

  // If compiled with OpenMP posssiby a parallel dispatch so get the
  // associated variables that determines the dispatching.
#if defined(_OPENMP)
  int num_partitions        = Uintah::Parallel::getNumPartitions();
  int threads_per_partition = Uintah::Parallel::getThreadsPerPartition();
#endif

  while ( m_num_tasks_done < m_num_tasks )
  {
    // Check the associated variables for a parallel dispatch request.
#if defined(USE_KOKKOS_PARTITION_MASTER)
    if( num_partitions > 1 || threads_per_partition > 1 )
    {
      Kokkos::Profiling::pushRegion("partition_master");

      // Task runner functor.
      auto task_runner = [&] (int thread_id, int num_threads) {

        // Each partition created executes this block of code
        // A task_runner can run a serial task (threads_per_partition == 1) or
        //   a Kokkos-based data parallel task (threads_per_partition  > 1)
        this->runTasks( thread_id );
      };

      // OpenMP Partition Master is deprecated in Kokkos. The
      // parallelization is over the paritions.
      Kokkos::OpenMP::partition_master( task_runner,
                                        num_partitions,
                                        threads_per_partition );
    }
    else
#elif defined(_OPENMP)
    if( num_partitions > 1 )
    {
#if _OPENMP >= 201511
      if (omp_get_max_active_levels() > 1)
#else
      if (omp_get_nested())
#endif
      {
        Kokkos::Profiling::pushRegion("OpenMP Parallel");

        #pragma omp parallel num_threads(num_partitions)
        {
          omp_set_num_threads(threads_per_partition);

          // omp_get_num_threads() is not used so call runTasks directly
          // task_runner(omp_get_thread_num(), omp_get_num_threads());

          this->runTasks(omp_get_thread_num());
        }
      }
      else
      {
        Kokkos::Profiling::pushRegion("runTasks");

        this->runTasks( 0 );
      }
    }
    else
#endif // _OPENMP
    {
      Kokkos::Profiling::pushRegion("runTasks");

      this->runTasks( 0 );
    }

    Kokkos::Profiling::popRegion();

    if ( g_have_hypre_task ) {
      DOUT( g_dbg, " Exited runTasks to run a " << g_HypreTask->getTask()->getType() << " task" );

      runTask(g_HypreTask, m_curr_iteration, 0, g_hypre_task_event);

#if defined(UINTAH_USING_GPU)
      if(g_hypre_task_event == CallBackEvent::GPU)
        m_detailed_tasks->addDeviceExecutionPending(g_HypreTask);
#endif
      g_have_hypre_task = false;
    }
  } // end while ( m_num_tasks_done < m_num_tasks )

  ASSERT(m_num_tasks_done == m_num_tasks);

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
  if( m_runtimeStats ) {
    MPIScheduler::computeNetRuntimeStats();
  }

  // only do on toplevel scheduler
  if (m_parent_scheduler == nullptr) {
    MPIScheduler::outputTimingStats("KokkosScheduler");
  }

  RuntimeStats::report(d_myworld->getComm());

} // end execute()


//______________________________________________________________________
//
void
KokkosScheduler::markTaskConsumed( int          & numTasksDone
                                 , int          & currphase
                                 , int            numPhases
                                 , DetailedTask * dtask
                                 )
{
  std::lock_guard<Uintah::MasterLock> task_consumed_guard(g_mark_task_consumed_mutex);

  // Update the count of tasks consumed by the scheduler.
  numTasksDone++;

  // Task ordering debug info - please keep this here, APH 05/30/18
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
KokkosScheduler::runTask( DetailedTask  * dtask
                        , int             iteration
                        , int             thread_id /* = 0 */
                        , CallBackEvent   event
                        )
{
  Kokkos::Profiling::pushRegion("thread " + std::to_string(thread_id) + ": " + dtask->getName());

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

#if defined(UINTAH_USING_GPU)
    // DS: 10312019: If GPU task is going to modify any variable, mark
    // that variable as invalid on CPU.
    if (event == CallBackEvent::GPU) {
      dtask->markHostAsInvalid(m_dws);
    }
#endif

    dtask->doit(d_myworld, m_dws, plain_old_dws, event);

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_AFTER_EXEC) {
      printTrackedVars(dtask, SchedulerCommon::PRINT_AFTER_EXEC);
    }
  }

  // For CPU and postGPU task runs, post MPI sends and call task->done;
  if (event == CallBackEvent::CPU || event == CallBackEvent::postGPU) {

#if defined(UINTAH_USING_GPU)
    if (Uintah::Parallel::usingDevice()) {

      //DS: 10312019: If CPU task is going to modify any variable,
      //mark that variable as invalid on GPU.
      if (event == CallBackEvent::CPU) {
        dtask->markHostComputesDataAsValid(m_dws);
        dtask->markDeviceAsInvalidHostAsValid(m_dws);
      }

      //DS: 10312019: If GPU task is going to modify any variable, mark that variable as invalid on CPU.
      if (event == CallBackEvent::postGPU) {
        dtask->markHostAsInvalid(m_dws);
      }

      // TODO: Don't make every task run through this
      // TODO: Verify that it only looks for data that's valid in the
      // GPU, and not assuming it's valid.
      // Load up the prepareDeviceVars by preparing ghost cell regions
      // to copy out.
      dtask->findIntAndExtGpuDependencies(m_dws, m_no_copy_data_vars,
                                          m_reloc_new_pos_label,
                                          m_parent_scheduler ? m_parent_scheduler->m_reloc_new_pos_label : nullptr,
                                          iteration, thread_id);

      // The ghost cell destinations indicate which devices we're using,
      // and which ones we'll need streams for.
      dtask->assignDevicesAndStreamsFromGhostVars();
      dtask->createTaskGpuDWs();

      // place everything in the GPU data warehouses
      dtask->prepareDeviceVars(m_dws);
      dtask->prepareTaskVarsIntoTaskDW(m_dws);
      dtask->prepareGhostCellsIntoTaskDW();
      dtask->syncTaskGpuDWs();

      // Get these ghost cells to contiguous arrays so they can be
      // copied to host.
      dtask->performInternalGhostCellCopies();  //TODO: Fix for multiple GPUs

      // Now that we've done internal ghost cell copies, we can mark
      // the staging vars as being valid.

      // TODO: Sync required?  We shouldn't mark data as valid until
      // it has copied.
      dtask->markDeviceRequiresAndModifiesDataAsValid(m_dws);

      dtask->copyAllGpuToGpuDependences(m_dws);
      // TODO: Mark destination staging vars as valid.

      // copy all dependencies to arrays
      dtask->copyAllExtGpuDependenciesToHost(m_dws);

      // In order to help copy values to another on-node GPU or
      // another MPI rank, ghost cell data was placed in a var in the
      // patch it is *going to*.  It helps reuse gpu dw engine code
      // this way.  But soon, after this task is done, we are likely
      // going to receive a different region of that patch from a
      // neighboring on-node GPU or neighboring MPI rank.  So we need
      // to remove this foreign variable now so it can be used again.
      // clearForeignGpuVars(deviceVars);
    }
#endif

  MPIScheduler::postMPISends(dtask, iteration);

#if defined(UINTAH_USING_GPU)
    if (Uintah::Parallel::usingDevice()) {
      dtask->deleteTaskGpuDataWarehouses();
    }
#endif

    dtask->done(m_dws);

    g_lb_mutex.lock();
    {
      // Do the global and local per task monitoring
      sumTaskMonitoringValues( dtask );

      double total_task_time = dtask->task_exec_time();
      if (g_exec_out) {
        m_task_info[dtask->getTask()->getName()][TaskStatsEnum::ExecTime] += total_task_time;
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
    }
  }

  Kokkos::Profiling::popRegion();
}  // end runTask()


//______________________________________________________________________
//
void
KokkosScheduler::runTasks( int thread_id )
{
  Kokkos::Profiling::pushRegion("thread " + std::to_string(thread_id) + ": runTasks");

  while( m_num_tasks_done < m_num_tasks && !g_have_hypre_task ) {

    DetailedTask* readyTask = nullptr;
    DetailedTask* initTask  = nullptr;

    bool havework = false;

#if defined(UINTAH_USING_GPU)
    bool usingDevice = Uintah::Parallel::usingDevice();
    bool gpuInitReady = false;
    bool gpuValidateRequiresAndModifiesCopies = false;
    bool gpuValidateRequiresDelayedCopies = false;
    bool gpuPerformGhostCopies = false;
    bool gpuValidateGhostCopies = false;
    bool gpuCheckIfExecutable = false;
    bool gpuRunReady = false;
    bool gpuPending = false;
    bool cpuInitReady = false;
    bool cpuValidateRequiresAndModifiesCopies = false;
    bool cpuCheckIfExecutable = false;
    bool cpuRunReady = false;
#endif

    // ----------------------------------------------------------------------------------
    // Part 1:
    //    Check if anything this thread can do concurrently.
    //    If so, then update the various scheduler counters.
    // ----------------------------------------------------------------------------------
    //g_scheduler_mutex.lock();
    while (!havework) {
      /*
      * (1.0)
      *
      * If it is time for a Hypre task, exit partitions.
      *
      */
      if ( g_have_hypre_task ) {
        Kokkos::Profiling::popRegion();
        return;
      }

      /*
       * (1.1)
       *
       * If it is time to setup for a reduction task, then do so.
       *
       */
      if ((m_phase_sync_task[m_curr_phase] != nullptr) &&
          (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
        g_scheduler_mutex.lock();
        if ((m_phase_sync_task[m_curr_phase] != nullptr) &&
            (m_phase_tasks_done[m_curr_phase] == m_phase_tasks[m_curr_phase] - 1)) {
          readyTask = m_phase_sync_task[m_curr_phase];
          havework = true;
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
#if defined(UINTAH_USING_GPU)
          cpuRunReady = true;
#endif
        }
        g_scheduler_mutex.unlock();
        break;
      }
      /*
       * (1.2)
       *
       * Run a CPU task that has its MPI communication complete. These
       * tasks get in the external ready queue automatically when
       * their receive count hits 0 in DependencyBatch::received,
       * which is called when a MPI message is delivered.
       *
       * NOTE: This is also where a GPU-enabled task gets into the GPU
       * initially-ready queue
       *
       */
      else if ((readyTask = m_detailed_tasks->getNextExternalReadyTask())) {
        havework = true;

#if defined(UINTAH_USING_GPU)
        /*
         * (1.2.1)
         *
         * If it's a GPU-enabled task, assign it to a device (patches
         * were assigned devices previously) and initiate its H2D
         * computes and requires data copies. This is where the
         * execution cycle begins for each GPU-enabled Task.
         *
         * gpuInitReady = true
         */
        if (usingDevice == false || readyTask->getPatches() == nullptr) {
          // These tasks won't ever have anything to pull out of the device
          // so go ahead and mark the task "done" and say that it's ready
          // to start running as a CPU task.
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
          cpuRunReady = true;
        } else if (usingDevice && !readyTask->getTask()->usesDevice()) {
          // These tasks can't start unless we copy and/or verify all
          // data into host memory
          cpuInitReady = true;
        } else if (readyTask->getTask()->usesDevice()) {
          // These tasks can't start until we copy and/or verify all
          // data into GPU memory
          gpuInitReady = true;
        } else {
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
          cpuRunReady = true;
        }
#else
        // If NOT compiled with device support, then this is a CPU
        // task and we can mark the task consumed
        markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
#endif
        break;
      }

      /*
       * (1.3)
       *
       * If we have an internally-ready CPU task, initiate its MPI
       * receives, preparing it for CPU external ready queue. The task
       * is moved to the CPU external-ready queue in the call to
       * task->checkExternalDepCount().
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

#if defined(UINTAH_USING_GPU)
      else if (usingDevice) {
        /*
         * (1.4)
         *
         * Check if highest priority GPU task's asynchronous H2D
         * copies are completed. If so, then reclaim the streams and
         * events it used for these operations, and mark as valid the
         * vars for which this task was responsible.  (If the vars are
         * awaiting ghost cells then those vars will be updated with a
         * status to reflect they aren't quite valid yet)
         *
         * gpuVerifyDataTransferCompletion = true
         */
        if (m_detailed_tasks->getDeviceValidateRequiresAndModifiesCopiesTask(readyTask)) {
            if (readyTask->getDelayedCopy()) {
              gpuValidateRequiresDelayedCopies = true;
              gpuValidateRequiresAndModifiesCopies = false;
            }
            else {
              gpuValidateRequiresDelayedCopies = false;
              gpuValidateRequiresAndModifiesCopies = true;
            }
            havework = true;
            break;
        }
        /*
         * (1.4.1)
         *
         * Check if all vars and staging vars needed for ghost cell
         * copies are present and valid.  If so, start the ghost cell
         * gathering.  Otherwise, put it back in this pool and try
         * again later.  gpuPerformGhostCopies = true
         */
        else if (m_detailed_tasks->getDevicePerformGhostCopiesTask(readyTask)) {
            gpuPerformGhostCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.2)
         *
         * Prevously the task was gathering in ghost vars.  See if one
         * of those tasks is done, and if so mark the vars it was
         * processing as valid with ghost cells.
         * gpuValidateGhostCopies = true
         */
        else if (m_detailed_tasks->getDeviceValidateGhostCopiesTask(readyTask)) {
            gpuValidateGhostCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.3)
         *
         * Check if all GPU variables for the task are either valid or
         * valid and awaiting ghost cells.  If any aren't yet at that
         * state (due to another task having not copied it in yet),
         * then repeat this step.  If all variables have been copied
         * in and some need ghost cells, then process that.  If no
         * variables need to have their ghost cells processed, the GPU
         * to GPU ghost cell copies.  Also make GPU data as being
         * valid as it is now copied into the device.
         *
         * gpuCheckIfExecutable = true
         */
        else if (m_detailed_tasks->getDeviceCheckIfExecutableTask(readyTask)) {
            gpuCheckIfExecutable = true;
            havework = true;
            break;
        }
        /*
         * (1.4.4)
         *
         * Check if highest priority GPU task's asynchronous device to
         * device ghost cell copies are finished. If so, then reclaim
         * the streams and events it used for these operations,
         * execute the task and then put it into the GPU
         * completion-pending queue.
         *
         * gpuRunReady = true
         */
        else if (m_detailed_tasks->getDeviceReadyToExecuteTask(readyTask)) {
            gpuRunReady = true;
            havework    = true;
            break;

        }
        /*
         * (1.4.5)
         *
         * Check if a CPU task needs data into host memory from GPU
         * memory If so, copies data D2H.  Also checks if all data has
         * arrived and is ready to process.
         *
         * cpuValidateRequiresAndModifiesCopies = true
         */
        else if (m_detailed_tasks->getHostValidateRequiresAndModifiesCopiesTask(readyTask)) {
            cpuValidateRequiresAndModifiesCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.6)
         *
         * Check if all CPU variables for the task are either valid or
         * valid and awaiting ghost cells.  If so, this task can be
         * executed.  If not, (perhaps due to another task having not
         * completed a D2H yet), then repeat this step.
         *
         * cpuCheckIfExecutable = true
         */
        else if (m_detailed_tasks->getHostCheckIfExecutableTask(readyTask)) {
            cpuCheckIfExecutable = true;
            havework = true;
            break;
        }
        /*
         * (1.4.7)
         *
         * Check if highest priority GPU task's asynchronous D2H
         * copies are completed. If so, execute the task and then put
         * it into the CPU completion-pending queue.
         *
         * cpuRunReady = true
         */
        else if (usingDevice
            && m_detailed_tasks->getHostReadyToExecuteTask(readyTask)) {
            markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
            cpuRunReady = true;
            havework    = true;
            break;
        }
        /*
         * (1.5)
         *
         * Check to see if any GPU tasks have been completed. This
         * means the kernel(s) have executed (which prevents out of
         * order kernels, and also keeps tasks that depend on its data
         * to wait until the async kernel call is done).  This task's
         * MPI sends can then be posted and done() can be called.
         *
         * gpuPending = true
         */
        else if (m_detailed_tasks->getDeviceExecutionPendingTask(readyTask)) {
            havework   = true;
            gpuPending = true;
            markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
            break;

        }
      } // if(usingDevice)
#endif
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
    // g_scheduler_mutex.unlock();

    // ----------------------------------------------------------------------------------
    // Part 2
    //    Concurrent Part:
    //      Each thread does its own thing here... modify this code with caution
    // ----------------------------------------------------------------------------------

    if (initTask != nullptr) {
      // Moving initiateTask to executed Tasks. Observed crashes in
      // some cases due an MPI conflict between Uintah and Hypre.  Rank
      // 0 sends an MPI message with a tag before hypre is called, but
      // recv is not posted by the rank 1 before hypre.  Later rank 1's
      // Hypre task calls MPI_Iprobe with the same tag value expecting
      // a message from hypre but receives the message sent by rank 0's
      // Uintah task rather than getting the message by Hypre
      // task. Moving initiateTask posts recvs all tasks in advance and
      // matches the sequence.  Hopefully, it will not conflict in
      // another way that all recvs are placed in advance, uintah can
      // not post send in advance from a task which is executed after
      // Hypre and then hypre's send matches one of the uintah's
      // advance receive starving hypre's recv :) Tried setting
      // task->usesMPI(true); inside hypre's task dependencies with a
      // hope that it will create a new phase for hypre task an no
      // other task / MPI will be executed in parallel, but got some
      // dependency errors. This approach is complex but more robust.
      // Keeping it aside for now and can revisit later if conflict
      // with hypre's recv starving occurs.

      //MPIScheduler::initiateTask(initTask, m_abort, m_abort_point, m_curr_iteration);

      DOUT(g_task_dbg, myRankThread() << " Task internal ready 2 " << *initTask << " deps needed: " << initTask->getExternalDepCount());

      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask) {

      DOUT(g_task_dbg, myRankThread() << " Task external ready " << *readyTask)

      if (readyTask->getTask()->getType() == Task::Reduction) {
        MPIScheduler::initiateReduction(readyTask);
      }
#if defined(UINTAH_USING_GPU)
      else if (gpuInitReady) {
        // Prepare to run a GPU task.

        // Ghost cells from CPU same device to variable not yet on GPU
        // -> Managed already by getGridVar()

        // Ghost cells from CPU same device to variable already on GPU
        // -> Managed in initiateH2DCopies(), then copied with
        // performInternalGhostCellCopies()

        // Ghost cells from GPU other device to variable not yet on
        // GPU -> new MPI code and getGridVar()

        // Ghost cells from GPU other device to variable already on
        // GPU -> new MPI code, then initiateH2DCopies(), and copied
        // with performInternalGhostCellCopies()

        // Ghost cells from GPU same device to variable not yet on GPU
        // -> managed in initiateH2DCopies(), and copied with
        // performInternalGhostCellCopies()

        // Ghost cells from GPU same device to variable already on GPU
        // -> Managed in initiateH2DCopies()?
        readyTask->assignDevicesAndStreams();
        readyTask->initiateH2DCopies(m_dws);
        readyTask->syncTaskGpuDWs();

        // Determine which queue it should go into.
        // TODO: Skip queues if possible, not all tasks performed
        // copies or ghost cell gathers.
        m_detailed_tasks->addDeviceValidateRequiresAndModifiesCopies(readyTask);

      } else if (gpuValidateRequiresAndModifiesCopies) {
        // Mark all requires and modifies vars this task is
        // responsible for copying in as valid.
        readyTask->markDeviceRequiresAndModifiesDataAsValid(m_dws);

        if (readyTask->delayedDeviceVarsValid(m_dws)) {
          // Initiate delayed copy of variables which needs larger
          // ghost cells. We have to wait untill origianal copy is
          // completed to avoid race conditions
          readyTask->setDelayedCopy(1); // Set delayed copy flag to
                                        // indicate next round of
                                        // getDeviceValidateRequiresAndModifiesCopiesTask
                                        // will be for delayed copies
          readyTask->copyDelayedDeviceVars();
        }
        m_detailed_tasks->addDeviceValidateRequiresAndModifiesCopies(readyTask);
      } else if (gpuValidateRequiresDelayedCopies) {
          readyTask->setDelayedCopy(0);    //reset delayed copy flag
          // Marked delayed vars to be valid and mark ready for the next stage
          readyTask->markDeviceRequiresAndModifiesDataAsValid(m_dws);
          m_detailed_tasks->addDevicePerformGhostCopies(readyTask);
      } else if (gpuPerformGhostCopies) {
        // Make sure all staging vars are valid before copying ghost cells in
        if (readyTask->ghostCellsProcessingReady(m_dws)) {
          readyTask->performInternalGhostCellCopies();
          m_detailed_tasks->addDeviceValidateGhostCopies(readyTask);
        } else {
          // Another task must still be copying them.  Put it back in the pool.
          m_detailed_tasks->addDevicePerformGhostCopies(readyTask);
        }
      } else if (gpuValidateGhostCopies) {
        readyTask->markDeviceGhostsAsValid(m_dws);
        m_detailed_tasks->addDeviceCheckIfExecutable(readyTask);
      } else if (gpuCheckIfExecutable) {
        if (readyTask->allGPUVarsProcessingReady(m_dws)) {
          //It's ready to execute.
          m_detailed_tasks->addDeviceReadyToExecute(readyTask);
        } else {
          // Not all ghost cells are ready. Another task must still be
          // working on it.  Put it back in the pool.
          m_detailed_tasks->addDeviceCheckIfExecutable(readyTask);
        }
      } else if (gpuRunReady) {

        // Get out of the partition master if there is a hypre task
        if ( readyTask->getTask()->getType() == Task::Hypre ) {
          g_hypre_task_event = CallBackEvent::GPU;
          g_HypreTask = readyTask;
          g_have_hypre_task = true;

          Kokkos::Profiling::popRegion();
          return;
        }

        // Run the task on the GPU.
        runTask(readyTask, m_curr_iteration, thread_id, CallBackEvent::GPU);

        // See if we're dealing with 32768 ghost cells per patch.  If
        // so, it's easier to manage them on the host for now than on
        // the GPU.  We can issue these on the same stream as runTask,
        // and it won't process until after the GPU kernel completed.
        // initiateD2HForHugeGhostCells(readyTask);

        m_detailed_tasks->addDeviceExecutionPending(readyTask);

      } else if (gpuPending) {
        // The GPU task has completed. All of the computes data is now
        // valid and should be marked as such.

        // Go through all computes for the task. Mark them as valid.
        readyTask->markDeviceComputesDataAsValid(m_dws);

        // Go through all modifies for the task.  Mark any that had
        // valid ghost cells as being invalid.  (Consider task A
        // computes, task B modifies ghost cell layer 1, task C
        // requires ghost cell layer 1.  Task C should not consider
        // the ghost cells valid, as task B just updated the data, and
        // so task C needs new ghost cells.
        readyTask->markDeviceModifiesGhostAsInvalid(m_dws);

        // The Task GPU Datawarehouses are no longer needed.  Delete
        // them on the host and device.
        readyTask->deleteTaskGpuDataWarehouses();

        readyTask->deleteTemporaryTaskVars();

        // Run post GPU part of task.  It won't actually rerun the task
        // But it will run post computation management logic, which includes
        // marking the task as done.
        runTask(readyTask, m_curr_iteration, thread_id, CallBackEvent::postGPU);

        // Recycle this task's stream
#ifdef USE_KOKKOS_INSTANCE
        // Not needed instances are freed when the detailed task is deleted.
#else
        readyTask->reclaimCudaStreamsIntoPool();
#endif
      } // if(gpuPending)
#endif
      else {
        // Prepare to run a CPU task.
#if defined(UINTAH_USING_GPU)
        if (cpuInitReady) {

          // Some CPU tasks still interact with the GPU.  For example,
          // DataArchiver,::ouputVariables, or RMCRT task which copies
          // over old data warehouse variables to the new data
          // warehouse, or even CPU tasks which locally invoke their
          // own quick self contained kernels for quick and dirty local
          // code which use the GPU in a way that the data warehouse or
          // the scheduler never needs to know about it
          // (e.g. transferFrom()).  So because we aren't sure which
          // CPU tasks could use the GPU, just go ahead and assign each
          // task a GPU stream.
          // readyTask->assignStatusFlagsToPrepareACpuTask();
          readyTask->assignDevicesAndStreams();

          // Run initiateD2H on all tasks in case the data we need is
          // in GPU memory but not in host memory.  The exception
          // being we don't run an output task in a non-output
          // timestep.  (It would be nice if the task graph didn't
          // have this OutputVariables task if it wasn't going to
          // output data, but that would require more task graph
          // recompilations, which can be even costlier overall.  So
          // we do the check here.)

          // ARS NOTE: Outputing and Checkpointing may be done out of
          // snyc now. I.e. turned on just before it happens rather
          // than turned on before the task graph execution.  As such,
          // one should also be checking:

          // m_application->activeReductionVariable( "outputInterval" );
          // m_application->activeReductionVariable( "checkpointInterval" );

          // However, if active the code below would be called regardless
          // if an output or checkpoint time step or not. Not sure that is
          // desired but not sure of the effect of not calling it and doing
          // an out of sync output or checkpoint.

          if ((m_output->isOutputTimeStep() || m_output->isCheckpointTimeStep())
              || ((readyTask->getTask()->getName() != "DataArchiver::outputVariables")
                  && (readyTask->getTask()->getName() != "DataArchiver::outputVariables(checkpoint)"))) {
            readyTask->initiateD2H(d_myworld, m_dws);
          }
          if (readyTask->getVarsBeingCopiedByTask().getMap().empty()) {
            if (readyTask->allHostVarsProcessingReady(m_dws)) {
              m_detailed_tasks->addHostReadyToExecute(readyTask);
              //runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
              //GPUStreamPool::reclaimCudaStreamsIntoPool(readyTask);
            } else {
              m_detailed_tasks->addHostCheckIfExecutable(readyTask);
            }
          } else {
            //for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = readyTask->getVarsBeingCopiedByTask().getMap().begin(); it != readyTask->getVarsBeingCopiedByTask().getMap().end(); ++it) {
            //}
            //Once the D2H transfer is done, we mark those vars as valid.
            m_detailed_tasks->addHostValidateRequiresAndModifiesCopies(readyTask);
          }
        } else if (cpuValidateRequiresAndModifiesCopies) {
          readyTask->markHostRequiresAndModifiesDataAsValid(m_dws);
          if (readyTask->allHostVarsProcessingReady(m_dws)) {
            m_detailed_tasks->addHostReadyToExecute(readyTask);
            //runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
            //GPUStreamPool::reclaimCudaStreamsIntoPool(readyTask);
          } else {
            m_detailed_tasks->addHostCheckIfExecutable(readyTask);
          }
        } else if (cpuCheckIfExecutable) {
          if (readyTask->allHostVarsProcessingReady(m_dws)) {
            m_detailed_tasks->addHostReadyToExecute(readyTask);
            //runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
            //GPUStreamPool::reclaimCudaStreamsIntoPool(readyTask);
          }  else {
            // Some vars aren't valid and ready, We must be waiting on
            // another task to finish copying in some of the variables
            // we need.
            m_detailed_tasks->addHostCheckIfExecutable(readyTask);
          }
        } else if (cpuRunReady) {
#endif
          // Get out of the partition master if there is a hypre task
          if ( readyTask->getTask()->getType() == Task::Hypre ) {
            g_hypre_task_event = CallBackEvent::CPU;
            g_HypreTask = readyTask;
            g_have_hypre_task = true;

            Kokkos::Profiling::popRegion();
            return;
          }

          // Run the task on the CPU.
          runTask(readyTask, m_curr_iteration, thread_id, CallBackEvent::CPU);

#if defined(UINTAH_USING_GPU)
          // See note above near cpuInitReady.  Some CPU tasks may
          // internally interact with GPUs without modifying the
          // structure of the data warehouse.

#ifdef USE_KOKKOS_INSTANCE
          // Not needed instances are freed when the detailed task is deleted.
#else
          readyTask->reclaimCudaStreamsIntoPool();
#endif
        }
#endif
      } // CPU
    } // if(readyTask)
    else {
      if (m_recvs.size() != 0u) {
        MPIScheduler::processMPIRecvs(TEST);
      }
    }
  }  // end while (numTasksDone < ntasks)

  Kokkos::Profiling::popRegion();
}  // end runTasks()

//______________________________________________________________________
//  generate string   <MPI_rank>.<Thread_ID>
std::string
KokkosScheduler::myRankThread()
{
  std::ostringstream out;
  out << Uintah::Parallel::getMPIRank()<< "." << Impl::t_tid;
  return out.str();
}
