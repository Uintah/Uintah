/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <sci_defs/cuda_defs.h>
#include <sci_defs/visit_defs.h>

#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUDataWarehouse.h>
  #include <Core/Grid/Variables/GPUGridVariable.h>
  #include <Core/Grid/Variables/GPUStencil7.h>
  #include <Core/Util/DebugStream.h>
#endif

#include <atomic>
#include <cstring>
#include <iomanip>
#include <thread>


#define USE_PACKING


using namespace Uintah;


extern Dout g_task_dbg;
extern Dout g_task_order;
extern Dout g_exec_out;

extern std::map<std::string, double> g_exec_times;

//______________________________________________________________________
//
namespace {

Dout g_dbg(         "Unified_DBG"        , false);
Dout g_queuelength( "Unified_QueueLength", false);

using Mutex = Uintah::MasterLock;
Mutex g_scheduler_mutex{};           // main scheduler lock for multi-threaded task selection
Mutex g_mark_task_consumed_mutex{};  // allow only one task at a time to enter the task consumed section
Mutex g_lb_mutex{};                  // load balancer lock

} // namespace


#ifdef HAVE_CUDA

  DebugStream gpu_stats(              "GPUStats"             , false );
  DebugStream simulate_multiple_gpus( "GPUSimulateMultiple"  , false );
  DebugStream gpudbg(                 "GPUDataWarehouse"     , false );

  namespace {

  std::mutex g_GridVarSuperPatch_mutex{};   // An ugly hack to get superpatches for host levels to work.

  }

  extern std::mutex cerrLock;

#endif



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

  // set main thread's affinity (core-0) and tid
  set_affinity(g_cpu_affinities[0]);
  t_tid = 0;

  // TaskRunner threads start at g_runners[1]
  for (int i = 1; i < g_num_threads; ++i) {
    g_runners[i] = new UnifiedSchedulerWorker(sched);
  }

  // spawn worker threads
  // TaskRunner threads start at [1]
  for (int i = 1; i < g_num_threads; ++i) {
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
                                  , const Output           * oport
                                  ,       UnifiedScheduler * parentScheduler
                                  )
  : MPIScheduler(myworld, oport, parentScheduler)
{

#ifdef HAVE_CUDA
  //__________________________________
  //    
  if ( Uintah::Parallel::usingDevice() ) {
    gpuInitialize();

    // we need one of these for each GPU, as each device will have it's own CUDA context
    //for (int i = 0; i < m_num_devices; i++) {
    //  GPUMemoryPool::getCudaStreamFromPool(i);
    //}

    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::d_combineMemory = false;

    //get the true numDevices (in case we have the simulation turned on)
    int numDevices;
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
            printf("ERROR\n GPU device #%d cannot access GPU device #%d\n.  Uintah is not yet configured to work with multiple GPUs in different NUMA regions.  For now, use the environment variable CUDA_VISIBLE_DEVICES and don't list GPU device #%d\n." , i, j, j);
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
UnifiedScheduler::~UnifiedScheduler()
{
}


//______________________________________________________________________
//
int
UnifiedScheduler::verifyAnyGpuActive()
{

#ifdef HAVE_CUDA
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
UnifiedScheduler::problemSetup( const ProblemSpecP     & prob_spec
                              ,       SimulationStateP & state
                              )
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

  if ( (m_num_threads < 1) &&  Uintah::Parallel::usingDevice() ) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: no thread number specified for Unified Scheduler"
          << std::endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads to be in the range [2, 64],\n.... please use -nthreads <num>, and -gpu if using GPUs",
          __FILE__, __LINE__);
    }
  } else if (m_num_threads > MAX_THREADS) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: Number of threads too large..." << std::endl;
      throw ProblemSetupException(
          "Too many threads. Reduce MAX_THREADS and recompile.", __FILE__,
          __LINE__);
    }
  }

  if (d_myworld->myrank() == 0) {
    std::string plural = (m_num_threads == 1) ? " thread" : " threads";
    std::cout
        << "   WARNING: Multi-threaded Unified scheduler is EXPERIMENTAL, not all tasks are thread safe yet.\n"
        << "   Creating " << m_num_threads << " additional "
        << plural + " for task execution (total task execution threads = "
        << m_num_threads + 1 << ")." << std::endl;

#ifdef HAVE_CUDA
    if (Uintah::Parallel::usingDevice()) {
      cudaError_t retVal;
      int availableDevices;
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

  SchedulerCommon::problemSetup(prob_spec, state);

#ifdef HAVE_CUDA
  // Now pick out the materials out of the file.  This is done with an assumption that there
  // will only be ICE or MPM problems, and no problem will have both ICE and MPM materials in it.
  // I am unsure if this assumption is correct.
  // TODO: Add in MPM material support, just needs to look for an MPM block instead of an ICE block.
  ProblemSpecP mp = prob_spec->findBlockWithOutAttribute("MaterialProperties");
  if (mp) {
    ProblemSpecP group = mp->findBlock("ICE");
    if (group) {
      for (ProblemSpecP child = group->findBlock("material"); child != nullptr; child = child->findNextBlock("material")) {
        ProblemSpecP EOS_ps = child->findBlock("EOS");
        if (!EOS_ps) {
          throw ProblemSetupException("ERROR ICE: Cannot find EOS tag", __FILE__, __LINE__);
        }

        std::string EOS;
        if (!EOS_ps->getAttribute("type", EOS)) {
          throw ProblemSetupException("ERROR ICE: Cannot find EOS 'type' tag", __FILE__, __LINE__);
        }

        // add this material to the collection of materials
        m_material_names.push_back(EOS);
      }
    }
  }
#endif

  // this spawns threads, sets affinity, etc
  init_threads(this, m_num_threads);

#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_shared_state->getVisIt() && !initialized ) {
    m_shared_state->d_douts.push_back( &g_dbg  );
    m_shared_state->d_douts.push_back( &g_queuelength  );

#ifdef HAVE_CUDA
    m_shared_state->d_debugStreams.push_back( &gpu_stats  );
    m_shared_state->d_debugStreams.push_back( &simulate_multiple_gpus  );
    m_shared_state->d_debugStreams.push_back( &gpudbg  );
#endif
    
    initialized = true;
  }
#endif
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
UnifiedScheduler::runTask( DetailedTask*         dtask
                         , int                   iteration
                         , int                   thread_id /* = 0 */
                         , Task::CallBackEvent   event
                         )
{
  // end of per-thread wait time - how long has a thread waited before executing another task
  if (thread_id > 0) {
    Impl::g_runners[thread_id]->stopWaitTime();
  }

  // Only execute CPU or GPU tasks.  Don't execute postGPU tasks a second time.
  if ( event == Task::CPU || event == Task::GPU) {
    
    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_BEFORE_EXEC) {
      printTrackedVars(dtask, SchedulerCommon::PRINT_BEFORE_EXEC);
    }

    std::vector<DataWarehouseP> plain_old_dws(m_dws.size());
    for (int i = 0; i < static_cast<int>(m_dws.size()); i++) {
      plain_old_dws[i] = m_dws[i].get_rep();
    }

    dtask->doit(d_myworld, m_dws, plain_old_dws, event);

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_AFTER_EXEC) {
      printTrackedVars(dtask, SchedulerCommon::PRINT_AFTER_EXEC);
    }
  }

  // For CPU and postGPU task runs, post MPI sends and call task->done;
  if (event == Task::CPU || event == Task::postGPU) {

#ifdef HAVE_CUDA
    if (Uintah::Parallel::usingDevice()) {

      // TODO: Don't make every task run through this
      // TODO: Verify that it only looks for data that's valid in the GPU, and not assuming it's valid.
      //Load up the prepareDeviceVars by preparing ghost cell regions to copy out.
      findIntAndExtGpuDependencies( dtask, iteration, thread_id);

      // The ghost cell destinations indicate which devices we're using,
      // and which ones we'll need streams for.
      assignDevicesAndStreamsFromGhostVars(dtask);
      createTaskGpuDWs(dtask);

      // place everything in the GPU data warehouses
      prepareDeviceVars(dtask);
      prepareTaskVarsIntoTaskDW(dtask);
      prepareGhostCellsIntoTaskDW(dtask);
      syncTaskGpuDWs(dtask);

      // get these ghost cells to contiguous arrays so they can be copied to host.
      performInternalGhostCellCopies(dtask);  //TODO: Fix for multiple GPUs

      // Now that we've done internal ghost cell copies, we can mark the staging vars as being valid.
      // TODO: Sync required?  We shouldn't mark data as valid until it has copied.
      markDeviceRequiresDataAsValid(dtask);

      copyAllGpuToGpuDependences(dtask);
      // TODO: Mark destination staging vars as valid.

      // copy all dependencies to arrays
      copyAllExtGpuDependenciesToHost(dtask);

      // In order to help copy values to another on-node GPU or another MPI rank, ghost cell data
      // was placed in a var in the patch it is *going to*.  It helps reuse gpu dw engine code this way.
      // But soon, after this task is done, we are likely going to receive a different region of that patch
      // from a neighboring on-node GPU or neighboring MPI rank.  So we need to remove this foreign variable
      // now so it can be used again.
      // clearForeignGpuVars(deviceVars);
    }
#endif

  MPIScheduler::postMPISends(dtask, iteration);

#ifdef HAVE_CUDA
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
        g_exec_times[dtask->getTask()->getName()] += total_task_time;
      }
      // if I do not have a sub scheduler
      if (!dtask->getTask()->getHasSubScheduler()) {
        //add my task time to the total time
        mpi_info_[TotalTask] += total_task_time;
        if (!m_shared_state->isCopyDataTimestep() &&
	    dtask->getTask()->getType() != Task::Output) {
          //add contribution for patchlist
          getLoadBalancer()->addContribution(dtask, total_task_time);
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
      for (size_t i = 0; i < mpi_info_.size(); ++i) {
        MPIScheduler::TimingStat e = (MPIScheduler::TimingStat)i;
        m_parent_scheduler->mpi_info_[e] += mpi_info_[e];
      }
      mpi_info_.reset(0);
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
  if (m_shared_state->isCopyDataTimestep()) {
    MPIScheduler::execute( tgnum, iteration );
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
  for (int i = 0; i < m_num_tasks; i++) {
    m_detailed_tasks->localTask(i)->resetDependencyCounts();
  }

  int my_rank = d_myworld->myrank();

  // This only happens if "-emit_taskgraphs" is passed to sus
  makeTaskGraphDoc(m_detailed_tasks, my_rank);

  mpi_info_.reset( 0 );

  m_num_tasks_done = 0;
  m_abort = false;
  m_abort_point = 987654;

  if( m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr ) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(m_detailed_tasks, getLoadBalancer(), m_reloc_new_pos_label, iteration);
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


  //------------------------------------------------------------------------------------------------
  // activate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!m_shared_state->isCopyDataTimestep()) {
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
  if (!m_shared_state->isCopyDataTimestep()) {
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

    proc0cout << "average queue length:" << allqueuelength / d_myworld->size() << std::endl;
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
  if ( m_shared_state != nullptr ) {

    // Stats specific to this threaded scheduler - TaskRunner threads start at g_runners[1]
    for (int i = 1; i < m_num_threads; ++i) {
      m_shared_state->d_runTimeStats[SimulationState::TaskWaitThreadTime] += Impl::g_runners[i]->getWaitTime();
    }

    MPIScheduler::computeNetRunTimeStats(m_shared_state->d_runTimeStats);
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
  std::lock_guard<Mutex> task_consumed_guard(g_mark_task_consumed_mutex);

  // Update the count of tasks consumed by the scheduler.
  numTasksDone++;

  if (g_task_order && d_myworld->myrank() == d_myworld->size() / 2) {
    DOUT(g_task_dbg, myRankThread() << " Running task static order: " << dtask->getStaticOrder() << ", scheduled order: " << numTasksDone);
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
    DetailedTask* initTask = nullptr;

    bool havework = false;

#ifdef HAVE_CUDA
    bool usingDevice = Uintah::Parallel::usingDevice();
    bool gpuInitReady = false;
    bool gpuValidateRequiresCopies = false;
    bool gpuPerformGhostCopies = false;
    bool gpuValidateGhostCopies = false;
    bool gpuCheckIfExecutable = false;
    bool gpuRunReady = false;
    bool gpuPending = false;
    bool cpuInitReady = false;
    bool cpuValidateRequiresCopies = false;
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
#ifdef HAVE_CUDA
          cpuRunReady = true;
#endif
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
        if (usingDevice == false || readyTask->getPatches() == nullptr) {
          //These tasks won't ever have anything to pull out of the device
          //so go ahead and mark the task "done" and say that it's ready
          //to start running as a CPU task.

          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
          cpuRunReady = true;
        } else if (!readyTask->getTask()->usesDevice() && usingDevice) {
          //These tasks can't start unless we copy and/or verify all data into host memory
          cpuInitReady = true;
        } else if (readyTask->getTask()->usesDevice()) {
          //These tasks can't start until we copy and/or verify all data into GPU memory
          gpuInitReady = true;
        } else {
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
          cpuRunReady = true;
        }
#else
        // if NOT compiled with device support, then this is a CPU task and we can mark the task consumed
        markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
#endif
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

#ifdef HAVE_CUDA
      else if (usingDevice){
        /*
         * (1.4)
         *
         * Check if highest priority GPU task's asynchronous H2D copies are completed. If so,
         * then reclaim the streams and events it used for these operations, and mark as valid
         * the vars for which this task was responsible.  (If the vars are awaiting ghost cells
         * then those vars will be updated with a status to reflect they aren't quite valid yet)
         *
         * gpuVerifyDataTransferCompletion = true
         */
        if (usingDevice
            && m_detailed_tasks->getDeviceValidateRequiresCopiesTask(readyTask)) {
            gpuValidateRequiresCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.1)
         *
         * Check if all vars and staging vars needed for ghost cell copies are present and valid.
         * If so, start the ghost cell gathering.  Otherwise, put it back in this pool and try
         * again later.
         * gpuPerformGhostCopies = true
         */
        else if (usingDevice
            && m_detailed_tasks->getDevicePerformGhostCopiesTask(readyTask)) {
            gpuPerformGhostCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.2)
         *
         * Prevously the task was gathering in ghost vars.  See if one of those tasks is done,
         * and if so mark the vars it was processing as valid with ghost cells.
         * gpuValidateGhostCopies = true
         */
        else if (usingDevice
            && m_detailed_tasks->getDeviceValidateGhostCopiesTask(readyTask)) {
            gpuValidateGhostCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.3)
         *
         * Check if all GPU variables for the task are either valid or valid and awaiting ghost cells.
         * If any aren't yet at that state (due to another task having not copied it in yet), then
         * repeat this step.  If all variables have been copied in and some need ghost cells, then
         * process that.  If no variables need to have their ghost cells processed,
         * the GPU to GPU ghost cell copies.  Also make GPU data as being valid as it is now
         * copied into the device.
         *
         * gpuCheckIfExecutable = true
         */
        else if (usingDevice
            && m_detailed_tasks->getDeviceCheckIfExecutableTask(readyTask)) {
            gpuCheckIfExecutable = true;
            havework = true;
            break;
        }
        /*
         * (1.4.4)
         *
         * Check if highest priority GPU task's asynchronous device to device ghost cell copies are
         * finished. If so, then reclaim the streams and events it used for these operations, execute
         * the task and then put it into the GPU completion-pending queue.
         *
         * gpuRunReady = true
         */
        else if (usingDevice
            && m_detailed_tasks->getDeviceReadyToExecuteTask(readyTask)) {
            gpuRunReady = true;
            havework    = true;
            break;

        }
        /*
         * (1.4.5)
         *
         * Check if a CPU task needs data into host memory from GPU memory
         * If so, copies data D2H.  Also checks if all data has arrived and is ready to process.
         *
         * cpuValidateRequiresCopies = true
         */
        else if (usingDevice
            && m_detailed_tasks->getHostValidateRequiresCopiesTask(readyTask)) {
            cpuValidateRequiresCopies = true;
            havework = true;
            break;
        }
        /*
         * (1.4.6)
         *
         * Check if all CPU variables for the task are either valid or valid and awaiting ghost cells.
         * If so, this task can be executed.
         * If not, (perhaps due to another task having not completed a D2H yet), then
         * repeat this step.
         *
         * cpuCheckIfExecutable = true
         */
        else if (usingDevice
            && m_detailed_tasks->getHostCheckIfExecutableTask(readyTask)) {
            cpuCheckIfExecutable = true;
            havework = true;
            break;
        }
        /*
         * (1.4.7)
         *
         * Check if highest priority GPU task's asynchronous D2H copies are completed. If so,
         * execute the task and then put it into the CPU completion-pending queue.
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
         * Check to see if any GPU tasks have been completed. This means the kernel(s)
         * have executed (which prevents out of order kernels, and also keeps tasks that depend on
         * its data to wait until the async kernel call is done).
         * This task's MPI sends can then be posted and done() can be called.
         *
         * gpuPending = true
         */
        else if (usingDevice
            && m_detailed_tasks->getDeviceExecutionPendingTask(readyTask)) {
            havework   = true;
            gpuPending = true;
            markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases, readyTask);
            break;

        }
      }
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
#ifdef HAVE_CUDA
      else if (gpuInitReady) {
        // prepare to run a GPU task.

        // Ghost cells from CPU same  device to variable not yet on GPU -> Managed already by getGridVar()
        // Ghost cells from CPU same  device to variable already on GPU -> Managed in initiateH2DCopies(), then copied with performInternalGhostCellCopies()
        // Ghost cells from GPU other device to variable not yet on GPU -> new MPI code and getGridVar()
        // Ghost cells from GPU other device to variable already on GPU -> new MPI code, then initiateH2DCopies(), and copied with performInternalGhostCellCopies()
        // Ghost cells from GPU same  device to variable not yet on GPU -> managed in initiateH2DCopies(), and copied with performInternalGhostCellCopies()
        // Ghost cells from GPU same  device to variable already on GPU -> Managed in initiateH2DCopies()?
        assignDevicesAndStreams(readyTask);
        initiateH2DCopies(readyTask);
        syncTaskGpuDWs(readyTask);

        //Determine which queue it should go into.
        //TODO: Skip queues if possible, not all tasks performed copies or ghost cell gathers
        m_detailed_tasks->addDeviceValidateRequiresCopies(readyTask);

      } else if (gpuValidateRequiresCopies) {
        //Mark all requires vars this task is responsible for copying in as valid.
        markDeviceRequiresDataAsValid(readyTask);
        m_detailed_tasks->addDevicePerformGhostCopies(readyTask);
      } else if (gpuPerformGhostCopies) {
        //make sure all staging vars are valid before copying ghost cells in
        if (ghostCellsProcessingReady(readyTask)) {
          performInternalGhostCellCopies(readyTask);
          m_detailed_tasks->addDeviceValidateGhostCopies(readyTask);
        } else {
          //Another task must still be copying them.  Put it back in the pool.
          m_detailed_tasks->addDevicePerformGhostCopies(readyTask);
        }
      } else if (gpuValidateGhostCopies) {
        markDeviceGhostsAsValid(readyTask);
        m_detailed_tasks->addDeviceCheckIfExecutable(readyTask);
      } else if (gpuCheckIfExecutable) {
        if (allGPUVarsProcessingReady(readyTask)) {
          //It's ready to execute.
          m_detailed_tasks->addDeviceReadyToExecute(readyTask);
        } else {
          //Not all ghost cells are ready. Another task must still be working on it.  Put it back in the pool.
          m_detailed_tasks->addDeviceCheckIfExecutable(readyTask);
        }
      } else if (gpuRunReady) {

        // Run the task on the GPU!
        runTask(readyTask, m_curr_iteration, thread_id, Task::GPU);

        // See if we're dealing with 32768 ghost cells per patch.  If so,
        // it's easier to manage them on the host for now than on the GPU.  We can issue
        // these on the same stream as runTask, and it won't process until after the GPU
        // kernel completed.
        // initiateD2HForHugeGhostCells(readyTask);

        m_detailed_tasks->addDeviceExecutionPending(readyTask);

      } else if (gpuPending) {
        // The GPU task has completed. All of the computes data is now valid and should be marked as such.

        // Go through all computes for the task. Mark them as valid.
        markDeviceComputesDataAsValid(readyTask);

        // The Task GPU Datawarehouses are no longer needed.  Delete them on the host and device.
        readyTask->deleteTaskGpuDataWarehouses();

        readyTask->deleteTemporaryTaskVars();

        // Run post GPU part of task.  It won't actually rerun the task
        // But it will run post computation management logic, which includes
        // marking the task as done.
        runTask(readyTask, m_curr_iteration, thread_id, Task::postGPU);

        // recycle this task's stream
        GPUMemoryPool::reclaimCudaStreamsIntoPool(readyTask);
      }
#endif
      else {
        // prepare to run a CPU task.
#ifdef HAVE_CUDA
        if (cpuInitReady) {

          //Some CPU tasks still interact with the GPU.  For example, DataArchiver,::ouputVariables, or
          //RMCRT task which copies over old data warehouse variables to the new data warehouse, or even
          //CPU tasks which locally invoke their own quick self contained kernels for quick and dirty local code
          //which use the GPU in a way that the data warehouse or the scheduler never needs to know about it (e.g. transferFrom()).
          //So because we aren't sure which CPU tasks could use the GPU, just go ahead and assign each task
          //a GPU stream.
          //assignStatusFlagsToPrepareACpuTask(readyTask);
          assignDevicesAndStreams(readyTask);

          // Run initiateD2H on all tasks in case the data we need is in GPU memory but not in host memory.
          // The exception being we don't run an output task in a non-output timestep.  
          // (It would be nice if the task graph didn't have this OutputVariables task if
          // it wasn't going to output data, but that would require more task graph recompilations,
          // which can be even costlier overall.  So we do the check here.)

          if ((m_out_port->isOutputTimestep() || m_out_port->isCheckpointTimestep())
              || ((readyTask->getTask()->getName() != "DataArchiver::outputVariables")
                  && (readyTask->getTask()->getName() != "DataArchiver::outputVariables(checkpoint)"))) {
            initiateD2H(readyTask);
          }
          if (readyTask->getVarsBeingCopiedByTask().getMap().empty()) {
            if (allHostVarsProcessingReady(readyTask)) {
              m_detailed_tasks->addHostReadyToExecute(readyTask);
              //runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
              //GPUMemoryPool::reclaimCudaStreamsIntoPool(readyTask);
            } else {
              m_detailed_tasks->addHostCheckIfExecutable(readyTask);
            }
          } else {
            //for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = readyTask->getVarsBeingCopiedByTask().getMap().begin(); it != readyTask->getVarsBeingCopiedByTask().getMap().end(); ++it) {
            //}
            //Once the D2H transfer is done, we mark those vars as valid.
            m_detailed_tasks->addHostValidateRequiresCopies(readyTask);
          }
        } else if (cpuValidateRequiresCopies) {
          markHostRequiresDataAsValid(readyTask);
          if (allHostVarsProcessingReady(readyTask)) {
            m_detailed_tasks->addHostReadyToExecute(readyTask);
            //runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
            //GPUMemoryPool::reclaimCudaStreamsIntoPool(readyTask);
          } else {
            m_detailed_tasks->addHostCheckIfExecutable(readyTask);
          }
        } else if (cpuCheckIfExecutable) {
          if (allHostVarsProcessingReady(readyTask)) {
            m_detailed_tasks->addHostReadyToExecute(readyTask);
            //runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
            //GPUMemoryPool::reclaimCudaStreamsIntoPool(readyTask);
          }  else {
            // Some vars aren't valid and ready,  We must be waiting on another task to finish
            // copying in some of the variables we need.
            m_detailed_tasks->addHostCheckIfExecutable(readyTask);
          }
        } else if (cpuRunReady) {
#endif
          // run CPU task.
          runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
#ifdef HAVE_CUDA
          //See note above near cpuInitReady.  Some CPU tasks may internally interact
          //with GPUs without modifying the structure of the data warehouse.
          //GPUMemoryPool::reclaimCudaStreamsIntoPool(readyTask);
        }
#endif
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


#ifdef HAVE_CUDA


//______________________________________________________________________
//
void
UnifiedScheduler::prepareGpuDependencies( DetailedTask          * dtask
                                        , DependencyBatch       * batch
                                        , const VarLabel        * pos_var
                                        , OnDemandDataWarehouse * dw
                                        , OnDemandDataWarehouse * old_dw
                                        , const DetailedDep     * dep
                                        , LoadBalancerPort      * lb
                                        , DeviceVarDest           dest
                                        )
{

  // This should handle the following scenarios:
  // GPU -> different GPU same node  (write to GPU array, move to other device memory, copy in via copyGPUGhostCellsBetweenDevices)
  // GPU -> different GPU another node (write to GPU array, move to host memory, copy via MPI)
  // GPU -> CPU another node (write to GPU array, move to host memory, copy via MPI)
  // It should not handle
  // GPU -> CPU same node (handled in initateH2D)
  // GPU -> same GPU same node (handled in initateH2D)

  // This method therefore indicates that staging/contiguous arrays are needed, and what ghost cell copies
  // need to occur in the GPU.

  if (dep->isNonDataDependency()) {
    return;
  }

  const VarLabel* label = dep->m_req->m_var;
  const Patch* fromPatch = dep->m_from_patch;
  const int matlIndx = dep->m_matl;
  const Level* level = fromPatch->getLevel();
  const int levelID = level->getID();

  // TODO: Ask Alan about everything in the dep object.
  // the toTasks (will there be more than one?)
  // the dep->comp (computes?)
  // the dep->req (requires?)

  DetailedTask* toTask = nullptr;
  //Go through all toTasks
  for (std::list<DetailedTask*>::const_iterator iter = dep->m_to_tasks.begin(); iter != dep->m_to_tasks.end(); ++iter) {
    toTask = (*iter);
    
    constHandle<PatchSubset> patches = toTask->getPatches();
    const int numPatches = patches->size();
    //const Patch* toPatch = toTask->getPatches()->get(0);
    //if (toTask->getPatches()->size() > 1) {
    //  printf("ERROR:\nUnifiedScheduler::prepareGpuDependencies() does not yet support a dependency that has multiple destination patches\n");
    //  SCI_THROW( InternalError("UnifiedScheduler::prepareGpuDependencies() does not yet support a dependency that has multiple destination patches", __FILE__, __LINE__));
    //}
    for (int i = 0; i < numPatches; i++) {
      const Patch* toPatch = patches->get(i);
      const int fromresource = dtask->getAssignedResourceIndex();
      const int toresource = toTask->getAssignedResourceIndex();

      const int fromDeviceIndex = GpuUtilities::getGpuIndexForPatch(fromPatch);

      // for now, assume that task will only work on one device
      const int toDeviceIndex = GpuUtilities::getGpuIndexForPatch(toTask->getPatches()->get(0));

      if ((fromresource == toresource) && (fromDeviceIndex == toDeviceIndex)) {
        // don't handle GPU -> same GPU same node here
        continue;
      }

      GPUDataWarehouse* gpudw = nullptr;
      if (fromDeviceIndex != -1) {
        gpudw = dw->getGPUDW(fromDeviceIndex);
        if (!gpudw->isValidOnGPU(label->d_name.c_str(), fromPatch->getID(), matlIndx, levelID)) {
          continue;
        }
      } else {
        SCI_THROW(
            InternalError("Device index not found for "+label->getFullName(matlIndx, fromPatch), __FILE__, __LINE__));
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

          // TODO, This compiles a list of regions we need to copy into contiguous arrays.
          // We don't yet handle a scenario where the ghost cell region is exactly the same
          // size as the variable, meaning we don't need to create an array and copy to it.

          // We're going to copy the ghost vars from the source variable (already in the GPU) to a destination
          // array (not yet in the GPU).  So make sure there is a destination.

          // See if we're already planning on making this exact copy.  If so, don't do it again.
          IntVector host_low, host_high, host_offset, host_size;
          host_low = dep->m_low;
          host_high = dep->m_high;
          host_offset = dep->m_low;
          host_size = dep->m_high - dep->m_low;
          const size_t elementDataSize = OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType());
          const size_t memSize = host_size.x() * host_size.y() * host_size.z() * elementDataSize;
          // If this staging var already exists, then assume the full ghost cell copying information
          // has already been set up previously.  (Duplicate dependencies show up by this point, so
          // just ignore the duplicate).

          //TODO, this section should be treated atomically.  Duplicates do happen, and we don't yet handle
          //if two of the duplicates try to get added to dtask->getDeviceVars().add() simultaneously.

          // NOTE: On the CPU, a ghost cell face may be sent from patch A to patch B, while a ghost cell
          // edge/line may be sent from patch A to patch C, and the line of data for C is wholly within
          // the face data for B.
          // For the sake of preparing for cuda aware MPI, we still want to create two staging vars here,
          // a contiguous face for B, and a contiguous edge/line for C.
          if (!(dtask->getDeviceVars().stagingVarAlreadyExists(dep->m_req->m_var, fromPatch, matlIndx, levelID, host_low, host_size, dep->m_req->mapDataWarehouse()))) {


            // TODO: This host var really should be created last minute only if it's copying data to host.  Not here.
            //TODO: Verify this cleans up.  If so change the comment.
            GridVariableBase* tempGhostVar = dynamic_cast<GridVariableBase*>(label->typeDescription()->createInstance());
            tempGhostVar->allocate(dep->m_low, dep->m_high);

            // Indicate we want a staging array in the device.
            dtask->getDeviceVars().add(fromPatch, matlIndx, levelID, true, host_size, memSize, elementDataSize,
                                         host_offset, dep->m_req, Ghost::None, 0, fromDeviceIndex, tempGhostVar, dest);

            // let this Task GPU DW know about this staging array
            dtask->getTaskVars().addTaskGpuDWStagingVar(fromPatch, matlIndx, levelID, host_offset, host_size, elementDataSize, dep->m_req, fromDeviceIndex);

            // Now make sure the Task DW knows about the non-staging variable where the staging variable's data will come from.
            // Scenarios occur in which the same source region is listed to send to two different patches.
            // This task doesn't need to know about the same source twice.
            if (!(dtask->getTaskVars().varAlreadyExists(dep->m_req->m_var, fromPatch, matlIndx, levelID, dep->m_req->mapDataWarehouse()))) {
              // let this Task GPU DW know about the source location.
              dtask->getTaskVars().addTaskGpuDWVar(fromPatch, matlIndx, levelID, elementDataSize, dep->m_req, fromDeviceIndex);
            } else {
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " prepareGpuDependencies - Already had a task GPUDW Var for label " << dep->m_req->m_var->getName()
                            << " patch " << fromPatch->getID()
                            << " matl " << matlIndx
                            << " level " << levelID
                            << std::endl;
                }
                cerrLock.unlock();
              }
            }

            // Handle a GPU-another GPU same device transfer.  We have already queued up the staging array on
            // source GPU.  Now queue up the staging array on the destination GPU.
            if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
              // Indicate we want a staging array in the device.
              // TODO: We don't need a host array, it's going GPU->GPU.  So get rid of tempGhostVar here.
              dtask->getDeviceVars().add(toPatch, matlIndx, levelID, true, host_size,
                                         tempGhostVar->getDataSize(), elementDataSize, host_offset,
                                         dep->m_req, Ghost::None, 0, toDeviceIndex, tempGhostVar, dest);

              // And the task should know of this staging array.
              dtask->getTaskVars().addTaskGpuDWStagingVar(toPatch, matlIndx, levelID, host_offset, host_size, elementDataSize, dep->m_req, toDeviceIndex);

            }

            if (gpu_stats.active()) {
              cerrLock.lock();
              {
                gpu_stats << myRankThread()
                    << " prepareGpuDependencies - Preparing a GPU contiguous ghost cell array ";
                if (dest == GpuUtilities::anotherMpiRank) {
                  gpu_stats << "to prepare for a later copy from MPI Rank " << fromresource << " to MPI Rank " << toresource;
                } else if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
                  gpu_stats << "to prepare for a later GPU to GPU copy from on-node device # " << fromDeviceIndex << " to on-node device # " << toDeviceIndex;
                } else {
                  gpu_stats << "to UNKNOWN ";
                }
                gpu_stats << " for " << dep->m_req->m_var->getName().c_str()
                    << " from patch " << fromPatch->getID()
                    << " to patch " << toPatch->getID()
                    << " between shared low (" << dep->m_low.x() << ", " << dep->m_low.y() << ", " << dep->m_low.z() << ")"
                    << " and shared high (" << dep->m_high.x() << ", " << dep->m_high.y() << ", " << dep->m_high.z() << ")"
                    << " and host offset (" << host_offset.x() << ", " << host_offset.y() << ", " << host_offset.z() << ")"
                    << std::endl;
              }
              cerrLock.unlock();
            }

            // we always write this to a "foreign" staging variable. We are going to copying it from the foreign = false var to the foreign = true var.
            // Thus the patch source and destination are the same, and it's staying on device.
            IntVector temp(0,0,0);
            dtask->getGhostVars().add(dep->m_req->m_var, fromPatch, fromPatch,
                matlIndx, levelID, false, true, host_offset, host_size, dep->m_low, dep->m_high,
                OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType()),
                dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                temp,
                fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                (Task::WhichDW) dep->m_req->mapDataWarehouse(), GpuUtilities::sameDeviceSameMpiRank);



            if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
              // GPU to GPU copies needs another entry indicating a peer to peer transfer.

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                      << " prepareGpuDependencies - Preparing a GPU to GPU peer copy "
                      << " for " << dep->m_req->m_var->getName().c_str()
                      << " from patch " << fromPatch->getID()
                      << " to patch " << toPatch->getID()
                      << " between shared low (" << dep->m_low.x() << ", " << dep->m_low.y() << ", " << dep->m_low.z() << ")"
                      << " and shared high (" << dep->m_high.x() << ", " << dep->m_high.y() << ", " << dep->m_high.z() << ")"
                      << " and host offset (" << host_offset.x() << ", " << host_offset.y() << ", " << host_offset.z() << ")"
                      << std::endl;
                }
                cerrLock.unlock();
              }

              dtask->getGhostVars().add(dep->m_req->m_var, fromPatch, toPatch,
                 matlIndx, levelID, true, true, host_offset, host_size, dep->m_low, dep->m_high,
                 OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType()),
                 dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                 temp,
                 fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                 (Task::WhichDW) dep->m_req->mapDataWarehouse(), GpuUtilities::anotherDeviceSameMpiRank);

            } else if (dest == GpuUtilities::anotherMpiRank)  {
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                      << " prepareGpuDependencies - Preparing a GPU to host ghost cell copy"
                      << " for " << dep->m_req->m_var->getName().c_str()
                      << " from patch " << fromPatch->getID()
                      << " to patch " << toPatch->getID()
                      << " between shared low (" << dep->m_low.x() << ", " << dep->m_low.y() << ", " << dep->m_low.z() << ")"
                      << " and shared high (" << dep->m_high.x() << ", " << dep->m_high.y() << ", " << dep->m_high.z() << ")"
                      << " and host offset (" << host_offset.x() << ", " << host_offset.y() << ", " << host_offset.z() << ")"
                      << std::endl;
                }
                cerrLock.unlock();
              }
              dtask->getGhostVars().add(dep->m_req->m_var, fromPatch, toPatch,
                 matlIndx, levelID, true, true, host_offset, host_size, dep->m_low, dep->m_high,
                 OnDemandDataWarehouse::getTypeDescriptionSize(dep->m_req->m_var->typeDescription()->getSubType()->getType()),
                 dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                 temp,
                 fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                 (Task::WhichDW) dep->m_req->mapDataWarehouse(), GpuUtilities::anotherMpiRank);

            }
          }
        }
          break;
        default: {
          std::cerr << "UnifiedScheduler::prepareGPUDependencies(), unsupported variable type" << std::endl;
        }
      }
    }
  }
}

//______________________________________________________________________
//
void
UnifiedScheduler::gpuInitialize( bool reset )
{

  cudaError_t retVal;
  if (simulate_multiple_gpus.active()) {
    printf("SimulateMultipleGPUs is on, simulating 3 GPUs\n");
    m_num_devices = 3;
  } else {
    int numDevices = 0;
    CUDA_RT_SAFE_CALL(retVal = cudaGetDeviceCount(&numDevices));
    m_num_devices = numDevices;
  }

  if (simulate_multiple_gpus.active()) {

    // we're simulating many, but we only will use one.
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));
    if (reset) {
      CUDA_RT_SAFE_CALL(retVal = cudaDeviceReset());
    }
  } else {
    for (int i = 0; i < m_num_devices; i++) {
      if (reset) {
        CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(i));
        CUDA_RT_SAFE_CALL(retVal = cudaDeviceReset());
      }
    }
    // set it back to the 0th device
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(0));
    m_current_device = 0;
  }

}


//______________________________________________________________________
//
void UnifiedScheduler::turnIntoASuperPatch(GPUDataWarehouse* const gpudw, 
                                           const Level* const level,
                                           const IntVector& low,
                                           const IntVector& high, 
                                           const VarLabel* const label, 
                                           const Patch * const patch, 
                                           const int matlIndx, 
                                           const int levelID ) {
  //Handle superpatch stuff
  //This was originally designed for the use case of turning an entire level into a variable.  
  //We need to set up the equivalent of a super patch.
  //For example, suppose a simulation has 8 patches and 2 ranks and 1 level, and this rank owns
  //patches 0, 1, 2, and 3.  Further suppose this scheduler thread is checking
  //to see the status of a patch 1 variable which has a ton of ghost cells associated
  //with it, enough to envelop all seven other patches.  Also suppose patch 1 is
  //found on the CPU, ghost cells for patches 4, 5, 6, and 7 have previously been sent to us,
  //patch 1 is needed on the GPU, and this is the first thread to process this situation.
  //This thread's job should be to claim it is responsible for processing the variable for
  //patches 0, 1, 2, and 3.  Four GPU data warehouse entries should be created, one for each
  //patch.  

  //Patches 0, 1, 2, and 3 should be given the same pointer, same low, same high, (TODO: but different offsets).
  //In order to avoid concurrency problems when marking all patches in the superpatch region as 
  //belonging to the superpatch, we need to avoid Dining Philosophers problem.  That is accomplished 
  //by claiming patches in *sorted* order, and no scheduler thread can attempt to claim any later patch
  //if it hasn't yet claimed a former patch.  The first thread to claim all will have claimed the
  //"superpatch" region.
  
  //Superpatches essentially are just windows into a shared variable, it uses shared_ptrs behind the scenes
  //With this later only one alloaction or H2D transfer can be done.  This method's job is just 
  //to concurrently set up all the underlying shared_ptr work.  
 
  //Note: Superpatch approaches won't work if for some reason a prior task copied a patch in a non-superpatch
  //manner, at the current moment no known simulation will ever do this.  It is also why we try to prepare
  //the superpatch a bit upstream before concurrency checks start, and not down in prepareDeviceVars(). Brad P - 8/6/2016
  //Future note:   A lock free reference counter should also be created and set to 4 for the above example.
  // If a patch is "vacated" from the GPU, the reference counter should be reduced.  If it hits 0, it
  //shouldn't be automatically deleted, but only available for removal if the memory space hits capacity.

  bool thisThreadHandlesSuperPatchWork = false;
  char label_cstr[80];
  strcpy (label_cstr, label->getName().c_str());


  //Get all patches in the superpatch. Assuming our superpatch is the entire level.
  //This also sorts the neighbor patches by ID for us.  Note that if the current patch is 
  //smaller than all the neighbors, we have to work that in too.
  
  Patch::selectType neighbors;
  //IntVector low, high;
  //level->computeVariableExtents(type, low, high);  //Get the low and high for the level
  level->selectPatches(low, high, neighbors);

  //mark the lowest patch as being the superpatch
  const Patch* firstPatchInSuperPatch = nullptr;
  if (neighbors.size() == 0) {
    //this must be a one patch simulation, there are no neighbors.
    firstPatchInSuperPatch = patch;
  } else {
    firstPatchInSuperPatch = neighbors[0]->getRealPatch();
    //seeing if this patch is lower in ID number than the neighbor patches.
    if (patch->getID() < firstPatchInSuperPatch->getID()) {
      firstPatchInSuperPatch = patch;
    }
  }

  //The firstPatchInSuperPatch may not have yet been handled by a prior task  (such as it being a patch
  //assigned to a different node).  So make an entry if needed.
  gpudw->putUnallocatedIfNotExists(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID,
                                                         false, make_int3(0,0,0), make_int3(0,0,0));
  thisThreadHandlesSuperPatchWork = gpudw->compareAndSwapFormASuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID);

  //At this point the patch has been marked as a superpatch.

  if (thisThreadHandlesSuperPatchWork) {
    
    gpudw->setSuperPatchLowAndSize(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID, 
                                   make_int3(low.x(), low.y(), low.z()), 
                                   make_int3(high.x() - low.x(), high.y() - low.y(), high.z() - low.z())); 
    
    //This thread turned the lowest ID'd patch in the region into a superpatch.  Go through *neighbor* patches
    //in the superpatch region and flag them as being a superpatch as well (the copySuperPatchInfo call below
    //can also flag it as a superpatch.
    for( int i = 0; i < neighbors.size(); i++) {
      if (neighbors[i]->getRealPatch() != firstPatchInSuperPatch) {  //This if statement is because there is no need to merge itself

        //These neighbor patches may not have yet been handled by a prior task.  So go ahead and make sure they show up in the database
        gpudw->putUnallocatedIfNotExists(label_cstr, neighbors[i]->getRealPatch()->getID(), matlIndx, levelID,
                                         false, make_int3(0,0,0), make_int3(0,0,0));

        //TODO: Ensure these variables weren't yet allocated, in use, being copied in, etc. At the time of
        //writing, this scenario didn't exist.  Some ways to solve this include 1) An "I'm using this" reference counter.
        //2) Moving superpatch creation to the start of a timestep, and not at the start of initiateH2D, or 
        //3) predetermining at the start of a timestep what superpatch regions will be, and then we can just form 
        //them together here

        //Shallow copy this neighbor patch into the superaptch
        gpudw->copySuperPatchInfo(label_cstr, firstPatchInSuperPatch->getID(), neighbors[i]->getRealPatch()->getID(), matlIndx, levelID);

      }
    }
    gpudw->compareAndSwapSetSuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID);

  } else {
     //spin and wait until it's done.
     while (!gpudw->isSuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID));
  }
}

//______________________________________________________________________
//
//initiateH2DCopies is a key method for the GPU Data Warehouse and the Unified Scheduler
//It helps manage which data needs to go H2D, what allocations and ghost cells need to be copied, etc.
//It also manages concurrency so that no two threads could process the same task.
//A general philosophy is that this section should do atomic compareAndSwaps if it find it is the one
//to allocate, copy in, or copy in with ghosts.  After any of those actions are seen to have completed
//then they can get marked as being allocated, copied in, or copied in with ghosts.

void
UnifiedScheduler::initiateH2DCopies( DetailedTask * dtask )
{

  const Task* task = dtask->getTask();
  dtask->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
  // transfer some variables twice).
  // Note: A task can only run on one level at a time.  It could run multiple patches and multiple
  // materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }
  for (const Task::Dependency* dependantVar = task->getComputes(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Computes);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  unsigned int device_id = -1;
  // The task runs on one device.  The first patch we see can be used to tell us
  // which device we should be on.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  varIter = vars.begin();
  if (varIter != vars.end()) {
    device_id = GpuUtilities::getGpuIndexForPatch(varIter->second->getPatchesUnderDomain(dtask->getPatches())->get(0));
    OnDemandDataWarehouse::uintahSetCudaDevice(device_id);
  }

  // Go through each unique dependent var and see if we should allocate space and/or queue it to be copied H2D.
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {

    const Task::Dependency* curDependency = varIter->second;
    const TypeDescription::Type type = curDependency->m_var->typeDescription()->getType();

    //make sure we're dealing with a variable we support
    if (type == TypeDescription::CCVariable
        || type == TypeDescription::NCVariable
        || type == TypeDescription::SFCXVariable
        || type == TypeDescription::SFCYVariable
        || type == TypeDescription::SFCZVariable
        || type == TypeDescription::PerPatch
        || type == TypeDescription::ReductionVariable) {

      constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(dtask->getMaterials());
      const int numPatches = patches->size();

      const int patchID = varIter->first.m_patchID;
      const Patch * patch = nullptr;
      for (int i = 0; i < numPatches; i++) {
        if (patches->get(i)->getID() == patchID) {
          patch = patches->get(i);
        }
      }
      if (!patch) {
        printf("ERROR:\nUnifiedScheduler::initiateD2H() patch not found.\n");
        SCI_THROW( InternalError("UnifiedScheduler::initiateD2H() patch not found.", __FILE__, __LINE__));
      }
      const int matlID = varIter->first.m_matlIndex;
      const Level* level = getLevel(patches.get_rep());
      int levelID = level->getID();
      if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
        levelID = -1;
      }

      const int deviceIndex = GpuUtilities::getGpuIndexForPatch(patch);

      // For this dependency, get its CPU Data Warehouse and GPU Datawarehouse.
      const int dwIndex = curDependency->mapDataWarehouse();
      OnDemandDataWarehouseP dw = m_dws[dwIndex];
      GPUDataWarehouse* gpudw = dw->getGPUDW(deviceIndex);

      // a fix for when INF ghost cells are requested such as in RMCRT e.g. tsk->requires(abskg_dw, d_abskgLabel, gac, SHRT_MAX);
      bool uses_SHRT_MAX = (curDependency->m_num_ghost_cells == SHRT_MAX);

      //Get all size information about this dependency.
      IntVector low, high; // lowOffset, highOffset;
      if (uses_SHRT_MAX) {
        level->computeVariableExtents(type, low, high);
      } else {
        Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
        patch->computeVariableExtents(basis, curDependency->m_var->getBoundaryLayer(), curDependency->m_gtype, curDependency->m_num_ghost_cells, low, high);
      }
      const IntVector host_size = high - low;
      const size_t elementDataSize = OnDemandDataWarehouse::getTypeDescriptionSize(curDependency->m_var->typeDescription()->getSubType()->getType());
      size_t memSize = 0;
      if (type == TypeDescription::PerPatch
          || type == TypeDescription::ReductionVariable) {
        memSize = elementDataSize;
      } else {
        memSize = host_size.x() * host_size.y() * host_size.z() * elementDataSize;
      }

      // Set up/get status flags
      // Start by checking if an entry doesn't exist in the GPU data warehouse.  If so, create one.
      gpudw->putUnallocatedIfNotExists(curDependency->m_var->getName().c_str(), patchID, matlID, levelID, false,
                                       make_int3(low.x(), low.y(), low.z()),
                                       make_int3(host_size.x(), host_size.y(), host_size.z()));

      bool correctSize = false;
      bool allocating = false;
      bool allocated = false;
      bool copyingIn = false;
      bool validOnGPU = false;
      bool gatheringGhostCells = false;
      bool validWithGhostCellsOnGPU = false;
      bool deallocating = false;
      bool formingSuperPatch = false;
      bool superPatch = false;


      gpudw->getStatusFlagsForVariableOnGPU(correctSize, allocating, allocated, copyingIn,
                                      validOnGPU, gatheringGhostCells, validWithGhostCellsOnGPU,
                                      deallocating, formingSuperPatch, superPatch,
                                      curDependency->m_var->getName().c_str(), patchID, matlID, levelID,
                                      make_int3(low.x(), low.y(), low.z()),
                                      make_int3(host_size.x(), host_size.y(), host_size.z()));


      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
              << " InitiateH2D - Handling this task's dependency for "
              << curDependency->m_var->getName() << " for patch: " << patchID
              << " material: " << matlID << " level: " << levelID;
          if (curDependency->m_dep_type == Task::Requires) {
            gpu_stats << " - A REQUIRES dependency";
          } else if (curDependency->m_dep_type == Task::Computes) {
            gpu_stats << " - A COMPUTES dependency";
          }
          gpu_stats << " with a size of (" << host_size.x() << ", " << host_size.y() << ", " << host_size.z() << ")"
                    << " and each element in it has a size of " << elementDataSize << " bytes"
                    << " and the entire variable uses  " << memSize << " bytes." << std::endl;
        }
        cerrLock.unlock();
      }

      if (curDependency->m_dep_type == Task::Requires) {

        // For any variable, only ONE task should manage all ghost cells for it.
        // It is a giant mess to try and have two tasks simultaneously managing ghost cells for a single var.
        // So if ghost cells are required, attempt to claim that we're the ones going to manage ghost cells
        // This changes a var's status to valid awaiting ghost data if this task claims ownership of managing ghost cells
        // Otherwise the var's status is left alone (perhaps the ghost cells were already processed by another task a while ago)
        bool gatherGhostCells = false;
        if (curDependency->m_gtype != Ghost::None && curDependency->m_num_ghost_cells > 0) {
          
          if(uses_SHRT_MAX) { 
            //Turn this into a superpatch if not already done so:
            turnIntoASuperPatch(gpudw, level, low, high, curDependency->m_var, patch, matlID, levelID);
  
            //At the moment superpatches are gathered together through an upcoming getRegionModifiable() call.  So we 
            //still need to mark it as AWAITING_GHOST_CELLS. It should trigger as one of the simpler scenarios
            //below where it knows it can gather the ghost cells host-side before sending it into GPU memory.
          }
          
          //See if we get to be the lucky thread that processes all ghost cells for this simulation variable
          gatherGhostCells = gpudw->compareAndSwapAwaitingGhostDataOnGPU(curDependency->m_var->getName().c_str(), patchID, matlID, levelID);
        }

        if ((allocating || allocated) && correctSize && (copyingIn || validOnGPU)) {
          //This variable exists or soon will exist on the destination.  So the non-ghost cell part of this
          //variable doesn't need any more work.

          //Queue it to be added to this tasks's TaskDW.
          //It's possible this variable data already was queued to be sent in due to this patch being a ghost cell region of another patch
          //So just double check to prevent duplicates.
          if (!dtask->getTaskVars().varAlreadyExists(curDependency->m_var, patch, matlID, levelID, curDependency->mapDataWarehouse())) {
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
          }

          if (gatherGhostCells) {
            // The variable's space exists or will soon exist on the GPU.  Now copy in any ghost cells
            // into the GPU and let the GPU handle the ghost cell copying logic.

            if (gpu_stats.active()) {
              cerrLock.lock();
              {
                gpu_stats << myRankThread()
                    << " InitiateH2D() - The variable "
                    << curDependency->m_var->getName().c_str()
                    << " patch " << patchID << " material " << matlID
                    << " has been copied in or is copying into the GPU.  But ghost cells are not copied in, so starting that process now." << std::endl;
              }
              cerrLock.unlock();
            }

            // Indicate to the scheduler later on that this variable can be marked as valid with ghost cells.
            dtask->getVarsToBeGhostReady().addVarToBeGhostReady(dtask->getName(), patch, matlID, levelID, curDependency, deviceIndex);

            std::vector<OnDemandDataWarehouse::ValidNeighbors> validNeighbors;
            dw->getValidNeighbors(curDependency->m_var, matlID, patch, curDependency->m_gtype, curDependency->m_num_ghost_cells, validNeighbors);
            for (std::vector<OnDemandDataWarehouse::ValidNeighbors>::iterator iter = validNeighbors.begin(); iter != validNeighbors.end(); ++iter) {

              const Patch* sourcePatch = nullptr;
              if (iter->neighborPatch->getID() >= 0) {
                sourcePatch = iter->neighborPatch;
              } else {
                // This occurs on virtual patches.  They can be "wrap around" patches, meaning if you go to one end of a domain
                // you will show up on the other side.  Virtual patches have negative patch IDs, but they know what real patch they
                // are referring to.
                sourcePatch = iter->neighborPatch->getRealPatch();
              }

              IntVector ghost_host_low(0,0,0), ghost_host_high(0,0,0), ghost_host_size(0,0,0);
              IntVector ghost_host_offset(0,0,0), ghost_host_strides(0,0,0);

              IntVector virtualOffset = iter->neighborPatch->getVirtualOffset();

              int sourceDeviceNum = GpuUtilities::getGpuIndexForPatch(sourcePatch);
              int destDeviceNum = deviceIndex;

              //Find out who has our ghost cells.  Listed in priority...
              //It could be in the GPU as a staging/foreign var
              //Or in the GPU as a full variable
              //Or in the CPU as a foreign var
              //Or in the CPU as a regular variable
              bool useGpuStaging = false;
              bool useGpuGhostCells = false;
              bool useCpuForeign = false;
              bool useCpuGhostCells = false;

              //See if it's in the GPU as a staging/foreign var
              useGpuStaging = gpudw->stagingVarExists(curDependency->m_var->getName().c_str(),
                                                    patchID, matlID, levelID,
                                                    make_int3(iter->low.x(), iter->low.y(), iter->low.z()),
                                                    make_int3(iter->high.x() - iter->low.x(), iter->high.y()- iter->low.y(), iter->high.z()- iter->low.z()));

              //See if we have the entire neighbor patch in the GPU (not just a staging)
              useGpuGhostCells = gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(), sourcePatch->getID(), matlID, levelID);

              //See if we have CPU foreign var data or just the plain CPU variable we can use
              //Note: We don't have a full system in place to set valid all CPU variables.  Specifically foreign variables are not set, and
              //so the line below is commented out.  In the meantime assume that if it's not on the GPU, it must be on the CPU.
              //if (gpudw->isValidOnCPU(curDependency->m_var->getName().c_str(), sourcePatch->getID(), matlID, levelID)) {
                if (iter->validNeighbor && iter->validNeighbor->isForeign()) {
                  useCpuForeign = true;
                } else {
                  useCpuGhostCells = true;
                }
              //}


              //get the sizes of the source variable
              if (useGpuStaging) {
                ghost_host_low = iter->low;
                ghost_host_high = iter->high;
                ghost_host_size = ghost_host_high - ghost_host_low;
              }else if (useGpuGhostCells) {
                GPUDataWarehouse::GhostType throwaway1;
                int throwaway2;
                int3 ghost_host_low3, ghost_host_high3, ghost_host_size3;

                gpudw->getSizes(ghost_host_low3, ghost_host_high3, ghost_host_size3, throwaway1, throwaway2,
                               curDependency->m_var->getName().c_str(), patchID, matlID, levelID);
                ghost_host_low = IntVector(ghost_host_low3.x, ghost_host_low3.y, ghost_host_low3.z);
                ghost_host_high = IntVector(ghost_host_high3.x, ghost_host_high3.y, ghost_host_high3.z);
                ghost_host_size = IntVector(ghost_host_size3.x, ghost_host_size3.y, ghost_host_size3.z);

              }else if (useCpuForeign || useCpuGhostCells) {
                iter->validNeighbor->getSizes(ghost_host_low, ghost_host_high, ghost_host_offset, ghost_host_size, ghost_host_strides);
              }
              const size_t ghost_mem_size =  ghost_host_size.x() * ghost_host_size.y() * ghost_host_size.z() * elementDataSize;

              if (useGpuStaging) {
                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << myRankThread()
                        << " InitiateH2D() - Using source staging variable in the GPU "
                        << sourcePatch->getID() << " to "
                        << patchID << " from device "
                        << destDeviceNum << " to device " << destDeviceNum
                        << std::endl;
                  }
                  cerrLock.unlock();
                }

                // Make sure this task GPU DW knows about the staging var
                dtask->getTaskVars().addTaskGpuDWStagingVar(patch, matlID, levelID, iter->low, iter->high - iter->low,
                                                            elementDataSize, curDependency, destDeviceNum);

                // Assume for now that the ghost cell region is also the exact same size as the
                // staging var.  (If in the future ghost cell data is managed a bit better as
                // it currently does on the CPU, then some ghost cell regions will be found
                // *within* an existing staging var.  This is known to happen with Wasatch
                // computations involving periodic boundary scenarios.)
                dtask->getGhostVars().add(curDependency->m_var,
                    patch, patch,   /*We're merging the staging variable on in*/
                    matlID, levelID,
                    true, false,
                    iter->low,              /*Assuming ghost cell region is the variable size */
                    IntVector(iter->high.x() - iter->low.x(), iter->high.y() - iter->low.y(), iter->high.z() - iter->low.z()),
                    iter->low,
                    iter->high,
                    elementDataSize,
                    curDependency->m_var->typeDescription()->getSubType()->getType(),
                    virtualOffset,
                    destDeviceNum, destDeviceNum, -1, -1,   /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                    (Task::WhichDW) curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);

                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << myRankThread()
                        << " InitaiteH2D() - Internal GPU ghost cell copy queued for "
                        << curDependency->m_var->getName().c_str() << " from patch "
                        << patchID << " staging true to patch " << patchID
                        << " staging false using a variable starting at ("
                        << iter->low.x() << ", " << iter->low.y() << ", "
                        << iter->low.z() << ") and size ("
                        << (iter->high.x() - iter->low.x()) << ", " << (iter->high.y() - iter->low.y()) << ", "
                        << (iter->high.z() - iter->low.z()) << ")"
                        << " copying from ("
                        << iter->low.x() << ", " << iter->low.y() << ", "
                        << iter->low.z() << ")" << " to (" << iter->high.x()
                        << ", " << iter->high.y() << ", " << iter->high.z()
                        << ")" << " with virtual patch offset ("
                        << virtualOffset.x() << ", " << virtualOffset.y()
                        << ", " << virtualOffset.z() << ")"
                        << "." << std::endl;
                  }
                  cerrLock.unlock();
                }
              } else if (useGpuGhostCells) {
                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << myRankThread()
                        << " InitiateH2D() - Host memory does not need to supply ghost cells to GPU memory from patch "
                        << sourcePatch->getID() << " to "
                        << patchID
                        << std::endl;
                  }
                  cerrLock.unlock();
                }

                // If this task doesn't own this source patch, then we need to make sure
                // the upcoming task data warehouse at least has knowledge of this GPU variable that
                // already exists in the GPU.  So queue up to load the neighbor patch metadata into the
                // task datawarehouse.
                if (!patches->contains(sourcePatch)) {
                  if (!(dtask->getTaskVars().varAlreadyExists(curDependency->m_var, sourcePatch, matlID, levelID,
                                                                (Task::WhichDW) curDependency->mapDataWarehouse()))) {
                      dtask->getTaskVars().addTaskGpuDWVar(sourcePatch, matlID, levelID,
                                                           elementDataSize, curDependency, sourceDeviceNum);
                  }
                }

                // Store the source and destination patch, and the range of the ghost cells
                // A GPU kernel will use this collection to do all internal GPU ghost cell copies for
                // that one specific GPU.
                dtask->getGhostVars().add(curDependency->m_var,
                    sourcePatch, patch, matlID, levelID,
                    false, false,
                    ghost_host_low, ghost_host_size,
                    iter->low, iter->high,
                    elementDataSize,
                    curDependency->m_var->typeDescription()->getSubType()->getType(),
                    virtualOffset,
                    destDeviceNum, destDeviceNum, -1, -1,   /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                    (Task::WhichDW) curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);
                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << myRankThread()
                        << " InitaiteH2D() - Internal GPU memory ghost cell copy queued for "
                        << curDependency->m_var->getName().c_str() << " from patch "
                        << sourcePatch->getID() << " to patch " << patchID
                        << " using a variable starting at ("
                        << ghost_host_low.x() << ", " << ghost_host_low.y() << ", "
                        << ghost_host_low.z() << ") and size ("
                        << ghost_host_size.x() << ", " << ghost_host_size.y() << ", "
                        << ghost_host_size.z() << ")"
                        << " copying from ("
                        << iter->low.x() << ", " << iter->low.y() << ", "
                        << iter->low.z() << ")" << " to (" << iter->high.x()
                        << ", " << iter->high.y() << ", " << iter->high.z()
                        << ")" << " with virtual patch offset ("
                        << virtualOffset.x() << ", " << virtualOffset.y()
                        << ", " << virtualOffset.z() << ")"
                        << "." << std::endl;
                  }
                  cerrLock.unlock();
                }
              } else if (useCpuForeign) {

                if (gpu_stats.active()) {
                   cerrLock.lock();
                   {
                     gpu_stats << myRankThread()
                         << " InitiateH2D() -  The host memory has foreign ghost cells that we will use for "
                         << curDependency->m_var->getName().c_str() << " for patch "
                         << sourcePatch->getID() << " to "
                         << patchID
                         << " with size (" << ghost_host_size.x()
                         << ", " << ghost_host_size.y()
                         << ", " << ghost_host_size.z()
                         << ") with low (" << ghost_host_low.x()
                         << ", " << ghost_host_low.y()
                         << ", " << ghost_host_low.z() << ")"
                         << ".  The iter low is (" << iter->low.x()
                         << ", " << iter->low.y()
                         << ", " << iter->low.z()
                         << ") and iter high is (" << iter->high.x()
                         << ", " << iter->high.y()
                         << ", " << iter->high.z()
                         << ") and the neighbor variable has a virtual offset (" << virtualOffset.x()
                         << ", " << virtualOffset.y()
                         << ", " << virtualOffset.z() << ")"
                         << std::endl;
                   }
                   cerrLock.unlock();
                 }

                // Prepare to tell the host-side GPU DW to allocate space for this variable.
                // Since we already got the gridVariableBase pointer to that foreign var, go ahead and add it in here.
                // (The OnDemandDataWarehouse is weird, it doesn't let you query foreign vars, it will try to inflate a regular
                // var and deep copy the foreign var on in.  So for now, just pass in the pointer.)
                dtask->getDeviceVars().add(sourcePatch, matlID, levelID, true, ghost_host_size, ghost_mem_size,
                                           elementDataSize, ghost_host_low, curDependency, Ghost::None, 0,
                                           destDeviceNum, iter->validNeighbor, GpuUtilities::sameDeviceSameMpiRank);

                // Let this Task GPU DW know about this staging array.  We may end up not needed it if another thread processes it or it became
                // part of a superpatch.  We'll figure that out later when we go actually add it.
                dtask->getTaskVars().addTaskGpuDWStagingVar(sourcePatch, matlID, levelID, ghost_host_low, ghost_host_size,
                                                            elementDataSize, curDependency, sourceDeviceNum);

                dtask->getGhostVars().add(curDependency->m_var, sourcePatch, patch, matlID, levelID,
                                          true, false, ghost_host_low, ghost_host_size,
                                          iter->low, iter->high, elementDataSize,
                                          curDependency->m_var->typeDescription()->getSubType()->getType(), virtualOffset,
                                          destDeviceNum, destDeviceNum, -1, -1, /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                                          (Task::WhichDW)curDependency->mapDataWarehouse(),
                                          GpuUtilities::sameDeviceSameMpiRank);

              } else if (useCpuGhostCells) {
                //This handles the scenario where the variable is in the GPU, but the ghost cell data is only found in the
                //neighboring normal patch (non-foreign) in host memory.  Ghost cells haven't been gathered in or started
                //to be gathered in.

                // Check if we should copy this patch into the GPU.

                // TODO: Instead of copying the entire patch for a ghost cell, we should just create a foreign var, copy
                // a contiguous array of ghost cell data into that foreign var, then copy in that foreign var.  If it's a foreign var,
                // then the foreign var section above should handle it, not here.
                if (!dtask->getDeviceVars().varAlreadyExists(curDependency->m_var, sourcePatch, matlID, levelID, curDependency->mapDataWarehouse())) {

                  if (gpu_stats.active()) {
                    cerrLock.lock();
                    {
                      gpu_stats << myRankThread()
                          << " InitiateH2D() -  The CPU variable has ghost cells needed, use it.  For "
                          << curDependency->m_var->getName().c_str() << " from patch "
                          << sourcePatch->getID() << " to "
                          << patchID
                          << " with size (" << host_size.x()
                          << ", " << host_size.y()
                          << ", " << host_size.z()
                          << ") with low (" << ghost_host_low.x()
                          << ", " << ghost_host_low.y()
                          << ", " << ghost_host_low.z() << ")"
                          << ".  The iter low is (" << iter->low.x()
                          << ", " << iter->low.y()
                          << ", " << iter->low.z()
                          << ") and iter high is (" << iter->high.x()
                          << ", " << iter->high.y()
                          << ", " << iter->high.z()
                          << ") and the neighbor variable has a virtual offset (" << virtualOffset.x()
                          << ", " << virtualOffset.y()
                          << ", " << virtualOffset.z() << ")"
                          << std::endl;
                    }
                    cerrLock.unlock();
                  }
                  // Prepare to tell the host-side GPU DW to possibly allocate and/or copy this variable.
                  dtask->getDeviceVars().add(sourcePatch, matlID, levelID, false,
                      ghost_host_size, ghost_mem_size,
                      elementDataSize, ghost_host_low,
                      curDependency, Ghost::None, 0, destDeviceNum,
                      nullptr, GpuUtilities::sameDeviceSameMpiRank);

                  // Prepare this task GPU DW for knowing about this variable on the GPU.
                  dtask->getTaskVars().addTaskGpuDWVar(sourcePatch, matlID, levelID, elementDataSize, curDependency, destDeviceNum);

                } else {  // else the variable is already in deviceVars
                  if (gpu_stats.active()) {
                    cerrLock.lock();
                    {
                      gpu_stats << myRankThread()
                            << " InitiateH2D() - The CPU has ghost cells needed but it's already been queued to go into the GPU.  Patch "
                            << sourcePatch->getID() << " to "
                            << patchID << " from device "
                            << sourceDeviceNum << " to device " << destDeviceNum
                            << ".  The ghost variable is at (" << ghost_host_low.x()
                            << ", " << ghost_host_low.y()
                            << ", " << ghost_host_low.z()
                            << ") with size (" << ghost_host_size.x()
                            << ", " << ghost_host_size.y()
                            << ", " << ghost_host_size.z()
                            << ") with low (" << ghost_host_low.x()
                            << ", " << ghost_host_low.y()
                            << ", " << ghost_host_low.z() << ")"
                            << ".  The iter low is (" << iter->low.x()
                            << ", " << iter->low.y()
                            << ", " << iter->low.z()
                            << ") and iter high is *" << iter->high.x()
                            << ", " << iter->high.y()
                            << ", " << iter->high.z()
                            << ") the patch ID is " << patchID
                            << " and the neighbor variable has a virtual offset (" << virtualOffset.x()
                            << ", " << virtualOffset.y()
                            << ", " << virtualOffset.z() << ")"
                            << std::endl;
                    }
                    cerrLock.unlock();
                  }
                }

                // Add in info to perform a GPU ghost cell copy.  (It will ensure duplicates can't be entered.)
                dtask->getGhostVars().add(curDependency->m_var,
                                          sourcePatch, patch, matlID, levelID,
                                          false, false,
                                          ghost_host_low, ghost_host_size,
                                          iter->low, iter->high,
                                          elementDataSize,
                                          curDependency->m_var->typeDescription()->getSubType()->getType(),
                                          virtualOffset,
                                          destDeviceNum, destDeviceNum, -1, -1,   /* we're copying within a device, so destDeviceNum -> destDeviceNum */
                                          (Task::WhichDW) curDependency->mapDataWarehouse(),
                                          GpuUtilities::sameDeviceSameMpiRank);
              } else {
                printf("%s ERROR: Needed ghost cell data not found on the CPU or a GPU.  Looking for ghost cells to be sent to label %s patch %d matl %d.  Couldn't find the source from patch %d.\n",
                    myRankThread().c_str(), curDependency->m_var->getName().c_str(), patchID, matlID, sourcePatch->getID());
                SCI_THROW(InternalError("Needed ghost cell data not found on the CPU or a GPU\n",__FILE__, __LINE__));
              }
            } //end neighbors for loop
          } // end if(gatherGhostCells)
        } else if ((allocated || allocating) && !correctSize) {
          // At the moment this isn't allowed. So it does an exit(-1).  There are two reasons for this.
          // First, the current CPU system always needs to "resize" variables when ghost cells are required.
          // Essentially the variables weren't created with room for ghost cells, and so room  needs to be created.
          // This step can be somewhat costly (I've seen a benchmark where it took 5% of the total computation time).
          // And at the moment this hasn't been coded to resize on the GPU.  It would require an additional step and
          // synchronization to make it work.
          // The second reason is with concurrency.  Suppose a patch that CPU thread A own needs
          // ghost cells from a patch that CPU thread B owns.
          // A can recognize that B's data is valid on the GPU, and so it stores for the future to copy B's
          // data on in.  Meanwhile B notices it needs to resize.  So A could start trying to copy in B's
          // ghost cell data while B is resizing its own data.
          // I believe both issues can be fixed with proper checkpoints.  But in reality
          // we shouldn't be resizing variables on the GPU, so this event should never happen.
          gpudw->remove(curDependency->m_var->getName().c_str(), patchID, matlID, levelID);
          std::cerr << "Resizing of GPU grid vars not implemented at this time.  "
                    << "For the GPU, computes need to be declared with scratch computes to have room for ghost cells."
                    << "Requested var of size (" << host_size.x() << ", " << host_size.y() << ", " << host_size.z() << ") "
                    << "with offset (" << low.x() << ", " << low.y() << ", " << low.z() << ")" << std::endl;
          exit(-1);

        } else if ((!allocated && !allocating)
                   || ((allocated || allocating) && correctSize && !validOnGPU && !copyingIn)) {

          // It's either not on the GPU, or space exists on the GPU for it but it is invalid.
          // Either way, gather all ghost cells host side (if needed), then queue the data to be
          // copied in H2D.  If the data doesn't exist in the GPU, then the upcoming allocateAndPut
          // will allocate space for it.  Otherwise if it does exist on the GPU, the upcoming
          // allocateAndPut will notice that and simply configure it to reuse the pointer.

          if (type == TypeDescription::CCVariable
              || type == TypeDescription::NCVariable
              || type == TypeDescription::SFCXVariable
              || type == TypeDescription::SFCYVariable
              || type == TypeDescription::SFCZVariable) {

            // Queue this CPU var to go into the host-side GPU DW.
            // Also queue that this GPU DW var should also be found in this tasks's Task DW.

            dtask->getDeviceVars().add(patch, matlID, levelID, false, host_size, memSize, elementDataSize,
                                       low, curDependency, curDependency->m_gtype, curDependency->m_num_ghost_cells, deviceIndex,
                                       nullptr, GpuUtilities::sameDeviceSameMpiRank);
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);

            //Mark that when this variable is copied in, it will have its ghost cells ready too.
            if (gatherGhostCells) {
              dtask->getVarsToBeGhostReady().addVarToBeGhostReady(dtask->getName(), patch, matlID, levelID, curDependency, deviceIndex);
            }
          } else if (type == TypeDescription::PerPatch) {
            //PerPatchBase* patchVar = dynamic_cast<PerPatchBase*>(curDependency->m_var->typeDescription()->createInstance());
            //dw->get(*patchVar, curDependency->m_var, matlID, patch);
            dtask->getDeviceVars().add(patch, matlID, levelID, elementDataSize, elementDataSize, curDependency, deviceIndex,
                                       nullptr, GpuUtilities::sameDeviceSameMpiRank);
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
          } else if (type == TypeDescription::ReductionVariable) {
            levelID = -1;
            //ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(curDependency->m_var->typeDescription()->createInstance());
            //dw->get(*reductionVar, curDependency->m_var, patch->getLevel(), matlID);
            dtask->getDeviceVars().add(patch, matlID, levelID, elementDataSize, elementDataSize, curDependency, deviceIndex,
                                       nullptr, GpuUtilities::sameDeviceSameMpiRank);
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
          }
          else {
            std::cerr << "UnifiedScheduler::initiateH2D(), unsupported variable type for computes variable "
                      << curDependency->m_var->getName() << std::endl;
          }
        }
      } else if (curDependency->m_dep_type == Task::Computes) {
        // compute the amount of space the host needs to reserve on the GPU for this variable.

        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << myRankThread()
                << " InitiateH2D() - Preparing to allocate computes space on device"
                << " for " << curDependency->m_var->getName()
                << " patch " << patchID
                << " material " << matlID
                << " level " << levelID
                << " on device "
                << deviceIndex
                << std::endl;
          }
          cerrLock.unlock();
        }

        if (type == TypeDescription::PerPatch) {
          //For PerPatch, it's not a mesh of variables, it's just a single variable, so elementDataSize is the memSize.
          dtask->getDeviceVars().add(patch, matlID, levelID, memSize, elementDataSize, curDependency, deviceIndex, nullptr,
                                     GpuUtilities::sameDeviceSameMpiRank);
          dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
        } else if (type == TypeDescription::ReductionVariable) {
          //For ReductionVariable, it's not a mesh of variables, it's just a single variable, so elementDataSize is the memSize.
          dtask->getDeviceVars().add(patch, matlID, levelID, memSize, elementDataSize, curDependency, deviceIndex, nullptr,
                                     GpuUtilities::sameDeviceSameMpiRank);
          dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);

        } else if (type == TypeDescription::CCVariable
            || type == TypeDescription::NCVariable
            || type == TypeDescription::SFCXVariable
            || type == TypeDescription::SFCYVariable
            || type == TypeDescription::SFCZVariable) {
          
          dtask->getDeviceVars().add(patch, matlID, levelID, false, host_size, memSize, elementDataSize, low, curDependency,
                                     curDependency->m_gtype, curDependency->m_num_ghost_cells, deviceIndex, nullptr,
                                     GpuUtilities::sameDeviceSameMpiRank);
          dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID, elementDataSize, curDependency, deviceIndex);
        } else {
          std::cerr << "UnifiedScheduler::initiateH2D(), unsupported variable type for computes variable "
                    << curDependency->m_var->getName() << std::endl;
        }
      }
    }
  }

  // We've now gathered up all possible things that need to go on the device.  Copy it over.

  // gpu_stats << myRankThread() << " Calling createTaskGpuDWs for " << dtask->getName() << std::endl;
  createTaskGpuDWs(dtask);

  // gpu_stats << myRankThread() << " Calling prepareDeviceVars for " << dtask->getName() << std::endl;
  prepareDeviceVars(dtask);

  //At this point all needed variables will have a pointer.

  // gpu_stats << myRankThread() << " Calling prepareTaskVarsIntoTaskDW for " << dtask->getName() << std::endl;
  prepareTaskVarsIntoTaskDW(dtask);

  // gpu_stats << myRankThread() << " Calling prepareGhostCellsIntoTaskDW for " << dtask->getName() << std::endl;
  prepareGhostCellsIntoTaskDW(dtask);

}


//______________________________________________________________________
//
void
UnifiedScheduler::prepareDeviceVars( DetailedTask * dtask )
{
  bool isStaging = false;

  std::string taskID = dtask->getName();
  //std::set<unsigned int> deviceNums = dtask->getDeviceNums();
  //for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums.begin(); deviceNums_it != deviceNums.end(); ++deviceNums_it) {
  isStaging = false;
  // Because maps are unordered, it is possible a staging var could be inserted before the regular var exists.
  // So just loop twice, once when all staging is false, then loop again when all staging is true
  for (int i = 0; i < 2; i++) {
    //Get all data in the GPU, and store it on the GPU Data Warehouse on the host, as only it
    //is responsible for management of data.  So this processes the previously collected deviceVars.
    std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = dtask->getDeviceVars().getMap();

    for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = varMap.begin();
        it != varMap.end(); ++it) {
      int whichGPU = it->second.m_whichGPU;
      int dwIndex = it->second.m_dep->mapDataWarehouse();

      OnDemandDataWarehouseP dw = m_dws[dwIndex];
      GPUDataWarehouse* gpudw = dw->getGPUDW(whichGPU);
      if (!gpudw) {
        if (gpu_stats.active()) {
          cerrLock.lock();
           {
            gpu_stats << myRankThread()
                      << " prepareDeviceVars() - ERROR - No task data warehouse found for device #" << it->second.m_whichGPU
                      << " and dwindex " << dwIndex
                      << std::endl;
          }
          cerrLock.unlock();
        }
        SCI_THROW(InternalError("No GPU data warehouse found\n",__FILE__, __LINE__));
      }

      if (it->second.m_staging == isStaging) {

        if (dtask->getDeviceVars().getTotalVars(whichGPU, dwIndex) >= 0) {

          void* device_ptr = nullptr;  // device base pointer to raw data

          const IntVector offset = it->second.m_offset;
          const IntVector size = it->second.m_sizeVector;
          const IntVector low = offset;
          const IntVector high = offset + size;
          const TypeDescription* type_description = it->second.m_dep->m_var->typeDescription();
          const TypeDescription::Type type = type_description->getType();
          const TypeDescription::Type subtype = type_description->getSubType()->getType();
          const VarLabel* label = it->second.m_dep->m_var;
          char label_cstr[80];
          strcpy (label_cstr, it->second.m_dep->m_var->getName().c_str());
          const Patch* patch = it->second.m_patchPointer;
          const int patchID = it->first.m_patchID;
          const int matlIndx = it->first.m_matlIndx;
          const int levelID = it->first.m_levelIndx;
          const size_t elementDataSize = it->second.m_sizeOfDataType;
          const bool staging = it->second.m_staging;
          const int numGhostCells = it->second.m_numGhostCells;
          Ghost::GhostType ghosttype = it->second.m_gtype;
          bool uses_SHRT_MAX = (numGhostCells == SHRT_MAX);

          //Allocate the vars if needed.  If they've already been allocated, then
          //this simply sets the var to reuse the existing pointer.
          switch (type) {
            case TypeDescription::PerPatch : {
              GPUPerPatchBase* patchVar = OnDemandDataWarehouse::createGPUPerPatch(subtype);
              gpudw->allocateAndPut(*patchVar, label_cstr, patchID, matlIndx, levelID, elementDataSize);
              device_ptr = patchVar->getVoidPointer();
              delete patchVar;
              break;
            }
            case TypeDescription::ReductionVariable : {

              GPUReductionVariableBase* reductionVar = OnDemandDataWarehouse::createGPUReductionVariable(subtype);
              gpudw->allocateAndPut(*reductionVar, label_cstr, patchID, matlIndx, levelID, elementDataSize);
              device_ptr = reductionVar->getVoidPointer();
              delete reductionVar;
              break;
            }
            case TypeDescription::CCVariable :
            case TypeDescription::NCVariable :
            case TypeDescription::SFCXVariable :
            case TypeDescription::SFCYVariable :
            case TypeDescription::SFCZVariable : {
              GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(subtype);

              if (!uses_SHRT_MAX) {
                gpudw->allocateAndPut(*device_var, label_cstr, patchID, matlIndx, levelID, staging,
                                      make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()),
                                      elementDataSize, (GPUDataWarehouse::GhostType)(it->second.m_gtype),
                                      it->second.m_numGhostCells);
              } else {

                //TODO, give it an offset so it could be requested as a patch or as a level.  Right now they all get the same low/high.
                gpudw->allocateAndPut(*device_var, label_cstr, patchID, matlIndx, levelID, staging,
                                        make_int3(low.x(), low.y(), low.z()), make_int3(high.x(), high.y(), high.z()),
                                        elementDataSize, (GPUDataWarehouse::GhostType)(it->second.m_gtype),
                                        it->second.m_numGhostCells);

              }
              device_ptr = device_var->getVoidPointer();
              delete device_var;
              break;
            }
            default : {
              cerrLock.lock();
              {
                std::cerr << "This variable's type is not supported." << std::endl;
              }
              cerrLock.unlock();
            }
          }

          // If it's a requires, copy the data on over.  If it's a computes, leave it as allocated but unused space.
          if (it->second.m_dep->m_dep_type == Task::Requires) {
            if (!device_ptr) {
              std::cerr << "ERROR: GPU variable's device pointer was nullptr" << std::endl;
              throw ProblemSetupException("ERROR: GPU variable's device pointer was nullptr", __FILE__, __LINE__);
            }

            if (it->second.m_dest == GpuUtilities::sameDeviceSameMpiRank) {

              //See if we get to be the thread that performs the H2D copy.
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " prepareDeviceVars() - Checking if we should copy"
                            << " data for variable " << label_cstr
                            << " patch " << patchID
                            << " material " << matlIndx
                            << " level " << levelID
                            << " staging " << staging;
                  if (staging) {
                    gpu_stats << " offset (" << low.x() << ", " << low.y() << ", " << low.z()
                              << ") and size (" << size.x() << ", " << size.y() << ", " << size.z() << ")";
                  }
                  gpu_stats << " destination enum is " << it->second.m_dest
                            << std::endl;
                }
                cerrLock.unlock();
              }

              //Figure out which thread gets to copy data H2D.  First touch wins.  In case of a superpatch,
              //the patch vars were shallow copied so they all patches in the superpatch refer to the same atomic status.
              bool performCopy = false;
              if (!staging) {
                performCopy = gpudw->compareAndSwapCopyingIntoGPU(label_cstr, patchID, matlIndx, levelID);
              }
              else {
                performCopy = gpudw->compareAndSwapCopyingIntoGPUStaging(label_cstr, patchID, matlIndx, levelID,
                                                                     make_int3(low.x(), low.y(), low.z()),
                                                                     make_int3(size.x(), size.y(), size.z()));
              }

              if (performCopy) {
                //This thread is doing the H2D copy for this simulation variable.

                //Start by getting the host pointer.
                void* host_ptr = nullptr;

                //The variable exists in host memory.  We just have to get one and copy it on in.
                switch (type) {
                  case TypeDescription::CCVariable :
                  case TypeDescription::NCVariable :
                  case TypeDescription::SFCXVariable :
                  case TypeDescription::SFCYVariable :
                  case TypeDescription::SFCZVariable : {

                    //The var on the host could either be a regular var or a foreign var.
                    // If it's a regular var, this will manage ghost cells by creating a host var, rewindowing it, then
                    // copying in the regions needed for the ghost cells.
                    // If this is the case, then ghost cells for this specific instance of this var is completed.
                    // If it's a foreign var, then there is no API at the moment to query it directly (if you try to
                    // getGridVar a foreign var, it doesn't work, it wasn't designed for that).  Fortunately
                    // we would have already seen it in whatever function called this. So use that instead.

                    // Note: Unhandled scenario:  If the adjacent patch is only in the GPU, this code doesn't gather it.
                    if (uses_SHRT_MAX) {
                      g_GridVarSuperPatch_mutex.lock();
                      {

                        //The variable wants the entire domain.  So we do a getRegion call instead.
                        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(type_description->createInstance());

                        //dw->allocateAndPut(*gridVar, label, matlIndx, patch, ghosttype, numGhostCells, true);
                        if (!dw->exists(label, matlIndx, patch->getLevel())) {
                          //This creates and deep copies a region from the OnDemandDatawarehouse.
                          //It does so by deep copying from the other patches and forming one large region.
                          dw->getRegionModifiable(*gridVar, label, matlIndx, patch->getLevel(), low, high, true);
                          //passing in a clone (really it's just a shallow copy) to increase the reference counter by one
                          dw->putLevelDB(gridVar->clone(), label, patch->getLevel(), matlIndx);
                          //dw->getLevel(*constGridVar, label, matlIndx, patch->getLevel());
                        } else {
                          exit(-1);
                        }
                        //get the host pointer as well
                        host_ptr = gridVar->getBasePointer();
                        if (gpu_stats.active()) {
                          cerrLock.lock();
                          {
                            gpu_stats << myRankThread()
                                      << " prepareDeviceVars() - Placed a level variable into the host-side levelDB for"
                                      << " patch " << it->first.m_patchID
                                      << " material " << it->first.m_matlIndx
                                      << " level " << it->first.m_levelIndx
                                      << " total variable size " << it->second.m_varMemSize
                                      << " in host address " << host_ptr
                                      << std::endl;
                          }
                          cerrLock.unlock();
                        }

                        //let go of our reference, allowing a single reference to remain and keep the variable alive in leveDB.
                        //delete gridVar;
                        //TODO: Verify this cleans up.  If so change the comment.
                      }
                      g_GridVarSuperPatch_mutex.unlock();
                    } else {
                      if (it->second.m_var)  {
                        //It's a foreign var.  We can't look it up, but we saw it previously.
                        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(it->second.m_var);
                        host_ptr = gridVar->getBasePointer();
                        //Since we didn't do a getGridVar() call, no reference to clean up
                      } else {
                        //I'm commenting carefully because this section has bit me several times.  If it's not done right, the bugs
                        //are a major headache to track down.  -- Brad P. Nov 30, 2016
                        //We need all the data in the patch.  Perform a getGridVar(), which will return a var with the same window/data as the
                        //OnDemand DW variable, or it will create a new window/data sized to hold the room of the ghost cells and copy it into
                        //the gridVar variable.  Internally it keeps track of refcounts for the window object and the data object.
                        //In any scenario treat the gridVar as a temporary copy of the actual var in the OnDemand DW,
                        //and as such that temporary variable needs to be reclaimed so there are no memory leaks.  The problem is that
                        //we need this temporary variable to live long enough to perform a device-to-host copy.
                        //* In one scenario with no ghost cells, you get back the same window/data just with refcounts incremented by 1.
                        //* In another scenario with ghost cells, the ref counts are at least 2, so deleting the gridVar won't automatically deallocate it
                        //* In another scenario with ghost cells, you get back a gridvar holding different window/data, their refcounts are 1
                        //  and so so deleting the gridVar will invoke deallocation.  That would be bad if an async device-to-host copy is needed.
                        //In all scenarios, the correct approach is just to delay deleting the gridVar object, and letting it persist until the
                        //all variable copies complete, then delete the object, which in turn decrements the refcounter, which then allows it to clean
                        //up later where needed (either immediately if the temp's refcounts hit 0, or later when the it does the scrub checks).

                        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(type_description->createInstance());
                        dw->getGridVar(*gridVar, label, matlIndx, patch, ghosttype, numGhostCells);
                        host_ptr = gridVar->getBasePointer();
                        it->second.m_tempVarToReclaim = gridVar;  //This will be held onto so it persists, and then cleaned up after the device-to-host copy

                      }
                    }
                    break;
                  }
                  case TypeDescription::PerPatch : {
                    PerPatchBase* patchVar = dynamic_cast<PerPatchBase*>(type_description->createInstance());
                    dw->get(*patchVar, label, matlIndx, patch);
                    host_ptr = patchVar->getBasePointer();
                    //let go of our reference
                    delete patchVar;
                    break;
                  }
                  case TypeDescription::ReductionVariable : {
                    ReductionVariableBase* reductionVar = dynamic_cast<ReductionVariableBase*>(type_description->createInstance());
                    dw->get(*reductionVar, label, patch->getLevel(), matlIndx);
                    host_ptr = reductionVar->getBasePointer();
                    //let go of our reference
                    delete reductionVar;
                    break;
                  }
                  default : {
                    cerrLock.lock();
                    {
                      std::cerr << "Variable " << label_cstr
                                << " is of a type that is not supported on GPUs yet."
                                << std::endl;
                    }
                    cerrLock.unlock();
                  }
                }

                if (host_ptr && device_ptr) {
                  if (gpu_stats.active()) {
                    cerrLock.lock();
                    {
                      gpu_stats << myRankThread()
                                << " prepareDeviceVars() - Copying into GPU #" << whichGPU
                                << " data for variable " << it->first.m_label
                                << " patch " << it->first.m_patchID
                                << " material " << it->first.m_matlIndx
                                << " level " << it->first.m_levelIndx
                                << " staging " << it->second.m_staging;

                      if (it->second.m_staging) {
                        gpu_stats << " offset (" << low.x() << ", " << low.y() << ", " << low.z()
                                  << ") and size (" << size.x() << ", " << size.y() << ", " << size.z() << ")";
                      }
                      gpu_stats << " total variable size " << it->second.m_varMemSize
                                << " from host address " << host_ptr
                                << " to device address " << device_ptr << " into REQUIRES GPUDW "
                                << std::endl;
                    }
                    cerrLock.unlock();
                  }

                  //Perform the copy!

                  cudaStream_t* stream = dtask->getCudaStreamForThisTask(whichGPU);
                  OnDemandDataWarehouse::uintahSetCudaDevice(whichGPU);
                  if (it->second.m_varMemSize == 0) {
                    printf("ERROR: For variable %s patch %d material %d level %d staging %s attempting to copy zero bytes to the GPU.\n",
                        label_cstr, patchID, matlIndx, levelID, staging ? "true" : "false" );
                    SCI_THROW(InternalError("Attempting to copy zero bytes to the GPU.  That shouldn't happen.", __FILE__, __LINE__));
                  }

                  //Debug loop in case you need to see the data baing sent.
                  //if (it->second.m_varMemSize == 968) {
                  //  printf("UnifiedScheduler - d_data is %p\n", host_ptr);
                  //  for (int i = 0; i < 968/elementDataSize; i++) {
                  //    printf("UnifiedScheduler - Array at index %d is %1.6lf\n", i, *(static_cast<double*>(host_ptr) + i));
                  //  }
                  //}

                  CUDA_RT_SAFE_CALL(cudaMemcpyAsync(device_ptr, host_ptr, it->second.m_varMemSize, cudaMemcpyHostToDevice, *stream));

                  // Tell this task that we're managing the copies for this variable.

                  dtask->getVarsBeingCopiedByTask().getMap().insert(std::pair<GpuUtilities::LabelPatchMatlLevelDw,
                                                                DeviceGridVariableInfo>(it->first, it->second));

                }
              }
            } else if (it->second.m_dest == GpuUtilities::anotherDeviceSameMpiRank || it->second.m_dest == GpuUtilities::anotherMpiRank) {
              // We're not performing a host to GPU copy.  This is just prepare a staging var.
              // So it is a a gpu normal var to gpu staging var copy.
              // It is to prepare for upcoming GPU to host (MPI) or GPU to GPU copies.
              // Tell this task that we're managing the copies for this variable.
              dtask->getVarsBeingCopiedByTask().getMap().insert(
                      std::pair<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>(it->first, it->second));
            }
          }
        }
      }
    }
    isStaging = !isStaging;
  }
  //} end for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums.begin() - this is commented out for now until multi-device support is added
}


//______________________________________________________________________
//
void
UnifiedScheduler::prepareTaskVarsIntoTaskDW( DetailedTask * dtask )
{
  // Copy all task variables metadata into the Task GPU DW.
  // All necessary metadata information must already exist in the host-side GPU DWs.

  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & taskVarMap = dtask->getTaskVars().getMap();

  // Because maps are unordered, it is possible a staging var could be inserted before the regular var exists.
  // So just loop twice, once when all staging is false, then loop again when all staging is true
  bool isStaging = false;

  for (int i = 0; i < 2; i++) {
    std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::const_iterator it;
    for (it = taskVarMap.begin(); it != taskVarMap.end(); ++it) {
      // If isStaging is false, do the non-staging vars, then if isStaging is true, do the staging vars.
      // isStaging is flipped after the first iteration of the i for loop.
      if (it->second.m_staging == isStaging) {
        switch (it->second.m_dep->m_var->typeDescription()->getType()) {
          case TypeDescription::PerPatch :
          case TypeDescription::ReductionVariable :
          case TypeDescription::CCVariable :
          case TypeDescription::NCVariable :
          case TypeDescription::SFCXVariable :
          case TypeDescription::SFCYVariable :
          case TypeDescription::SFCZVariable : {

            int dwIndex = it->second.m_dep->mapDataWarehouse();
            GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(it->second.m_whichGPU);
            int patchID = it->first.m_patchID;
            int matlIndx = it->first.m_matlIndx;
            int levelIndx = it->first.m_levelIndx;

            int3 offset;
            int3 size;
            if (it->second.m_staging) {
              offset = make_int3(it->second.m_offset.x(), it->second.m_offset.y(), it->second.m_offset.z());
              size = make_int3(it->second.m_sizeVector.x(), it->second.m_sizeVector.y(), it->second.m_sizeVector.z());
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " prepareTaskVarsIntoTaskDW() - data for staging variable "
                            << it->second.m_dep->m_var->getName()
                            << " patch " << patchID
                            << " material " << matlIndx
                            << " level " << levelIndx
                            << " offset (" << offset.x << ", " << offset.y << ", " << offset.z << ") "
                            << " size (" << size.x << ", " << size.y << ", " << size.z << ") "
                            << std::endl;
                }
                cerrLock.unlock();
              }

            }
            else {
              offset = make_int3(0, 0, 0);
              size = make_int3(0, 0, 0);
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " prepareTaskVarsIntoTaskDW() - data for variable "
                            << it->second.m_dep->m_var->getName()
                            << " patch " << patchID
                            << " material " << matlIndx
                            << " level " << levelIndx
                            << std::endl;
                }
                cerrLock.unlock();
              }
            }

            GPUDataWarehouse* taskgpudw = dtask->getTaskGpuDataWarehouse(it->second.m_whichGPU, (Task::WhichDW)dwIndex);
            if (taskgpudw) {
              taskgpudw->copyItemIntoTaskDW(gpudw, it->second.m_dep->m_var->getName().c_str(), patchID, matlIndx, levelIndx,
                                            it->second.m_staging, offset, size);
            }
            else {
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread()
                            << " prepareTaskVarsIntoTaskDW() - ERROR - No task data warehouse found for device #" << it->second.m_whichGPU
                            << " and dwindex " << dwIndex
                            << std::endl;
                }
                cerrLock.unlock();
              }
              printf("ERROR - No task data warehouse found for device %d for task %s\n", it->second.m_whichGPU, dtask->getTask()->getName().c_str());
              SCI_THROW(InternalError("No task data warehouse found\n", __FILE__, __LINE__));
            }
          }
            break;
          default : {
            cerrLock.lock();
            {
              std::cerr << "Variable " << it->second.m_dep->m_var->getName()
                        << " is of a type that is not supported on GPUs yet."
                        << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    }
    isStaging = !isStaging;
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::prepareGhostCellsIntoTaskDW( DetailedTask * dtask )
{

  // Tell the Task DWs about any ghost cells they will need to process.
  // This adds in entries into the task DW's d_varDB which isn't a var, but is instead
  // metadata describing how to copy ghost cells between two vars listed in d_varDB.

  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = dtask->getGhostVars().getMap();
  std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it;
  for (it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {
    // If the neighbor is valid on the GPU, we just send in from and to coordinates
    // and call a kernel to copy those coordinates
    // If it's not valid on the GPU, we copy in the grid var and send in from and to coordinates
    // and call a kernel to copy those coordinates.

    // Peer to peer GPU copies will be handled elsewhere.
    // GPU to another MPI ranks will be handled elsewhere.
    if (it->second.m_dest != GpuUtilities::anotherDeviceSameMpiRank && it->second.m_dest != GpuUtilities::anotherMpiRank) {
      int dwIndex = it->first.m_dataWarehouse;

      // We can copy it manually internally within the device via a kernel.
      // This apparently goes faster overall
      IntVector varOffset = it->second.m_varOffset;
      IntVector varSize = it->second.m_varSize;
      IntVector ghost_low = it->first.m_sharedLowCoordinates;
      IntVector ghost_high = it->first.m_sharedHighCoordinates;
      IntVector virtualOffset = it->second.m_virtualOffset;
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
              << " prepareGhostCellsIntoTaskDW() - Preparing ghost cell upcoming copy for " << it->first.m_label
              << " matl " << it->first.m_matlIndx << " level " << it->first.m_levelIndx
              << " from patch " << it->second.m_sourcePatchPointer->getID() << " staging "  << it->second.m_sourceStaging
              << " to patch " << it->second.m_destPatchPointer->getID() << " staging "  << it->second.m_destStaging
              << " from device #" << it->second.m_sourceDeviceNum
              << " to device #" << it->second.m_destDeviceNum
              << " in the Task GPU DW " << dwIndex << std::endl;
        }
        cerrLock.unlock();
      }

      // Add in an entry into this Task DW's d_varDB which isn't a var, but is instead
      // metadata describing how to copy ghost cells between two vars listed in d_varDB.
      dtask->getTaskGpuDataWarehouse(it->second.m_sourceDeviceNum, (Task::WhichDW)dwIndex)->putGhostCell(
                                     it->first.m_label.c_str(), it->second.m_sourcePatchPointer->getID(), it->second.m_destPatchPointer->getID(),
                                     it->first.m_matlIndx, it->first.m_levelIndx, it->second.m_sourceStaging, it->second.m_destStaging,
                                     make_int3(varOffset.x(), varOffset.y(), varOffset.z()), make_int3(varSize.x(), varSize.y(), varSize.z()),
                                     make_int3(ghost_low.x(), ghost_low.y(), ghost_low.z()), make_int3(ghost_high.x(), ghost_high.y(), ghost_high.z()),
                                     make_int3(virtualOffset.x(), virtualOffset.y(), virtualOffset.z()));
    }
  }
}


//______________________________________________________________________
//
bool
UnifiedScheduler::ghostCellsProcessingReady( DetailedTask * dtask )
{
  const Task* task = dtask->getTask();

  // Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
  // transfer some variables twice).
  // Note: A task can only run on one level at a time.  It could run multiple patches and multiple
  // materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  // Go through each var, see if it's valid or valid with ghosts.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* curDependency = varIter->second;

    constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int patchID = varIter->first.m_patchID;
    const Patch * patch = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
      }
    }
    const Level* level = getLevel(patches.get_rep());
    int levelID = level->getID();
    if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      levelID = -1;
    }
    const int matlID = varIter->first.m_matlIndex;
    const int dwIndex = curDependency->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse* gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
    if (curDependency->m_dep_type == Task::Requires) {
      if (curDependency->m_gtype != Ghost::None && curDependency->m_num_ghost_cells > 0) {
        if (!(gpudw->areAllStagingVarsValid(curDependency->m_var->getName().c_str(),patchID, matlID, levelID))) {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread() << " UnifiedScheduler::ghostCellsProcessingReady() -"
                  // Task: " << dtask->getName()
                  << " Not all staging vars were ready for "
                  << curDependency->m_var->getName() << " patch " << patchID
                  << " material " << matlID << " level " << levelID << std::endl;
            }
            cerrLock.unlock();
          }
          return false;
        }
      }
    }
  }

  //if we got there, then everything must be ready to go.
  return true;
}


//______________________________________________________________________
//
bool
UnifiedScheduler::allHostVarsProcessingReady( DetailedTask * dtask )
{

  const Task* task = dtask->getTask();

  dtask->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
  // transfer some variables twice).
  // Note: A task can only run on one level at a time.  It could run multiple patches and multiple
  // materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    if (patches) {
      constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
      const int numPatches = patches->size();
      const int numMatls = matls->size();
      for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numMatls; j++) {
          labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
          if (vars.find(lpmd) == vars.end()) {
            vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
          }
        }
      }
    } else {
      std::cout << myRankThread() << " In allHostVarsProcessingReady, no patches, task is " << dtask->getName() << std::endl;
    }
  }

  // Go through each var, see if it's valid or valid with ghosts.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* curDependency = varIter->second;

    constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int patchID = varIter->first.m_patchID;
    const Patch * patch = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
      }
    }
    const Level* level = getLevel(patches.get_rep());
    int levelID = level->getID();
    if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      levelID = -1;
    }
    const int matlID = varIter->first.m_matlIndex;
    const int dwIndex = curDependency->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse* gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
    if (curDependency->m_dep_type == Task::Requires) {
      if (gpudw->dwEntryExistsOnCPU(curDependency->m_var->getName().c_str(), patchID, matlID, levelID)) {
        if (!(gpudw->isValidOnCPU(curDependency->m_var->getName().c_str(), patchID, matlID, levelID))) {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats
                  << myRankThread()
                  << " UnifiedScheduler::allHostVarsProcessingReady() - Task: "
                  << dtask->getName()
                  << " CPU Task: "
                  << dtask->getName() << " is not ready because this var isn't valid in host memory.  Var "
                  << curDependency->m_var->getName() << " patch " << patchID << " material " << matlID << " level " << levelID
                  << std::endl;
            }
            cerrLock.unlock();
          }
          return false;
        }
      }
    }
  }

  // if we got there, then everything must be ready to go.
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats
          << myRankThread()
          << " UnifiedScheduler::allHostVarsProcessingReady() - Task: "
          << dtask->getName()
          << " CPU Task: "
          << dtask->getName() << " is ready to execute, all required vars are found in in host memory."
          << std::endl;
    }
    cerrLock.unlock();
  }

  return true;
}

//______________________________________________________________________
//
bool
UnifiedScheduler::allGPUVarsProcessingReady( DetailedTask * dtask )
{

  const Task* task = dtask->getTask();

  dtask->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove duplicates (we don't want to
  // transfer some variables twice).
  // Note: A task can only run on one level at a time.  It could run multiple patches and multiple
  // materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  // Go through each var, see if it's valid or valid with ghosts.
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* curDependency = varIter->second;

    constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = curDependency->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int patchID = varIter->first.m_patchID;
    const Patch * patch = nullptr;

    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
      }
    }

    const Level* level = getLevel(patches.get_rep());
    int levelID = level->getID();
    if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      levelID = -1;
    }

    const int matlID = varIter->first.m_matlIndex;
    const int dwIndex = curDependency->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse* gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
    if (curDependency->m_dep_type == Task::Requires) {
      if (curDependency->m_gtype != Ghost::None && curDependency->m_num_ghost_cells > 0) {
        // it has ghost cells.
        if (!(gpudw->isValidWithGhostsOnGPU(curDependency->m_var->getName().c_str(),patchID, matlID, levelID))) {
          return false;
        } else {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats
                  << myRankThread()
                  << " UnifiedScheduler::allGPUVarsProcessingReady() - Task: "
                  << dtask->getName()
                  << " GPU Task: "
                  << dtask->getName() << " verified that var " << curDependency->m_var->getName()
                  << " on patch " << patchID
                  << " is valid with ghost cells."
                  << std::endl;
            }
            cerrLock.unlock();
          }
        }
      } else {
        // If it's a gridvar, then we just don't have the ghost cells processed yet by another thread
        // If it's another type of variable, something went wrong, it should have been marked as valid previously.
        if (!(gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(),patchID, matlID, levelID))) {
          return false;
        } else {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats
                  << myRankThread()
                  << " UnifiedScheduler::allGPUVarsProcessingReady() - Task: "
                  << dtask->getName()
                  << " GPU Task: "
                  << dtask->getName() << " verified that var " << curDependency->m_var->getName()
                  << " on patch " << patchID
                  << " is valid."
                  << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    }
  }

  // if we got there, then everything must be ready to go.
  return true;
}


//______________________________________________________________________
//
void
UnifiedScheduler::markDeviceRequiresDataAsValid( DetailedTask * dtask )
{

  // This marks any Requires variable as valid that wasn't in the GPU but is now in the GPU.
  // If they were already in the GPU due to being computes from a previous time step, it was already
  // marked as valid.  So there is no need to do anything extra for them.
  // If they weren't in the GPU yet, this task or another task copied it in.
  // If it's another task that copied it in, we let that task manage it.
  // If it was this task, then those variables which this task copied in are found in varsBeingCopiedByTask.
  // By the conclusion of this method, some variables will be valid and awaiting ghost cells, some will
  // just be valid if they had no ghost cells, and some variables will be undetermined if they're being managed
  // by another task.
  // After this method, a kernel is invoked to process ghost cells.


  // Go through device requires vars and mark them as valid on the device.  They are either already
  // valid because they were there previously.  Or they just got copied in and the stream completed.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = dtask->getVarsBeingCopiedByTask().getMap();
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = varMap.begin();
            it != varMap.end(); ++it) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);
    if (it->second.m_dep->m_dep_type == Task::Requires) {
      if (!it->second.m_staging) {
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << myRankThread() << " markDeviceRequiresDataAsValid() -"
                << " Marking GPU memory as valid for " << it->second.m_dep->m_var->getName().c_str() << " patch " << it->first.m_patchID << std::endl;
          }
          cerrLock.unlock();
        }
        gpudw->compareAndSwapSetValidOnGPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
      } else {
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << myRankThread() << " markDeviceRequiresDataAsValid() -"
                << " Marking GPU memory as valid for " << it->second.m_dep->m_var->getName().c_str() << " patch " << it->first.m_patchID
                << " offset(" << it->second.m_offset.x() << ", " << it->second.m_offset.y() << ", " << it->second.m_offset.z()
                << ") size (" << it->second.m_sizeVector.x() << ", " << it->second.m_sizeVector.y() << ", " << it->second.m_sizeVector.z() << ")" << std::endl;
          }
          cerrLock.unlock();
        }
        gpudw->compareAndSwapSetValidOnGPUStaging(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx,
                                    make_int3(it->second.m_offset.x(),it->second.m_offset.y(),it->second.m_offset.z()),
                                    make_int3(it->second.m_sizeVector.x(), it->second.m_sizeVector.y(), it->second.m_sizeVector.z()));
      }

      if (it->second.m_tempVarToReclaim) {
        //Release our reference to the variable data that getGridVar returned
        delete it->second.m_tempVarToReclaim;
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::markDeviceGhostsAsValid( DetailedTask * dtask )
{
  // Go through requires vars and mark them as valid on the device.  They are either already
  // valid because they were there previously.  Or they just got copied in and the stream completed.
  // Now go through the varsToBeGhostReady collection.  Any in there should be marked as valid with ghost cells
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = dtask->getVarsToBeGhostReady().getMap();
  for (auto it = varMap.begin(); it != varMap.end(); ++it) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);


    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << myRankThread() << " markDeviceGhostsAsValid() -"
            << " Marking GPU memory as valid with ghosts for " << it->second.m_dep->m_var->getName().c_str() << " patch " << it->first.m_patchID
            << " offset(" << it->second.m_offset.x() << ", " << it->second.m_offset.y() << ", " << it->second.m_offset.z()
            << ") size (" << it->second.m_sizeVector.x() << ", " << it->second.m_sizeVector.y() << ", " << it->second.m_sizeVector.z() << ")" << std::endl;
      }
      cerrLock.unlock();
    }
    gpudw->setValidWithGhostsOnGPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::markDeviceComputesDataAsValid( DetailedTask * dtask )
{
  // Go through device computes vars and mark them as valid on the device.

  // The only thing we need to process is the requires.
  const Task* task = dtask->getTask();
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->m_next) {
    constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(dtask->getMaterials());
    // this is so we can allocate persistent events and streams to distribute when needed
    // one stream and one event per variable per H2D copy (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse * gpudw = dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      if (gpudw != nullptr) {
        for (int j = 0; j < numMatls; j++) {
          int patchID = patches->get(i)->getID();
          int matlID = matls->get(j);
          const Level* level = getLevel(patches.get_rep());
          int levelID = level->getID();
          if (gpudw->isAllocatedOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID)) {
            gpudw->compareAndSwapSetValidOnGPU(comp->m_var->getName().c_str(), patchID, matlID, levelID);
          }
        }
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::markHostRequiresDataAsValid( DetailedTask * dtask )
{
  // Data has been copied from the device to the host.  The stream has completed.
  // Go through all variables that this CPU task was responsible for copying mark them as valid on the CPU

  // The only thing we need to process is the requires.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> & varMap = dtask->getVarsBeingCopiedByTask().getMap();
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = varMap.begin();
            it != varMap.end(); ++it) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse* gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);
    if (it->second.m_dep->m_dep_type == Task::Requires) {
      if (!it->second.m_staging) {
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << myRankThread() << " markHostRequiresDataAsValid() -"
                << " Marking host memory as valid for " << it->second.m_dep->m_var->getName().c_str() << " patch " << it->first.m_patchID << std::endl;
          }
          cerrLock.unlock();
        }
        gpudw->compareAndSwapSetValidOnCPU(it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID, it->first.m_matlIndx, it->first.m_levelIndx);
      }
      if (it->second.m_var) {
        //Release our reference to the variable data that getGridVar returned
        delete it->second.m_var;
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::initiateD2HForHugeGhostCells( DetailedTask * dtask )
{
  // RMCRT problems use 32768 ghost cells as a way to force an "all to all" transmission of ghost cells
  // It is much easier to manage these ghost cells in host memory instead of GPU memory.  So for such
  // variables, after they are done computing, we will copy them D2H.  For RMCRT, this overhead
  // only adds about 1% or less to the overall computation time.  '
  // This only works with COMPUTES, it is not configured to work with requires.

  const Task* task = dtask->getTask();

  // determine which computes variables to copy back to the host
  for (const Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->m_next) {
    // Only process large number of ghost cells.
    if (comp->m_num_ghost_cells == SHRT_MAX) {
      constHandle<PatchSubset> patches = comp->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(dtask->getMaterials());

      int dwIndex = comp->mapDataWarehouse();
      OnDemandDataWarehouseP dw = m_dws[dwIndex];

      void* host_ptr   = nullptr;    // host base pointer to raw data
      void* device_ptr = nullptr;    // device base pointer to raw data
      size_t host_bytes = 0;         // raw byte count to copy to the device

      IntVector host_low, host_high, host_offset, host_size, host_strides;

      int numPatches = patches->size();
      int numMatls = matls->size();
      //__________________________________
      //
      for (int i = 0; i < numPatches; ++i) {
        for (int j = 0; j < numMatls; ++j) {
          const int patchID = patches->get(i)->getID();
          const int matlID  = matls->get(j);
          const Level* level = getLevel(patches.get_rep());
          const int levelID = level->getID();
          const std::string compVarName = comp->m_var->getName();

          const Patch * patch = nullptr;
          for (int i = 0; i < numPatches; i++) {
            if (patches->get(i)->getID() == patchID) {
             patch = patches->get(i);
            }
          }
          if (!patch) {
           printf("ERROR:\nUnifiedScheduler::initiateD2HForHugeGhostCells() patch not found.\n");
           SCI_THROW( InternalError("UnifiedScheduler::initiateD2HForHugeGhostCells() patch not found.", __FILE__, __LINE__));
          }

          const unsigned int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);
          GPUDataWarehouse * gpudw = dw->getGPUDW(deviceNum);
          OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum);
          cudaStream_t* stream = dtask->getCudaStreamForThisTask(deviceNum);

          if (gpudw != nullptr) {

            // It's not valid on the CPU but it is on the GPU.  Copy it on over.
            if (!gpudw->isValidOnCPU( compVarName.c_str(), patchID, matlID, levelID)) {
              const TypeDescription::Type type = comp->m_var->typeDescription()->getType();
              const TypeDescription::Type datatype = comp->m_var->typeDescription()->getSubType()->getType();
              switch (type) {
                case TypeDescription::CCVariable:
                case TypeDescription::NCVariable:
                case TypeDescription::SFCXVariable:
                case TypeDescription::SFCYVariable:
                case TypeDescription::SFCZVariable: {

                  if (gpu_stats.active()) {
                    cerrLock.lock();
                    {
                      gpu_stats << myRankThread() << " initiateD2HForHugeGhostCells() -"
                          // Task: " << dtask->getName()
                          << " Checking if we should copy of \""
                          << compVarName << "\" Patch " << patchID
                          << " Material " << matlID << std::endl;
                    }
                    cerrLock.unlock();
                  }
                  bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(compVarName.c_str(), patchID, matlID, levelID);
                  if (performCopy) {
                    // size the host var to be able to fit all r::oom needed.
                    IntVector host_low, host_high, host_lowOffset, host_highOffset, host_offset, host_size, host_strides;
                    level->computeVariableExtents(type, host_low, host_high);
                    int dwIndex = comp->mapDataWarehouse();
                    OnDemandDataWarehouseP dw = m_dws[dwIndex];

                    // It's possible the computes data may contain ghost cells.  But a task needing to get the data
                    // out of the GPU may not know this.  It may just want the var data.
                    // This creates a dilemma, as the GPU var is sized differently than the CPU var.
                    // So ask the GPU what size it has for the var.  Size the CPU var to match so it can copy all GPU data in.
                    // When the GPU->CPU copy is done, then we need to resize the CPU var if needed to match
                    // what the CPU is expecting it to be.
                    // GPUGridVariableBase* gpuGridVar;

                    int3 low;
                    int3 high;
                    int3 size;
                    GPUDataWarehouse::GhostType tempgtype;
                    Ghost::GhostType gtype;
                    int numGhostCells;
                    gpudw->getSizes(low, high, size, tempgtype, numGhostCells, compVarName.c_str(), patchID, matlID, levelID);

                    gtype = (Ghost::GhostType) tempgtype;


                    if (gpu_stats.active()) {
                      cerrLock.lock();
                      {
                        gpu_stats << myRankThread() << " initiateD2HForHugeGhostCells() -"
                            // Task: " << dtask->getName()
                            << " Yes, we are copying \""
                            << compVarName << "\" patch" << patchID
                            << " material " << matlID
                            << " number of ghost cells " << numGhostCells << " from device to host" << std::endl;
                      }
                      cerrLock.unlock();
                    }

                    GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(comp->m_var->typeDescription()->createInstance());

                    bool finalized = dw->isFinalized();
                    if (finalized) {
                      dw->unfinalize();
                    }

                    dw->allocateAndPut(*gridVar, comp->m_var, matlID, patch, gtype, numGhostCells);
                    if (finalized) {
                      dw->refinalize();
                    }

                    if (gpu_stats.active()) {
                      cerrLock.lock();
                      {
                        gpu_stats << myRankThread() << " initiateD2HForHugeGhostCells() -"
                            // Task: " << dtask->getName()
                            << " allocateAndPut for "
                            << compVarName << " patch" << patchID
                            << " material " << matlID
                            << " number of ghost cells " << numGhostCells << " from device to host" << std::endl;
                      }
                      cerrLock.unlock();
                    }

                    gridVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);
                    host_ptr = gridVar->getBasePointer();
                    host_bytes = gridVar->getDataSize();

                    int3 device_offset;
                    int3 device_size;
                    GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(datatype);
                    gpudw->get(*device_var, compVarName.c_str(), patchID, matlID, levelID);
                    device_var->getArray3(device_offset, device_size, device_ptr);
                    delete device_var;

                    // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
                    if (   device_offset.x == host_low.x()
                        && device_offset.y == host_low.y()
                        && device_offset.z == host_low.z()
                        && device_size.x   == host_size.x()
                        && device_size.y   == host_size.y()
                        && device_size.z   == host_size.z()) {

                      if (gpu_stats.active()) {
                        cerrLock.lock();
                        {
                          gpu_stats << myRankThread() << " initiateD2HForHugeGhostCells - Copy of \""
                              << compVarName << "\""
                              << " patch " << patchID
                              << " material " << matlID
                              << " level " << levelID
                              << ", size = "
                              << std::dec << host_bytes << " to " << std::hex
                              << host_ptr << " from " << std::hex << device_ptr
                              << ", using stream " << std::hex
                              << stream << std::dec << std::endl;
                        }
                        cerrLock.unlock();
                      }
                      cudaError_t retVal;
                      CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));


                      IntVector temp(0,0,0);
                      dtask->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                            false,
                                                            IntVector(device_size.x, device_size.y, device_size.z),
                                                            host_strides.x(), host_bytes,
                                                            IntVector(device_offset.x, device_offset.y, device_offset.z),
                                                            comp,
                                                            gtype, numGhostCells,  deviceNum,
                                                            gridVar, GpuUtilities::sameDeviceSameMpiRank);


                      if (retVal == cudaErrorLaunchFailure) {
                        SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                      } else {
                        CUDA_RT_SAFE_CALL(retVal);
                      }

                    }
                    delete gridVar;
                  }
                  break;
                }
                default:
                  std::ostringstream warn;
                  warn << "  ERROR: UnifiedScheduler::initiateD2HForHugeGhostCells (" << dtask->getName() << ") variable: " 
                       << comp->m_var->getName() << " not implemented " << std::endl;
                  SCI_THROW(InternalError( warn.str() , __FILE__, __LINE__));
                  
              }
            }
          }
        }
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::initiateD2H( DetailedTask * dtask )
{
  // Request that all contiguous device arrays from the device be sent to their contiguous host array counterparts.
  // We only copy back the data needed for an upcoming task.  If data isn't needed, it can stay on the device and
  // potentially even die on the device

  // Returns true if no device data is required, thus allowing a CPU task to immediately proceed.

  void* host_ptr    = nullptr;   // host base pointer to raw data
  void* device_ptr  = nullptr;   // device base pointer to raw data
  size_t host_bytes = 0;         // raw byte count to copy to the device

  const Task* task = dtask->getTask();
  dtask->clearPreparationCollections();

  // The only thing we need to process is the requires.
  // Gather up all possible dependents and remove duplicate (we don't want to transfer some variables twice)
  std::map<labelPatchMatlDependency, const Task::Dependency*> vars;
  for (const Task::Dependency* dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread() << " InitiateD2H - For task "
                  << dtask->getName() << " checking on requires \""
                  << dependantVar->m_var->getName() << "\""
                  << " patch " <<  patches->get(i)->getID()
                  << " material " << matls->get(j)
                  << std::endl;
            }
            cerrLock.unlock();
          }
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency*>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  for (const Task::Dependency* dependantVar = task->getComputes(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(), patches->get(i)->getID(), matls->get(j), Task::Computes);
        if (vars.find(lpmd) == vars.end()) {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread() << " InitiateD2H - For task "
                  << dtask->getName() << " checking on computes \""
                  << dependantVar->m_var->getName() << "\""
                  << " patch " <<  patches->get(i)->getID()
                  << " material " << matls->get(j)
                  << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    }
  }

  // Go through each unique dependent var and see if we should queue up a D2H copy
  std::map<labelPatchMatlDependency, const Task::Dependency*>::iterator varIter;
  for (varIter = vars.begin(); varIter != vars.end(); ++varIter) {
    const Task::Dependency* dependantVar = varIter->second;
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());


    // this is so we can allocate persistent events and streams to distribute when needed
    //   one stream and one event per variable per H2D copy (numPatches * numMatls)

    int numPatches = patches->size();
    int dwIndex = dependantVar->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    const int patchID = varIter->first.m_patchID;
    const Level* level = getLevel(patches.get_rep());
    int levelID = level->getID();
    if (dependantVar->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      levelID = -1;
    }
    const Patch * patch = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
      }
    }
    if (!patch) {
      printf("ERROR:\nUnifiedScheduler::initiateD2H() patch not found.\n");
      SCI_THROW( InternalError("UnifiedScheduler::initiateD2H() patch not found.", __FILE__, __LINE__));
    }
    const int matlID = varIter->first.m_matlIndex;

    unsigned int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);
    GPUDataWarehouse * gpudw = dw->getGPUDW(deviceNum);
    OnDemandDataWarehouse::uintahSetCudaDevice(deviceNum);
    cudaStream_t* stream = dtask->getCudaStreamForThisTask(deviceNum);

    const std::string varName = dependantVar->m_var->getName();
    //TODO: Titan production hack.  A clean hack, but should be fixed. Brad P Dec 1 2016
    //There currently exists a race condition.  Suppose cellType is in both host and GPU 
    //memory.  Currently the GPU data warehouse knows it is in GPU memory, but it doesn't
    //know if it's in host memory (the GPU DW doesn't track lifetimes of host DW vars).  
    //Thread 2 - Task A requests a requires var for cellType for the host newDW, and get sit.  .
    //Thread 3 - Task B invokes the initiateD2H check, thinks there is no host isntance of cellType, 
   //             so it initiates a D2H, which performs another host allocateAndPut, and the subsequent put
    //           deletes the old entry and creates a new entry.
    //Race condition is that thread 2's pointer has been cleaned up, while thread 3 has a new one.
    //A temp fix could be to check if all host vars exist in the host dw prior to launching the task.

    //if (varName != "divQ" && varName != "RMCRTboundFlux" && varName != "radiationVolq" ) {
    //  continue;
    //}
    if (gpudw != nullptr) {
      // It's not valid on the CPU but it is on the GPU.  Copy it on over.
      if (!gpudw->isValidOnCPU( varName.c_str(), patchID, matlID, levelID) &&
          gpudw->isAllocatedOnGPU( varName.c_str(), patchID, matlID, levelID) &&
          gpudw->isValidOnGPU( varName.c_str(), patchID, matlID, levelID)) {

        const TypeDescription::Type type = dependantVar->m_var->typeDescription()->getType();
        const TypeDescription::Type datatype = dependantVar->m_var->typeDescription()->getSubType()->getType();
        switch (type) {
          case TypeDescription::CCVariable:
          case TypeDescription::NCVariable:
          case TypeDescription::SFCXVariable:
          case TypeDescription::SFCYVariable:
          case TypeDescription::SFCZVariable: {

            if (gpu_stats.active()) {
              cerrLock.lock();
              {
                gpu_stats << myRankThread() << " InitiateD2H() -"
                    // Task: " << dtask->getName()
                    << " Checking if we should copy of \""
                    << varName << "\" Patch " << patchID
                    << " Material " << matlID << std::endl;
              }
              cerrLock.unlock();
            }
            bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(varName.c_str(), patchID, matlID, levelID);
            if (performCopy) {

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread() << " InitiateD2H() -"
                      // Task: " << dtask->getName()
                      << " Yes, we are copying \""
                      << varName << "\" patch" << patchID
                      << " material " << matlID
                      << " number of ghost cells " << dependantVar->m_num_ghost_cells << " from device to host" << std::endl;
                }
                cerrLock.unlock();
              }

              // It's possible the computes data may contain ghost cells.  But a task needing to get the data
              // out of the GPU may not know this.  It may just want the var data.
              // This creates a dilemma, as the GPU var is sized differently than the CPU var.
              // So ask the GPU what size it has for the var.  Size the CPU var to match so it can copy all GPU data in.
              // When the GPU->CPU copy is done, then we need to resize the CPU var if needed to match
              // what the CPU is expecting it to be.

              // Get the host var variable
              GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(dependantVar->m_var->typeDescription()->createInstance());
              const size_t elementDataSize =
                  OnDemandDataWarehouse::getTypeDescriptionSize(dependantVar->m_var->typeDescription()->getSubType()->getType());

              // The device will have our best knowledge of the exact dimensions/ghost cells of the variable, so lets get those values.
              int3 device_low;
              int3 device_offset;
              int3 device_high;
              int3 device_size;
              GPUDataWarehouse::GhostType tempgtype;
              Ghost::GhostType gtype;
              int numGhostCells;
              gpudw->getSizes(device_low, device_high, device_size, tempgtype, numGhostCells, varName.c_str(), patchID, matlID, levelID);
              gtype = (Ghost::GhostType) tempgtype;
              device_offset = device_low;

              // Now get dimensions for the host variable.
              bool uses_SHRT_MAX = (numGhostCells == SHRT_MAX);
              Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);

              // Get the size/offset of what the host var would be with ghost cells.
              IntVector host_low, host_high, host_lowOffset, host_highOffset, host_offset, host_size;
              if (uses_SHRT_MAX) {
                level->findCellIndexRange(host_low, host_high); // including extraCells
              } else {
                Patch::getGhostOffsets(type, gtype, numGhostCells, host_lowOffset, host_highOffset);
                patch->computeExtents(basis, dependantVar->m_var->getBoundaryLayer(), host_lowOffset, host_highOffset, host_low, host_high);
              }
              host_size = host_high - host_low;
              int dwIndex = dependantVar->mapDataWarehouse();
              OnDemandDataWarehouseP dw = m_dws[dwIndex];

              // Get/make the host var
              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread() << " InitiateD2H() -"
                      << " calling allocateAndPut for "
                      << varName << " patch" << patchID
                      << " material " << matlID
                      << " level " << levelID
                      << " number of ghost cells " << numGhostCells << " from device to host" << std::endl;
                 }
                cerrLock.unlock();
              }

              // get the device var so we can get the pointer.
              GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(datatype);
              gpudw->get(*device_var, varName.c_str(), patchID, matlID, levelID);
              device_var->getArray3(device_offset, device_size, device_ptr);
              delete device_var;

              bool proceedWithCopy = false;
              // See if the size of the host var and the device var match.

              if (   device_offset.x == host_low.x()
                  && device_offset.y == host_low.y()
                  && device_offset.z == host_low.z()
                  && device_size.x   == host_size.x()
                  && device_size.y   == host_size.y()
                  && device_size.z   == host_size.z()) {
                proceedWithCopy = true;

                //Note, race condition possible here
                bool finalized = dw->isFinalized();
                if (finalized) {
                  dw->unfinalize();
                }
                if (uses_SHRT_MAX) {
                  gridVar->allocate(host_low, host_high);
                } else {
                  dw->allocateAndPut(*gridVar, dependantVar->m_var, matlID, patch, gtype, numGhostCells);
                }
                if (finalized) {
                  dw->refinalize();
                }
              } else {
                // They didn't match.  Lets see if the device var doesn't have ghost cells.
                // This can happen prior to the first timestep during initial computations when no variables had room for ghost cells.
                Patch::getGhostOffsets(type, Ghost::None, 0, host_lowOffset, host_highOffset);
                patch->computeExtents(basis, dependantVar->m_var->getBoundaryLayer(), host_lowOffset, host_highOffset, host_low, host_high);

                host_size = host_high - host_low;
                if (   device_offset.x == host_low.x()
                    && device_offset.y == host_low.y()
                    && device_offset.z == host_low.z()
                    && device_size.x   == host_size.x()
                    && device_size.y   == host_size.y()
                    && device_size.z   == host_size.z()) {

                  proceedWithCopy = true;

                  // Note, race condition possible here
                  bool finalized = dw->isFinalized();
                  if (finalized) {
                    dw->unfinalize();
                  }
                  dw->allocateAndPut(*gridVar, dependantVar->m_var, matlID, patch, Ghost::None, 0);
                  if (finalized) {
                    dw->refinalize();
                  }
                } else {
                  // The sizes STILL don't match. One more last ditch effort.  Assume it was using up to 32768 ghost cells.
                  level->findCellIndexRange(host_low, host_high);
                  host_size = host_high - host_low;
                  if (device_offset.x == host_low.x()
                       && device_offset.y == host_low.y()
                       && device_offset.z == host_low.z()
                       && device_size.x == host_size.x()
                       && device_size.y == host_size.y()
                       && device_size.z == host_size.z()) {

                    // ok, this worked.  Allocate it the large ghost cell way with getRegion
                    // Note, race condition possible here
                    bool finalized = dw->isFinalized();
                    if (finalized) {
                      dw->unfinalize();
                    }
                    gridVar->allocate(host_low, host_high);
                    if (finalized) {
                      dw->refinalize();
                    }
                    proceedWithCopy = true;
                  } else {
                    printf("ERROR:\nUnifiedScheduler::initiateD2H() - Device and host sizes didn't match.  Device size is (%d, %d, %d), and host size is (%d, %d, %d)\n", device_size.x, device_size.y, device_size.y,host_size.x(), host_size.y(),host_size.z());
                    SCI_THROW( InternalError("UnifiedScheduler::initiateD2H() - Device and host sizes didn't match.", __FILE__, __LINE__));
                  }
                }
              }

              // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
              if (proceedWithCopy) {

                host_ptr = gridVar->getBasePointer();
                host_bytes = gridVar->getDataSize();

                if (gpu_stats.active()) {
                  cerrLock.lock();
                  {
                    gpu_stats << myRankThread() << " InitiateD2H() - Copy of \""
                        << varName << "\""
                        << " patch " << patchID
                        << " material " << matlID
                        << " level " << levelID
                        << ", size = "
                        << std::dec << host_bytes
                        << " offset (" << device_offset.x << ", " << device_offset.y << ", " << device_offset.z << ")"
                        << " size (" << device_size.x << ", " << device_size.y << ", " << device_size.z << ")"
                        << " to " << std::hex << host_ptr << " from " << std::hex << device_ptr
                        << ", using stream " << std::hex
                        << stream << std::dec << std::endl;
                  }
                  cerrLock.unlock();
                }
                cudaError_t retVal;

                if (host_bytes == 0) {
                  printf("ERROR:\nUnifiedScheduler::initiateD2H() - Transfer bytes is listed as zero.\n");
                  SCI_THROW( InternalError("UnifiedScheduler::initiateD2H() - Transfer bytes is listed as zero.", __FILE__, __LINE__));
                }
                if (!host_ptr) {
                  printf("ERROR:\nUnifiedScheduler::initiateD2H() - Invalid host pointer, it was nullptr.\n");
                  SCI_THROW( InternalError("UnifiedScheduler::initiateD2H() - Invalid host pointer, it was nullptr.", __FILE__, __LINE__));
                }

                CUDA_RT_SAFE_CALL(retVal = cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));

                IntVector temp(0,0,0);
                dtask->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                      false,
                                                      IntVector(device_size.x, device_size.y, device_size.z),
                                                      elementDataSize, host_bytes,
                                                      IntVector(device_offset.x, device_offset.y, device_offset.z),
                                                      dependantVar,
                                                      gtype, numGhostCells,  deviceNum,
                                                      gridVar, GpuUtilities::sameDeviceSameMpiRank);


                if (retVal == cudaErrorLaunchFailure) {
                  SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: "+ dtask->getName(), __FILE__, __LINE__));
                } else {
                  CUDA_RT_SAFE_CALL(retVal);
                }
              }
              //delete gridVar;
            }
            break;
          }
          case TypeDescription::PerPatch: {
            bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(varName.c_str(), patchID, matlID, levelID);
            if (performCopy) {

              PerPatchBase* hostPerPatchVar = dynamic_cast<PerPatchBase*>(dependantVar->m_var->typeDescription()->createInstance());
              const bool finalized = dw->isFinalized();
              if (finalized) {
                dw->unfinalize();
              }
              dw->put(*hostPerPatchVar, dependantVar->m_var, matlID, patch);
              if (finalized) {
                dw->refinalize();
              }
              host_ptr = hostPerPatchVar->getBasePointer();
              host_bytes = hostPerPatchVar->getDataSize();

              GPUPerPatchBase* gpuPerPatchVar = OnDemandDataWarehouse::createGPUPerPatch(datatype);
              gpudw->get(*gpuPerPatchVar, varName.c_str(), patchID, matlID, levelID);
              device_ptr = gpuPerPatchVar->getVoidPointer();
              size_t device_bytes = gpuPerPatchVar->getMemSize();
              delete gpuPerPatchVar;

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread() << "initiateD2H copy of \""
                      << varName << "\", size = "
                      << std::dec << host_bytes << " to " << std::hex
                      << host_ptr << " from " << std::hex << device_ptr
                      << ", using stream " << std::hex << stream
                      << std::dec << std::endl;
                }
                cerrLock.unlock();
              }

              // TODO: Verify no memory leaks
              if (host_bytes == device_bytes) {
                CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
                dtask->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                      host_bytes, host_bytes,
                                                      dependantVar,
                                                      deviceNum,
                                                      hostPerPatchVar,
                                                      GpuUtilities::sameDeviceSameMpiRank);
              } else {
                printf("ERROR: InitiateD2H - PerPatch variable memory sizes didn't match\n");
                SCI_THROW(InternalError("InitiateD2H - PerPatch variable memory sizes didn't match", __FILE__, __LINE__));
              }
              //delete hostPerPatchVar;
            }

            break;
          }
          case TypeDescription::ReductionVariable: {
            bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(varName.c_str(), patchID, matlID, levelID);
            if (performCopy) {
              ReductionVariableBase* hostReductionVar = dynamic_cast<ReductionVariableBase*>(dependantVar->m_var->typeDescription()->createInstance());
              const bool finalized = dw->isFinalized();
              if (finalized) {
                dw->unfinalize();
              }
              dw->put(*hostReductionVar, dependantVar->m_var, patch->getLevel(), matlID);
              if (finalized) {
                dw->refinalize();
              }
              host_ptr   = hostReductionVar->getBasePointer();
              host_bytes = hostReductionVar->getDataSize();

              GPUReductionVariableBase* gpuReductionVar = OnDemandDataWarehouse::createGPUReductionVariable(datatype);
              gpudw->get(*gpuReductionVar, varName.c_str(), patchID, matlID, levelID);
              device_ptr = gpuReductionVar->getVoidPointer();
              size_t device_bytes = gpuReductionVar->getMemSize();
              delete gpuReductionVar;

              if (gpu_stats.active()) {
                cerrLock.lock();
                {
                  gpu_stats << myRankThread() << "initiateD2H copy of \""
                      << varName << "\", size = "
                      << std::dec << host_bytes << " to " << std::hex
                      << host_ptr << " from " << std::hex << device_ptr
                      << ", using stream " << std::hex << stream
                      << std::dec << std::endl;
                }
                cerrLock.unlock();
              }

              if (host_bytes == device_bytes) {
                CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
                dtask->getVarsBeingCopiedByTask().add(patch, matlID, levelID,
                                                      host_bytes, host_bytes,
                                                      dependantVar,
                                                      deviceNum,
                                                      hostReductionVar,
                                                      GpuUtilities::sameDeviceSameMpiRank);
              } else {
                printf("ERROR: InitiateD2H - Reduction variable memory sizes didn't match\n");
                SCI_THROW(InternalError("InitiateD2H - Reduction variable memory sizes didn't match", __FILE__, __LINE__));
              }
              //delete hostReductionVar;
            }
            break;
          }
          default: {
            cerrLock.lock();
            {
              std::cerr << "Variable " << varName << " is of a type that is not supported on GPUs yet." << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//
void
UnifiedScheduler::createTaskGpuDWs( DetailedTask * dtask )
{
  // Create GPU datawarehouses for this specific task only.  They will get copied into the GPU.
  // This is sizing these datawarehouses dynamically and doing it all in only one alloc per datawarehouse.
  // See the bottom of the GPUDataWarehouse.h for more information.

  std::set<unsigned int> deviceNums = dtask->getDeviceNums();
  for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums.begin(); deviceNums_it != deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;
    unsigned int numItemsInDW = dtask->getTaskVars().getTotalVars(currentDevice, Task::OldDW) + dtask->getGhostVars().getNumGhostCellCopies(currentDevice, Task::OldDW);
    if (numItemsInDW > 0) {

      size_t objectSizeInBytes = sizeof(GPUDataWarehouse)
          - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS
          + sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;

      GPUDataWarehouse* old_taskGpuDW = (GPUDataWarehouse *) malloc(objectSizeInBytes);
      //cudaHostRegister(old_taskGpuDW, objectSizeInBytes, cudaHostRegisterDefault);
      std::ostringstream out;
      out << "Old task GPU DW" << " MPIRank: " << Uintah::Parallel::getMPIRank() << " Task: " << dtask->getTask()->getName();
      old_taskGpuDW->init( currentDevice, out.str());
      old_taskGpuDW->setDebug(gpudbg.active());

      old_taskGpuDW->init_device(objectSizeInBytes, numItemsInDW);
      dtask->setTaskGpuDataWarehouse(currentDevice, Task::OldDW, old_taskGpuDW);

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
             << " UnifiedScheduler::createTaskGpuDWs() - Created an old Task GPU DW for task " <<  dtask->getName()
             << " for device #" << currentDevice
             << " at host address " << old_taskGpuDW
             << " to contain " << dtask->getTaskVars().getTotalVars(currentDevice, Task::OldDW)
             << " task variables and " << dtask->getGhostVars().getNumGhostCellCopies(currentDevice, Task::OldDW)
             << " ghost cell copies." << std::endl;
        }
        cerrLock.unlock();
      }
    }

    numItemsInDW = dtask->getTaskVars().getTotalVars(currentDevice, Task::NewDW) + dtask->getGhostVars().getNumGhostCellCopies(currentDevice, Task::NewDW);
    if (numItemsInDW > 0) {

      size_t objectSizeInBytes = sizeof(GPUDataWarehouse)
          - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS
          + sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;
      GPUDataWarehouse* new_taskGpuDW = (GPUDataWarehouse *) malloc(objectSizeInBytes);
      //cudaHostRegister(new_taskGpuDW, objectSizeInBytes, cudaHostRegisterDefault);
      std::ostringstream out;
      out << "New task GPU DW" << " MPIRank: " << Uintah::Parallel::getMPIRank() << " Thread:" << Impl::t_tid << " Task: " << dtask->getName();
      new_taskGpuDW->init(currentDevice, out.str());
      new_taskGpuDW->setDebug(gpudbg.active());
      new_taskGpuDW->init_device(objectSizeInBytes, numItemsInDW);

      dtask->setTaskGpuDataWarehouse(currentDevice, Task::NewDW, new_taskGpuDW);

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
             << " UnifiedScheduler::createTaskGpuDWs() - Created a new Task GPU DW for task " <<  dtask->getName()
             << " for device #" << currentDevice
             << " at host address " << new_taskGpuDW
             << " to contain " << dtask->getTaskVars().getTotalVars(currentDevice, Task::NewDW)
             << " task variables and " << dtask->getGhostVars().getNumGhostCellCopies(currentDevice, Task::NewDW)
             << " ghost cell copies." << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::assignDevicesAndStreams( DetailedTask * dtask )
{

  // Figure out which device this patch was assigned to.
  // If a task has multiple patches, then assign all.  Most tasks should
  // only end up on one device.  Only tasks like data archiver's output variables
  // work on multiple patches which can be on multiple devices.
  std::map<const Patch *, int>::iterator it;
  for (int i = 0; i < dtask->getPatches()->size(); i++) {
    const Patch* patch = dtask->getPatches()->get(i);
    int index = GpuUtilities::getGpuIndexForPatch(patch);
    if (index >= 0) {
      for (int i = 0; i < dtask->getTask()->maxStreamsPerTask(); i++) {
        if (dtask->getCudaStreamForThisTask(i) == nullptr) {
          dtask->assignDevice(0); 
          cudaStream_t* stream = GPUMemoryPool::getCudaStreamFromPool(i);
          dtask->setCudaStreamForThisTask(i, stream);
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread() << " Assigning for task " << dtask->getName() << " at " << std::hex << dtask
                    << " stream " << stream << std::dec
                    << " for device " << index
                    << std::endl;
            }
            cerrLock.unlock();
          }
        }
      }
    
    } else {
      cerrLock.lock();
      {
        std::cerr << "ERROR: Could not find the assigned GPU for this patch." << std::endl;
      }
      cerrLock.unlock();
      exit(-1);
    }
  }
}


/*
  void
  UnifiedScheduler::assignDevicesAndStreams( DetailedTask * dtask )
  {

    // Figure out which device this patch was assigned to.
    // If a task has multiple patches, then assign all.  Most tasks should
    // only end up on one device.  Only tasks like data archiver's output variables
    // work on multiple patches which can be on multiple devices.
    std::map<const Patch *, int>::iterator it;
    for (int i = 0; i < dtask->getPatches()->size(); i++) {
      const Patch* patch = dtask->getPatches()->get(i);
      int index = GpuUtilities::getGpuIndexForPatch(patch);
      if (index >= 0) {
        // See if this task doesn't yet have a stream for this GPU device.
        if (dtask->getCudaStreamForThisTask(index) == nullptr) {
          dtask->assignDevice(index);
          cudaStream_t* stream = GPUMemoryPool::getCudaStreamFromPool(index);
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread() << " Assigning for task " << dtask->getName() << " at " << std::hex << dtask
              << " stream " << stream << std::dec
              << " for device " << index
              << std::endl;
            }
            cerrLock.unlock();
          }
          dtask->setCudaStreamForThisTask(index, stream);
        }
      } else {
        cerrLock.lock();
        {
          std::cerr << "ERROR: Could not find the assigned GPU for this patch." << std::endl;
        }
        cerrLock.unlock();
        exit(-1);
      }
    }
  }
*/
//______________________________________________________________________
//
void
UnifiedScheduler::assignDevicesAndStreamsFromGhostVars( DetailedTask * dtask )
{
  // Go through the ghostVars collection and look at the patch where all ghost cells are going.
  std::set<unsigned int> & destinationDevices = dtask->getGhostVars().getDestinationDevices();
  for (std::set<unsigned int>::iterator iter = destinationDevices.begin(); iter != destinationDevices.end(); ++iter) {
    // see if this task was already assigned a stream.
    if (dtask->getCudaStreamForThisTask(*iter) == nullptr) {
      dtask->assignDevice(*iter);
      dtask->setCudaStreamForThisTask(*iter, GPUMemoryPool::getCudaStreamFromPool(*iter));
    }
  }
}

//______________________________________________________________________
//
void
UnifiedScheduler::assignStatusFlagsToPrepareACpuTask( DetailedTask * dtask )
{
  //Keep track of all variables created or modified by a CPU task.  It also keeps track of ghost cells for a task.
  //This method seems more like fitting a square peg into a round hole.  It tries to temporarily bridge a gap between
  //the OnDemand Data Warehouse and the GPU Data Warehouse.  The OnDemand DW allocates variables on the fly during task execution
  //and also inflates vars to gather ghost cells on task execution.  The GPU DW prepares all variables and manages ghost cell copies
  //prior to task execution.
  //This method was designed to solve a use case where a CPU task created a var, then another CPU task modified it, then a GPU
  //task required it, then a CPU output task needed it.  Because the CPU variable didn't get status flags attached to it due to
  //it being in the OnDemand Data Warehouse, the Unified Scheduler assumed the only copy of the variable existed in GPU memory
  //so it copied it out of GPU memory into host memory right in the middle of when the CPU output task was executing, causing
  //a concurrency race condition because that variable was already in host memory.  By trying to track the host memory statuses
  //for variables, this should hopefully prevent those race conditions.

  //This probably isn't perfect, but should get us through the next few months, and hopefully gets replaced
  //when we can remove the "OnDemand" part of the OnDemand Data Warehouse with a Unified DataWarehouse.

  //Loop through all computes.  Create status flags of "allocating" for them.  Do not track ghost cells, as ghost cells
  //are created by copying a

  //Loop through all modifies.  Create status flags of "allocated", undoing any "valid" flags.

  //Loop through all requires.  If they have a ghost cell requirement, we can't do much about it.

}


//______________________________________________________________________
//
void
UnifiedScheduler::findIntAndExtGpuDependencies( DetailedTask * dtask
                                              , int            iteration
                                              , int            t_id
                                              )
{
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << myRankThread()
                << " findIntAndExtGpuDependencies - task "
                << *dtask
                << std::endl;
    }
    cerrLock.unlock();
  }

  dtask->clearPreparationCollections();

  // Prepare internal dependencies.  Only makes sense if we have GPUs that we are using.
  if (Uintah::Parallel::usingDevice()) {
    // If we have ghost cells coming from a GPU to another GPU on the same node
    // This does not cover going to another node (the loop below handles external
    // dependencies) That is handled in initiateH2D()
    // This does not handle preparing from a CPU to a same node GPU.  That is handled in initiateH2D()
    for (DependencyBatch* batch = dtask->getInternalComputes(); batch != 0; batch = batch->m_comp_next) {
      for (DetailedDep* req = batch->m_head; req != 0; req = req->m_next) {
        if ((req->m_comm_condition == DetailedDep::FirstIteration && iteration > 0)
            || (req->m_comm_condition == DetailedDep::SubsequentIterations
                && iteration == 0)
            || (m_no_copy_data_vars.count(req->m_req->m_var->getName()) > 0)) {
          // See comment in DetailedDep about CommCondition
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread()
                        << "   Preparing GPU dependencies, ignoring conditional send for requries " << *req
                        << std::endl;
            }
            cerrLock.unlock();
          }
          continue;
        }
        // if we send/recv to an output task, don't send/recv if not an output timestep
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output
            && !m_out_port->isOutputTimestep() && !m_out_port->isCheckpointTimestep()) {
          if (gpu_stats.active()) {
            cerrLock.lock();
            gpu_stats << myRankThread()
                << "   Preparing GPU dependencies, ignoring non-output-timestep send for "
                << *req << std::endl;
            cerrLock.unlock();
          }
          continue;
        }
        OnDemandDataWarehouse* dw = m_dws[req->m_req->mapDataWarehouse()].get_rep();
        const VarLabel* posLabel;
        OnDemandDataWarehouse* posDW;

        // the load balancer is used to determine where data was in the old dw on the prev timestep -
        // pass it in if the particle data is on the old dw
        LoadBalancerPort * lb = nullptr;

        if (!m_reloc_new_pos_label && m_parent_scheduler) {
          posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
          posLabel = m_parent_scheduler->m_reloc_new_pos_label;
        }
        else {
          // on an output task (and only on one) we require particle variables from the NewDW
          if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::NewDW)].get_rep();
          }
          else {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::OldDW)].get_rep();
            lb = getLoadBalancer();
          }
          posLabel = m_reloc_new_pos_label;
        }
        // Load information which will be used to later invoke a kernel to copy this range out of the GPU.
        prepareGpuDependencies(dtask, batch, posLabel, dw, posDW, req, lb, GpuUtilities::anotherDeviceSameMpiRank);
      }
    }  // end for (DependencyBatch * batch = task->getInteranlComputes() )

    // Prepare external dependencies.  The only thing that needs to be prepared is
    // getting ghost cell data from a GPU into a flat array and copied to host memory
    // so that the MPI engine can treat it normally.
    // That means this handles GPU->other node GPU and GPU->other node CPU.
    //

    for (DependencyBatch* batch = dtask->getComputes(); batch != 0;
        batch = batch->m_comp_next) {
      for (DetailedDep* req = batch->m_head; req != 0; req = req->m_next) {
        if ((req->m_comm_condition == DetailedDep::FirstIteration && iteration > 0) || (req->m_comm_condition == DetailedDep::SubsequentIterations && iteration == 0)
            || (m_no_copy_data_vars.count(req->m_req->m_var->getName()) > 0)) {
          // See comment in DetailedDep about CommCondition
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread()
                        << "   Preparing GPU dependencies, ignoring conditional send for requires: " << *req
                        << std::endl;
            }
            cerrLock.unlock();
          }
          continue;
        }
        // if we send/recv to an output task, don't send/recv if not an output timestep
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output && !m_out_port->isOutputTimestep() && !m_out_port->isCheckpointTimestep()) {
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread()
                        << "   Preparing GPU dependencies, ignoring non-output-timestep send for requires: " << *req
                        << std::endl;
            }
            cerrLock.unlock();
          }
          continue;
        }
        OnDemandDataWarehouse* dw = m_dws[req->m_req->mapDataWarehouse()].get_rep();

        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << myRankThread()
                      << " --> Preparing GPU dependencies for sending requires: " << *req
                      << ", ghost-type: " << req->m_req->m_gtype
                      << ", number of ghost cells: " << req->m_req->m_num_ghost_cells
                      << " from dw " << dw->getID()
                      << std::endl;
          }
          cerrLock.unlock();
        }
        const VarLabel* posLabel;
        OnDemandDataWarehouse* posDW;

        // the load balancer is used to determine where data was in the old dw on the prev timestep -
        // pass it in if the particle data is on the old dw
        LoadBalancerPort * lb = 0;

        if (!m_reloc_new_pos_label && m_parent_scheduler) {
          posDW    = m_dws[req->m_req->m_task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
          posLabel = m_parent_scheduler->m_reloc_new_pos_label;
        }
        else {
          // on an output task (and only on one) we require particle variables from the NewDW
          if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::NewDW)].get_rep();
          }
          else {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::OldDW)].get_rep();
            lb    = getLoadBalancer();
          }
          posLabel = m_reloc_new_pos_label;
        }
        // Load information which will be used to later invoke a kernel to copy this range out of the GPU.
        prepareGpuDependencies(dtask, batch, posLabel, dw, posDW, req, lb, GpuUtilities::anotherMpiRank);
      }
    }  // end for (DependencyBatch * batch = task->getComputes() )
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::syncTaskGpuDWs( DetailedTask * dtask )
{

  // For each GPU datawarehouse, see if there are ghost cells listed to be copied
  // if so, launch a kernel that copies them.
  std::set<unsigned int> deviceNums = dtask->getDeviceNums();
  GPUDataWarehouse *taskgpudw;
  for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums.begin(); deviceNums_it != deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;
    taskgpudw = dtask->getTaskGpuDataWarehouse(currentDevice,Task::OldDW);
    if (taskgpudw) {
      taskgpudw->syncto_device(dtask->getCudaStreamForThisTask(currentDevice));
    }
    taskgpudw = dtask->getTaskGpuDataWarehouse(currentDevice,Task::NewDW);
    if (taskgpudw) {
      taskgpudw->syncto_device(dtask->getCudaStreamForThisTask(currentDevice));
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::performInternalGhostCellCopies( DetailedTask * dtask )
{

  // For each GPU datawarehouse, see if there are ghost cells listed to be copied
  // if so, launch a kernel that copies them.
  std::set<unsigned int> deviceNums = dtask->getDeviceNums();
  for (std::set<unsigned int>::const_iterator deviceNums_it = deviceNums.begin(); deviceNums_it != deviceNums.end(); ++deviceNums_it) {
    const unsigned int currentDevice = *deviceNums_it;
    if (dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW) != nullptr
        && dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW)->ghostCellCopiesNeeded()) {
      dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW)->copyGpuGhostCellsToGpuVarsInvoker(dtask->getCudaStreamForThisTask(currentDevice));
    } else {
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
                    << " A No internal ghost cell copies needed for this task \""
                    << dtask->getName() << "\"\'s old DW"
                    << std::endl;
        }
        cerrLock.unlock();
      }
    }
    if (dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW) != nullptr
        && dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW)->ghostCellCopiesNeeded()) {
      dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW)->copyGpuGhostCellsToGpuVarsInvoker(dtask->getCudaStreamForThisTask(currentDevice));
    } else {
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
                    << " B No internal ghost cell copies needed for this task \""
                    << dtask->getName() << "\"\'s new DW"
                    << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::copyAllGpuToGpuDependences( DetailedTask * dtask )
{

  // Iterate through the ghostVars, find all whose destination is another GPU same MPI rank
  // Get the destination device, the size
  // And do a straight GPU to GPU copy.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = dtask->getGhostVars().getMap();
  for (std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {
    if (it->second.m_dest == GpuUtilities::anotherDeviceSameMpiRank) {
      //TODO: Needs a particle section

      IntVector ghostLow = it->first.m_sharedLowCoordinates;
      IntVector ghostHigh = it->first.m_sharedHighCoordinates;
      IntVector ghostSize(ghostHigh.x() - ghostLow.x(), ghostHigh.y() - ghostLow.y(), ghostHigh.z() - ghostLow.z());
      int3 device_source_offset;
      int3 device_source_size;

      //get the source variable from the source GPU DW
      void *device_source_ptr;
      size_t elementDataSize = it->second.m_xstride;
      size_t memSize = ghostSize.x() * ghostSize.y() * ghostSize.z() * elementDataSize;
      GPUGridVariableBase* device_source_var = OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];
      GPUDataWarehouse* gpudw = dw->getGPUDW(it->second.m_sourceDeviceNum);
      gpudw->getStagingVar(*device_source_var,
                 it->first.m_label.c_str(),
                 it->second.m_sourcePatchPointer->getID(),
                 it->first.m_matlIndx,
                 it->first.m_levelIndx,
                 make_int3(ghostLow.x(),ghostLow.y(), ghostLow.z()),
                 make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_source_var->getArray3(device_source_offset, device_source_size, device_source_ptr);

      //Get the destination variable from the destination GPU DW
      gpudw = dw->getGPUDW(it->second.m_destDeviceNum);
      int3 device_dest_offset;
      int3 device_dest_size;
      void *device_dest_ptr;
      GPUGridVariableBase* device_dest_var = OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      gpudw->getStagingVar(*device_dest_var,
                     it->first.m_label.c_str(),
                     it->second.m_destPatchPointer->getID(),
                     it->first.m_matlIndx,
                     it->first.m_levelIndx,
                     make_int3(ghostLow.x(),ghostLow.y(), ghostLow.z()),
                     make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_dest_var->getArray3(device_dest_offset, device_dest_size, device_dest_ptr);


      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << myRankThread()
                    << " GpuDependenciesToHost()  - \""
                     << "GPU to GPU peer transfer from GPU #"
                     << it->second.m_sourceDeviceNum << " to GPU #"
                     << it->second.m_destDeviceNum << " for label "
                     << it->first.m_label << " from patch "
                     << it->second.m_sourcePatchPointer->getID() << " to patch "
                     << it->second.m_destPatchPointer->getID() << " matl "
                     << it->first.m_matlIndx << " level "
                     << it->first.m_levelIndx << " size = "
                     << std::dec << memSize << " from ptr " << std::hex
                     << device_source_ptr << " to ptr " << std::hex << device_dest_ptr
                     << ", using stream " << std::hex
                     << dtask->getCudaStreamForThisTask(it->second.m_sourceDeviceNum) << std::dec
                     << std::endl;
        }
        cerrLock.unlock();
      }

      // We can run peer copies from the source or the device stream.  While running it
      // from the device technically is said to be a bit slower, it's likely just
      // to an extra event being created to manage blocking the destination stream.
      // By putting it on the device we are able to not need a synchronize step after
      // all the copies, because any upcoming API call will use the streams and be
      // naturally queued anyway.  When a copy completes, anything placed in the
      // destination stream can then process.
      //   Note: If we move to UVA, then we could just do a straight memcpy

      cudaStream_t* stream = dtask->getCudaStreamForThisTask(it->second.m_destDeviceNum);
      OnDemandDataWarehouse::uintahSetCudaDevice(it->second.m_destDeviceNum);

      if (simulate_multiple_gpus.active()) {
        CUDA_RT_SAFE_CALL(cudaMemcpyPeerAsync(device_dest_ptr, 0, device_source_ptr, 0, memSize, *stream));
       } else {
        CUDA_RT_SAFE_CALL(cudaMemcpyPeerAsync(device_dest_ptr, it->second.m_destDeviceNum, device_source_ptr, it->second.m_sourceDeviceNum, memSize, *stream));
      }
    }
  }
}


//______________________________________________________________________
//
void
UnifiedScheduler::copyAllExtGpuDependenciesToHost( DetailedTask * dtask )
{

  bool copiesExist = false;

  // If we put it in ghostVars, then we copied it to an array on the GPU (D2D).  Go through the ones that indicate
  // they are going to another MPI rank.  Copy them out to the host (D2H).  To make the engine cleaner for now,
  // we'll then do a H2H copy step into the variable.
  // In the future, to be more efficient, we could skip the host to host copy and instead have sendMPI() send the
  // array we get from the device instead.
  // To be even more efficient than that, if everything is pinned, unified addressing set up, and CUDA aware MPI
  // used, then we could pull everything out via MPI that way and avoid the manual D2H copy and the H2H copy.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> & ghostVarMap = dtask->getGhostVars().getMap();
  for (std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {
    //TODO: Needs a particle section
    if (it->second.m_dest == GpuUtilities::anotherMpiRank) {
      void* host_ptr    = nullptr;    // host base pointer to raw data
      void* device_ptr  = nullptr;    // device base pointer to raw data
      size_t host_bytes = 0;
      IntVector host_low, host_high, host_offset, host_size, host_strides;
      int3 device_offset;
      int3 device_size;


      // We created a temporary host variable for this earlier,
      // and the deviceVars collection knows about it.  It's set as a foreign var.
      IntVector ghostLow = it->first.m_sharedLowCoordinates;
      IntVector ghostHigh = it->first.m_sharedHighCoordinates;
      IntVector ghostSize(ghostHigh.x() - ghostLow.x(), ghostHigh.y() - ghostLow.y(), ghostHigh.z() - ghostLow.z());
      DeviceGridVariableInfo item = dtask->getDeviceVars().getStagingItem(it->first.m_label,
                 it->second.m_sourcePatchPointer,
                 it->first.m_matlIndx,
                 it->first.m_levelIndx,
                 ghostLow,
                 ghostSize,
                 (const int)it->first.m_dataWarehouse);
      GridVariableBase* tempGhostVar = (GridVariableBase*)item.m_var;

      tempGhostVar->getSizes(host_low, host_high, host_offset, host_size, host_strides);

      host_ptr = tempGhostVar->getBasePointer();
      host_bytes = tempGhostVar->getDataSize();

      // copy the computes data back to the host
      //d2hComputesLock_.writeLock();
      //{

        GPUGridVariableBase* device_var = OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
        OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];
        GPUDataWarehouse* gpudw = dw->getGPUDW(it->second.m_sourceDeviceNum);
        gpudw->getStagingVar(*device_var,
                   it->first.m_label.c_str(),
                   it->second.m_sourcePatchPointer->getID(),
                   it->first.m_matlIndx,
                   it->first.m_levelIndx,
                   make_int3(ghostLow.x(),ghostLow.y(), ghostLow.z()),
                   make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
        device_var->getArray3(device_offset, device_size, device_ptr);

        // if offset and size is equal to CPU DW, directly copy back to CPU var memory;
        if (device_offset.x == host_low.x()
            && device_offset.y == host_low.y()
            && device_offset.z == host_low.z()
            && device_size.x == host_size.x()
            && device_size.y == host_size.y()
            && device_size.z == host_size.z()) {

          // Since we know we need a stream, obtain one.
          cudaStream_t* stream = dtask->getCudaStreamForThisTask(it->second.m_sourceDeviceNum);
          OnDemandDataWarehouse::uintahSetCudaDevice(it->second.m_sourceDeviceNum);
          if (gpu_stats.active()) {
            cerrLock.lock();
            {
              gpu_stats << myRankThread()
                        << " copyAllExtGpuDependenciesToHost()  - \""
                        << it->first.m_label << "\", size = "
                        << std::dec << host_bytes << " to " << std::hex
                        << host_ptr << " from " << std::hex << device_ptr
                        << ", using stream " << std::hex
                        << dtask->getCudaStreamForThisTask(it->second.m_sourceDeviceNum) << std::dec
                        << std::endl;
            }
            cerrLock.unlock();
          }

          CUDA_RT_SAFE_CALL(cudaMemcpyAsync(host_ptr, device_ptr, host_bytes, cudaMemcpyDeviceToHost, *stream));
          copiesExist = true;
        } else {
          std::cerr << "unifiedSCheduler::GpuDependenciesToHost() - Error - The host and device variable sizes did not match.  Cannot copy D2H." << std::endl;
          SCI_THROW(InternalError("Error - The host and device variable sizes did not match.  Cannot copy D2H", __FILE__, __LINE__));
        }

      //}
      //d2hComputesLock_.writeUnlock();

      delete device_var;
    }
  }

  if (copiesExist) {

    // Wait until all streams are done
    // Further optimization could be to check each stream one by one and make copies before waiting for other streams to complete.
    // TODO: There's got to be a better way to do this.
    while (!dtask->checkAllCudaStreamsDoneForThisTask()) {
      // TODO - Let's figure this out soon, APH 06/09/16
      //sleep?
      //printf("Sleeping\n");
    }

    for (std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::const_iterator it = ghostVarMap.begin(); it != ghostVarMap.end(); ++it) {

      if (it->second.m_dest == GpuUtilities::anotherMpiRank) {
        // TODO: Needs a particle section
        IntVector host_low, host_high, host_offset, host_size, host_strides;
        OnDemandDataWarehouseP dw = m_dws[(const int)it->first.m_dataWarehouse];

        //We created a temporary host variable for this earlier,
        //and the deviceVars collection knows about it.
        IntVector ghostLow = it->first.m_sharedLowCoordinates;
        IntVector ghostHigh = it->first.m_sharedHighCoordinates;
        IntVector ghostSize(ghostHigh.x() - ghostLow.x(), ghostHigh.y() - ghostLow.y(), ghostHigh.z() - ghostLow.z());
        DeviceGridVariableInfo item = dtask->getDeviceVars().getStagingItem(it->first.m_label, it->second.m_sourcePatchPointer,
                                                                            it->first.m_matlIndx, it->first.m_levelIndx, ghostLow,
                                                                            ghostSize, (const int)it->first.m_dataWarehouse);
        GridVariableBase* tempGhostVar = (GridVariableBase*)item.m_var;

        //Also get the existing host copy
        GridVariableBase* gridVar = dynamic_cast<GridVariableBase*>(it->second.m_label->typeDescription()->createInstance());

        //Get the coordinate low/high of the host copy.
        const Patch * patch = it->second.m_sourcePatchPointer;
        TypeDescription::Type type = it->second.m_label->typeDescription()->getType();
        IntVector lowIndex, highIndex;
        bool uses_SHRT_MAX = (item.m_dep->m_num_ghost_cells == SHRT_MAX);
        if (uses_SHRT_MAX) {
          const Level * level = patch->getLevel();
          level->computeVariableExtents(type, lowIndex, highIndex);
        } else {
          Patch::VariableBasis basis = Patch::translateTypeToBasis(it->second.m_label->typeDescription()->getType(), false);
          patch->computeVariableExtents(basis, item.m_dep->m_var->getBoundaryLayer(), item.m_dep->m_gtype, item.m_dep->m_num_ghost_cells, lowIndex, highIndex);
        }

        // If it doesn't exist yet on the host, create it.  If it does exist on the host, then
        // if we got here that meant the host data was invalid and the device data was valid, so
        // nuke the old contents and create a new one.  (Should we just get a mutable var instead
        // as it should be the size we already need?)  This process is admittedly a bit hacky,
        // as now the var will be both partially valid and invalid.  The ghost cell region is now
        // valid on the host, while the rest of the host var would be invalid.
        // Since we are writing to an old data warehouse (from device to host), we need to
        // temporarily unfinalize it.
        const bool finalized = dw->isFinalized();
        if (finalized) {
          dw->unfinalize();
        }

        if (!dw->exists(item.m_dep->m_var, it->first.m_matlIndx, it->second.m_sourcePatchPointer)) {
          dw->allocateAndPut(*gridVar, item.m_dep->m_var, it->first.m_matlIndx,
                             it->second.m_sourcePatchPointer, item.m_dep->m_gtype,
                             item.m_dep->m_num_ghost_cells);
        } else {
          // Get a const variable in a non-constant way.
          // This assumes the variable has already been resized properly, which is why ghost cells are set to zero.
          // TODO: Check sizes anyway just to be safe.
          dw->getModifiable(*gridVar, item.m_dep->m_var, it->first.m_matlIndx, it->second.m_sourcePatchPointer, Ghost::None, 0);

        }
        // Do a host-to-host copy to bring the device data now on the host into the host-side variable so
        // that sendMPI can easily find the data as if no GPU were involved at all.
        gridVar->copyPatch(tempGhostVar, ghostLow, ghostHigh );
        if(finalized) {
          dw->refinalize();
        }

        // let go of our reference counters.
        delete gridVar;
        delete tempGhostVar;
      }
    }
  }
}

#endif


//______________________________________________________________________
//  generate string   <MPI_rank>.<Thread_ID>
std::string
UnifiedScheduler::myRankThread()
{
  std::ostringstream out;
  out << Uintah::Parallel::getMPIRank()<< "." << Impl::t_tid;
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
UnifiedSchedulerWorker::UnifiedSchedulerWorker( UnifiedScheduler * scheduler )
  : m_scheduler{ scheduler }
  , m_rank{ scheduler->d_myworld->myrank() }
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
double
UnifiedSchedulerWorker::getWaitTime()
{
  return m_wait_time;
}
