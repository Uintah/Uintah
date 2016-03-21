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

#include <CCA/Components/Schedulers/ThreadedTaskScheduler.hpp>

#include <CCA/Components/Schedulers/CommunicationList.hpp>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DOUT.hpp>

#include <atomic>
#include <thread>

#include <sched.h>

#define USE_PACKING

using namespace Uintah;

//______________________________________________________________________
//
namespace {

Dout g_mpi_dbg(   "MPIDBG"  , false );

std::mutex      g_lb_mutex;

thread_local CommPool::handle t_emplace{};
thread_local CommPool::handle t_find{};

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

TaskRunner           * g_runners[MAX_THREADS]        = {};
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
void init_threads( ThreadedTaskScheduler * sched, int num_threads )
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
ThreadedTaskScheduler::ThreadedTaskScheduler( const ProcessorGroup        * myworld
                                            , const Output                * oport
                                            )
  : SchedulerCommon( myworld, oport )
  , m_output_port{ oport }
{

}


//______________________________________________________________________
//
ThreadedTaskScheduler::~ThreadedTaskScheduler()
{

}


//______________________________________________________________________
//
void ThreadedTaskScheduler::problemSetup( const ProblemSpecP & prob_spec, SimulationStateP & state )
{
  m_num_threads = Uintah::Parallel::getNumThreads();

  m_task_pool = TaskPool{ static_cast<size_t>(m_num_threads) };

  if ((m_num_threads < 1) && Uintah::Parallel::usingMPI()) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "Error: no thread number specified for ThreadedScheduler" << std::endl;
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
    std::cout << "   WARNING: Component tasks must be thread safe when using this scheduler.\n"
              << "   Using " << m_num_threads << " threads for task execution,"
              << " main thread handles MPI collectives." << std::endl;
  }

  // this spawns threads, sets affinity, etc
  init_threads(this, m_num_threads);

  SchedulerCommon::problemSetup(prob_spec, state);
}


//______________________________________________________________________
//
SchedulerP ThreadedTaskScheduler::createSubScheduler()
{
  throw ProblemSetupException("createSubScheduler() not implemented for ThreadedTaskScheduler.", __FILE__, __LINE__);
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::execute(  int tgnum /*=0*/ , int iteration /*=0*/ )
{
  ASSERTRANGE(tgnum, 0, static_cast<int>(graphs.size()));

  RuntimeStats::initialize_timestep(graphs);

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

  TaskPool::handle insert_handle;
  TaskPool::handle find_handle;

  for (int i = 0; i < m_num_tasks; ++i) {
    DetailedTask* dtask = m_detailed_tasks->localTask(i);
    dtask->resetDependencyCounts();
  }

  makeTaskGraphDoc(m_detailed_tasks, d_myworld->myrank());

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(m_detailed_tasks, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  // clear & resize task phase, etc bookkeeping data structures
  m_num_messages      = 0;
  m_message_volume    = 0;
  m_abort             = false;
  m_abort_point       = 987654;
  m_current_iteration = iteration;
  m_num_tasks_done.store(0, std::memory_order_relaxed);
  m_current_phase.store(0, std::memory_order_relaxed);
  m_num_phases = tg->getNumTaskPhases();
  m_phase_tasks.clear();
  m_phase_tasks.resize(m_num_phases, 0);
  m_phase_sync_tasks.clear();
  m_phase_sync_tasks.resize(m_num_phases, nullptr);
  m_phase_tasks_done.release();
  m_phase_tasks_done = atomic_int_array(new std::atomic<int>[m_num_phases]{});

  for (int i = 0; i < m_num_tasks; ++i) {
    DetailedTask* dtask = m_detailed_tasks->localTask(i);
    // save the reduction task and once per proc task for later execution
    if ((dtask->getTask()->getType() == Task::Reduction) || (dtask->getTask()->usesMPI())) {
      m_phase_sync_tasks[dtask->getTask()->d_phase] = dtask;
    } else {
      insert_handle = m_task_pool.emplace(insert_handle, dtask);
    }
  }

  // count the number of tasks in each task-phase
  //   each task is assigned a task-phase in TaskGraph::createDetailedDependencies()
  for (int i = 0; i < m_num_tasks; ++i) {
    ++m_phase_tasks[m_detailed_tasks->localTask(i)->getTask()->d_phase];
  }

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
  while (m_num_tasks_done.load(std::memory_order_relaxed) < m_num_tasks) {

    if (m_phase_tasks[m_current_phase.load(std::memory_order_relaxed)] ==
        m_phase_tasks_done[m_current_phase.load(std::memory_order_relaxed)].load(std::memory_order_relaxed)) {  // this phase done, goto next phase
      m_current_phase.fetch_add(1, std::memory_order_relaxed);
    }
    // if it is time to run reduction or once-per-proc task
    else if ((m_phase_sync_tasks[m_current_phase.load(std::memory_order_relaxed)] != nullptr) &&
             (m_phase_tasks_done[m_current_phase].load(std::memory_order_relaxed) ==
              m_phase_tasks[m_current_phase.load(std::memory_order_relaxed)] - 1)) {
      DetailedTask* sync_task = m_phase_sync_tasks[m_current_phase.load(std::memory_order_relaxed)];
      if (sync_task->getTask()->getType() == Task::Reduction) {
        ASSERT(sync_task->getRequires().size() == 0)
        run_reduction_task(sync_task);
      }
      else {  // Task::OncePerProc task
        ASSERT(sync_task->getTask()->usesMPI());
        post_MPI_recvs(sync_task, m_abort, m_abort_point, iteration);
        sync_task->markInitiated();
        ASSERT(sync_task->getExternalDepCount() == 0)
        run_task(sync_task, iteration);
      }
      ASSERT(sync_task->getTask()->d_phase == m_current_phase.load(std::memory_order_relaxed));
      m_num_tasks_done.fetch_add(1, std::memory_order_relaxed);
      m_phase_tasks_done[sync_task->getTask()->d_phase].fetch_add(1, std::memory_order_relaxed);
    } else {
      select_tasks(iteration, find_handle);
    }
  }

  ASSERT(m_task_pool.empty());

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

  // Copy the restart flag to all processors
  copy_restart_flag(tgnum);

  finalizeTimestep();

  {
    int64_t num_patches = 0;
    int64_t num_cells = 0;
    int64_t num_particles = 0;

    // collect local grid information
    {
      OnDemandDataWarehouseP dw = dws[dws.size() - 1];
      const GridP grid(const_cast<Grid*>(dw->getGrid()));
      const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(grid)->getSubset(d_myworld->myrank());
      num_patches = myPatches->size();
      for (int p = 0; p < myPatches->size(); p++) {
        const Patch* patch = myPatches->get(p);
        IntVector range = patch->getExtraCellHighIndex() - patch->getExtraCellLowIndex();
        num_cells += range.x() * range.y() * range.z();

        // go through all materials since getting an MPMMaterial correctly would depend on MPM
        for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
          if (dw->haveParticleSubset(m, patch))
            num_particles += dw->getParticleSubset(m, patch)->numParticles();
        }
      }
    }

    Dout grid_stats{"GridStats", true};
    if (grid_stats) {

      RuntimeStats::register_report( grid_stats
                                   , "Patches"
                                   , RuntimeStats::Count
                                   , [num_patches]() { return num_patches; }
                                   );
      RuntimeStats::register_report( grid_stats
                                   , "Cells"
                                   , RuntimeStats::Count
                                   , [num_cells]() { return num_cells; }
                                   );
      RuntimeStats::register_report( grid_stats
                                   , "Particles"
                                   , RuntimeStats::Count
                                   , [num_particles]() { return num_particles; }
                                   );
    }
  }

  RuntimeStats::report(d_myworld->getComm(), d_sharedState->d_runTimeStats);

} // end execute()


//______________________________________________________________________
//
void ThreadedTaskScheduler::verifyChecksum()
{
#if SCI_ASSERTION_LEVEL >= 3
  if (Uintah::Parallel::usingMPI()) {

    // Compute a simple checksum to make sure that all processes are trying to
    // execute the same graph.  We should do two things in the future:
    //  - make a flag to turn this off
    //  - make the checksum more sophisticated
    int checksum = 0;
    int numSpatialTasks = 0;
    for (unsigned i = 0; i < graphs.size(); i++) {
      checksum += graphs[i]->getTasks().size();

      // This begins addressing the issue of making the global checksum more sophisticated:
      //   check if any tasks were spatially scheduled - TaskType::Spatial, meaning no computes, requires or modifies
      //     e.g. RMCRT radiometer task, which is not scheduled on all patches
      //          these Spatial tasks won't count toward the global checksum
      std::vector<Task*> tasks = graphs[i]->getTasks();
      std::vector<Task*>::const_iterator tasks_iter = tasks.begin();
      for (; tasks_iter != tasks.end(); ++tasks_iter) {
        Task* task = *tasks_iter;
        if (task->getType() == Task::Spatial) {
          numSpatialTasks++;
        }
      }
    }

    // Spatial tasks don't count against the global checksum
    checksum -= numSpatialTasks;

    DOUT(g_mpi_dbg, d_myworld->myrank() << " (MPI::Allreduce) Checking checksum of " << checksum);

    int result_checksum;
    MPI::Allreduce(&checksum, &result_checksum, 1, MPI_INT, MPI_MIN, d_myworld->getComm());

    if (checksum != result_checksum) {
      DOUT(g_mpi_dbg, "Failed task checksum comparison! Not all processes are executing the same taskgraph\n"
            << "  Rank-" << d_myworld->myrank() << " of " << d_myworld->size() - 1 << ": has sum " << checksum
            << "  and global is " << result_checksum);
      MPI::Abort(d_myworld->getComm(), 1);
    }
  }
#endif
}


//______________________________________________________________________
//
struct CompareDep {
    bool operator()( DependencyBatch* a,  DependencyBatch* b )
    {
      return a->m_message_tag < b->m_message_tag;
    }
};


//______________________________________________________________________
//
void ThreadedTaskScheduler::post_MPI_recvs( DetailedTask * task
                                          ,  bool          only_old_recvs
                                          ,  int           abort_point
                                          ,  int           iteration
                                          )
{
  RuntimeStats::RecvTimer recv_timer;

  // sort the requires, so in case there is a particle send we receive it with the right message tag
  std::vector<DependencyBatch*> sorted_reqs;
  std::map<DependencyBatch*, DependencyBatch*>::const_iterator iter = task->getRequires().begin();
  for (; iter != task->getRequires().end(); iter++) {
    sorted_reqs.push_back(iter->first);
  }

  CompareDep comparator;
  std::sort(sorted_reqs.begin(), sorted_reqs.end(), comparator);
  std::vector<DependencyBatch*>::iterator sorted_iter = sorted_reqs.begin();

  // Receive any of the foreign requires
  for (; sorted_iter != sorted_reqs.end(); sorted_iter++) {
    DependencyBatch* batch = *sorted_iter;

    // The first thread that calls this on the batch will return true while subsequent
    // threads calling this will block and wait for that first thread to receive the data.
    task->incrementExternalDepCount();
    if (!batch->makeMPIRequest()) {  // Someone else already receiving it
      continue;
    }

    if (only_old_recvs) {
      if (!(batch->m_from_task->getTask()->getSortedOrder() <= abort_point)) {
        continue;
      }
    }

    // Prepare to receive a message
    BatchReceiveHandler* pBatchRecvHandler = new BatchReceiveHandler(batch);
    PackBufferInfo* p_mpibuff = 0;

#ifdef USE_PACKING
    p_mpibuff = new PackBufferInfo();
    PackBufferInfo& mpibuff = *p_mpibuff;
#else
    BufferInfo mpibuff;
#endif

    // Create the MPI type
    for (DetailedDependency* req = batch->m_head; req != 0; req = req->m_next) {

      OnDemandDataWarehouse* dw = dws[req->m_req->mapDataWarehouse()].get_rep();
      if ((req->m_comm_condition == DetailedDependency::FirstIteration && iteration > 0) || (req->m_comm_condition == DetailedDependency::SubsequentIterations
                          && iteration == 0) || (notCopyDataVars_.count(req->m_req->var->getName()) > 0)) {
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->m_to_tasks.front()->getTask()->getType() == Task::Output && !m_output_port->isOutputTimestep() && !m_output_port->isCheckpointTimestep()) {
        continue;
      }

      OnDemandDataWarehouse* posDW;

      // the load balancer is used to determine where data was in the old dw on the prev timestep
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;
      if (!reloc_new_posLabel_) {
        posDW = dws[req->m_req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
          posDW = dws[req->m_req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        } else {
          posDW = dws[req->m_req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
      }

      dw->recvMPI(batch, mpibuff, posDW, req, lb);

      if (!req->isNonDataDependency()) {
        graphs[currentTG_]->getDetailedTasks()->setScrubCount(req->m_req, req->m_matl, req->m_from_patch, dws);
      }
    }

    // Post the receive
    if (mpibuff.count() > 0) {

      ASSERT(batch->m_message_tag > 0);

      void* buf;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      int from = batch->m_from_task->getAssignedResourceIndex();
      ASSERTRANGE(from, 0, d_myworld->size());

      CommPool::iterator iter = m_comm_requests.emplace(t_emplace, REQUEST_RECV, new RecvHandle(p_mpibuff, pBatchRecvHandler));
      t_emplace = iter;

      MPI::Irecv(buf, count, datatype, from, batch->m_message_tag, d_myworld->getComm(), iter->request());

    } else {
      // Nothing really need to be received, but let everyone else know that it has what is needed (nothing).
      batch->received(d_myworld);

#ifdef USE_PACKING
      // otherwise, these will be deleted after it receives and unpacks the data.
      delete p_mpibuff;
      delete pBatchRecvHandler;
#endif

    }
  }  // end for loop over requires
}  // end post_MPI_recvs()


//______________________________________________________________________
//
void ThreadedTaskScheduler::post_MPI_sends( DetailedTask * task, int iteration )
{
  RuntimeStats::SendTimer send_timer;

  int num_sends    = 0;
  int volume_sends = 0;

  // Send data to dependents
  for (DependencyBatch* batch = task->getComputes(); batch != 0; batch = batch->m_comp_next) {

    // Prepare to send a message
#ifdef USE_PACKING
    PackBufferInfo mpibuff;
#else
    BufferInfo mpibuff;
#endif
    // Create the MPI type
    int to = batch->m_to_tasks.front()->getAssignedResourceIndex();
    ASSERTRANGE(to, 0, d_myworld->size());

    for (DetailedDependency* req = batch->m_head; req != 0; req = req->m_next) {

      if ((req->m_comm_condition == DetailedDependency::FirstIteration && iteration > 0) || (req->m_comm_condition == DetailedDependency::SubsequentIterations
          && iteration == 0) || (notCopyDataVars_.count(req->m_req->var->getName()) > 0)) {
        continue;
      }

      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->m_to_tasks.front()->getTask()->getType() == Task::Output && !m_output_port->isOutputTimestep() && !m_output_port->isCheckpointTimestep()) {
        continue;
      }

      OnDemandDataWarehouse* dw = dws[req->m_req->mapDataWarehouse()].get_rep();

      // the load balancer is used to determine where data was in the old dw on the prev timestep -
      // pass it in if the particle data is on the old dw
      const VarLabel* posLabel;
      OnDemandDataWarehouse* posDW;
      LoadBalancer* lb = 0;

      if( !reloc_new_posLabel_) {
        posDW = dws[req->m_req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
      }
      else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
          posDW = dws[req->m_req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        }
        else {
          posDW = dws[req->m_req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
        posLabel = reloc_new_posLabel_;
      }

      dw->sendMPI(batch, posLabel, mpibuff, posDW, req, lb);
    }

    // Post the send
    if (mpibuff.count() > 0) {
      ASSERT(batch->m_message_tag > 0);

      void* buf;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
      mpibuff.pack(d_myworld->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      ++m_num_messages;
      ++num_sends;
      int typeSize;

      MPI::Type_size(datatype, &typeSize);
      m_message_volume += count * typeSize;
      volume_sends += count * typeSize;

      CommPool::iterator iter = m_comm_requests.emplace(t_emplace, REQUEST_SEND, new SendHandle(mpibuff.takeSendlist()));
      t_emplace = iter;

      MPI::Isend(buf, count, datatype, to, batch->m_message_tag, d_myworld->getComm(), iter->request());
    }
  }  // end for (DependencyBatch* batch = task->getComputes())

}  // end post_MPI_sends();


//______________________________________________________________________
//
bool ThreadedTaskScheduler::process_MPI_requests()
{
  RuntimeStats::TestTimer mpi_test_timer;

  if (m_comm_requests.empty()) {
    return false;
  }

  bool result = false;

  auto ready_request = [](CommRequest const& r) { return r.test(); };
  CommPool::iterator iter = m_comm_requests.find_any(t_find, ready_request);
  if (iter) {
    t_find = iter;
    MPI_Status status;
    iter->finishedCommunication(d_myworld, status);
    m_comm_requests.erase(iter);
    result = true;
  }
  return result;
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::run_task( DetailedTask * dtask, int iteration )
{
  // measure per thread exec_time
  RuntimeStats::ExecTimer exec_timer;

  std::vector<DataWarehouseP> plain_old_dws(dws.size());
  for (int i = 0; i < (int)dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
  }

  dtask->doit(d_myworld, dws, plain_old_dws);

  post_MPI_sends(dtask, iteration);

  dtask->done(dws);

  // add my task time to the total time
  if (!d_sharedState->isCopyDataTimestep() && dtask->getTask()->getType() != Task::Output) {
    // add contribution for patchlist
    std::lock_guard<std::mutex> lb_guard(g_lb_mutex);
    SchedulerCommon::getLoadBalancer()->addContribution(dtask, dtask->task_exec_time());
  }

  SchedulerCommon::emitNode(dtask, 0, dtask->task_exec_time(), 0);
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::run_reduction_task( DetailedTask * task )
{
  Timers::Simple simple;
  {
    const Task::Dependency* mod = task->getTask()->getModifies();
    ASSERT(!mod->next);

    OnDemandDataWarehouse* dw = dws[mod->mapDataWarehouse()].get_rep();
    ASSERT(task->getTask()->d_comm >= 0);

    dw->reduceMPI(mod->var, mod->reductionLevel, mod->matls, task->getTask()->d_comm);
  }

  task->done(dws);

  SchedulerCommon::emitNode(task, 0.0, simple().seconds(), 0);
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::emit_time( const char* label, double dt )
{
  m_labels.push_back(label);
  m_times.push_back(dt);
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::copy_restart_flag( int task_graph_num )
{
  if (restartable && task_graph_num == static_cast<int>(graphs.size() - 1)) {
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
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::select_tasks( int iteration, TaskPool::handle & find_handle)
{
  int flag = 0;

  auto find_task = [&](DetailedTask * dtask) {
    flag = 0;
    if (!dtask->isInitiated() &&
         dtask->getTask()->d_phase == m_current_phase.load(std::memory_order_relaxed)) {
      flag = 1;
    }
    else if (dtask->getExternalDepCount() == 0 &&
             dtask->areInternalDependenciesSatisfied() &&
             dtask->isInitiated() &&
             dtask->getTask()->d_phase == m_current_phase.load(std::memory_order_relaxed)) {
      flag = 2;
    }
    return flag > 0;
  };

  // initiate tasks, post MPI recvs
  TaskPool::iterator iter = m_task_pool.find_any(find_handle, find_task);
  if (iter) {
    find_handle = iter;
    DetailedTask * dtask = *iter;
    if (flag == 1) {
      post_MPI_recvs(dtask, m_abort, m_abort_point, iteration);
      dtask->markInitiated();
      dtask->checkExternalDepCount();
    } else if (flag == 2) {
      run_task(dtask, m_current_iteration);
      m_task_pool.erase(iter);
      m_num_tasks_done.fetch_add(1, std::memory_order_relaxed);
      m_phase_tasks_done[dtask->getTask()->d_phase].fetch_add(1, std::memory_order_relaxed);
    }
    iter.clear();
  } else {
    process_MPI_requests();
  }
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::set_runner( TaskRunner * runner, int tid )
{
  Impl::g_runners[tid] = runner;
  std::atomic_thread_fence(std::memory_order_seq_cst);
}


//______________________________________________________________________
//
void ThreadedTaskScheduler::init_threads(ThreadedTaskScheduler * sched, int num_threads )
{
  Impl::init_threads(sched, num_threads);
}


//______________________________________________________________________
//
void TaskRunner::run() const
{
  ThreadedTaskScheduler::TaskPool::handle handle;
  while ( Impl::g_run_tasks ) {
    m_scheduler->select_tasks(m_scheduler->m_current_iteration, handle);
  }
}

