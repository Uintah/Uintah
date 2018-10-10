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

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Components/Schedulers/SendState.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/CommunicationList.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahMPI.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/kokkos_defs.h>

#ifdef UINTAH_ENABLE_KOKKOS
#  include <Kokkos_Core.hpp>
#endif //UINTAH_ENABLE_KOKKOS

#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>

// Pack data into a buffer before sending -- testing to see if this
// works better and avoids certain problems possible when you allow
// tasks to modify data that may have a pending send.
//
// Note, we have found considerable memory savings and appreciable performance gains
// by commenting USE_PACKING out... however, on some machines this will cause a crash,
// but on Titan production runs it had no problems. Alan H./Brad P.,  05/10/2017
#define USE_PACKING


using namespace Uintah;


namespace Uintah {

// These are used externally, keep them visible outside this unit
  Dout g_task_order( "TaskOrder", "MPIScheduler", "task order debug stream", false );
  Dout g_task_dbg(   "TaskDBG"  , "MPIScheduler", "output each task name as it begins/ends", false );
  Dout g_mpi_dbg(    "MPIDBG"   , "MPIScheduler", "MPI debug stream", false );
  Dout g_exec_out(   "ExecOut"  , "MPIScheduler", "exec debug stream", false );

}


namespace {

  Uintah::MasterLock g_lb_mutex{};                // load balancer lock
  Uintah::MasterLock g_recv_mutex{};              // for postMPIRecvs
  
  Uintah::MasterLock g_msg_vol_mutex{};           // to report thread-safe msg volume info
  Uintah::MasterLock g_send_time_mutex{};         // for reporting thread-safe MPI send times
  Uintah::MasterLock g_recv_time_mutex{};         // for reporting thread-safe MPI recv times
  Uintah::MasterLock g_wait_time_mutex{};         // for reporting thread-safe MPI wait times
  
  Dout g_dbg(          "MPIScheduler_DBG"       , "MPIScheduler", "general dbg info for MPIScheduler", false );
  Dout g_send_stats(   "MPISendStats"           , "MPIScheduler", "MPI send statistics, num_sends, send volume", false );
  Dout g_reductions(   "ReductionTasks"         , "MPIScheduler", "rank-0 reports each reduction task", false );
  Dout g_time_out(     "MPIScheduler_TimingsOut", "MPIScheduler", "write MPI timing files: timingstats.avg, timingstats.max", false );
  Dout g_task_level(   "TaskLevel"              , "MPIScheduler", "output task name and each level's beginning patch when done", false );

}


//______________________________________________________________________
//
MPIScheduler::MPIScheduler( const ProcessorGroup * myworld
                          ,       MPIScheduler   * parentScheduler
                          )
  : SchedulerCommon(myworld)
  , m_parent_scheduler{parentScheduler}
{
#ifdef UINTAH_ENABLE_KOKKOS
  Kokkos::initialize();
#endif //UINTAH_ENABLE_KOKKOS

  if (g_time_out) {
    char filename[64];
    if (d_myworld->myRank() == 0) {
      sprintf(filename, "timingstats.avg");
      m_avg_stats.open(filename);
      sprintf(filename, "timingstats.max");
      m_max_stats.open(filename);
    }
  }

  std::string timeStr("seconds");

  mpi_info_.insert( TotalSend  , std::string("TotalSend")  ,    timeStr );
  mpi_info_.insert( TotalRecv  , std::string("TotalRecv")  ,    timeStr );
  mpi_info_.insert( TotalTest  , std::string("TotalTest")  ,    timeStr );
  mpi_info_.insert( TotalWait  , std::string("TotalWait")  ,    timeStr );
  mpi_info_.insert( TotalReduce, std::string("TotalReduce"),    timeStr );
  mpi_info_.insert( TotalTask  , std::string("TotalTask")  ,    timeStr );
}

//______________________________________________________________________
//
MPIScheduler::~MPIScheduler()
{
  if ( (g_time_out) && (d_myworld->myRank() == 0) ) {
    m_avg_stats.close();
    m_max_stats.close();
  }

#ifdef UINTAH_ENABLE_KOKKOS
  Kokkos::finalize();
#endif //UINTAH_ENABLE_KOKKOS

}

//______________________________________________________________________
//
void
MPIScheduler::problemSetup( const ProblemSpecP     & prob_spec
                          , const MaterialManagerP & materialManager
                          )
{
  SchedulerCommon::problemSetup(prob_spec, materialManager);
}

//______________________________________________________________________
//
SchedulerP
MPIScheduler::createSubScheduler()
{
  MPIScheduler * newsched = scinew MPIScheduler( d_myworld, this );

  newsched->setComponents( this );
  newsched->m_materialManager = m_materialManager;
  return newsched;
}

//______________________________________________________________________
//
void
MPIScheduler::verifyChecksum()
{
#if SCI_ASSERTION_LEVEL >= 3

  // Compute a simple checksum to make sure that all processes are trying to
  // execute the same graph.  We should do two things in the future:
  //  - make a flag to turn this off
  //  - make the checksum more sophisticated
  int checksum = 0;
  int numSpatialTasks = 0;
  size_t num_graphs = m_task_graphs.size();
  for (size_t i = 0; i < num_graphs; ++i) {
    checksum += m_task_graphs[i]->getTasks().size();

    // This begins addressing the issue of making the global checksum more sophisticated:
    //   check if any tasks were spatially scheduled - TaskType::Spatial, meaning no computes, requires or modifies
    //     e.g. RMCRT radiometer task, which is not scheduled on all patches
    //          these Spatial tasks won't count toward the global checksum
    std::vector<std::shared_ptr<Task> > tasks = m_task_graphs[i]->getTasks();
    for (auto iter = tasks.begin(); iter != tasks.end(); ++iter) {
      Task* task = iter->get();
      if (task->getType() == Task::Spatial) {
        numSpatialTasks++;
      }
    }
  }

  // Spatial tasks don't count against the global checksum
  checksum -= numSpatialTasks;

  int my_rank = d_myworld->myRank();
  DOUT(g_mpi_dbg, "Rank-" << my_rank << " (Uintah::MPI::Allreduce) Checking checksum of " << checksum);

  int result_checksum;
  Uintah::MPI::Allreduce(&checksum, &result_checksum, 1, MPI_INT, MPI_MIN, d_myworld->getComm());

  if (checksum != result_checksum) {
    std::cerr << "Failed task checksum comparison! Not all processes are executing the same taskgraph\n";
    std::cerr << "  Rank-" << my_rank << " of " << d_myworld->nRanks() - 1 << ": has sum " << checksum << "  and global is "
              << result_checksum << '\n';
    Uintah::MPI::Abort(d_myworld->getComm(), 1);
  }

  DOUT(g_mpi_dbg, "Rank-" << my_rank << " (Uintah::MPI::Allreduce) Check succeeded");

#endif
}

//______________________________________________________________________
//
void MPIScheduler::initiateTask( DetailedTask * dtask
                               , bool           only_old_recvs
                               , int            abort_point
                               , int            iteration
                               )
{
  if (only_old_recvs) {
    return;
  }

  postMPIRecvs(dtask, only_old_recvs, abort_point, iteration);
}

//______________________________________________________________________
//
void
MPIScheduler::initiateReduction( DetailedTask* dtask )
{
  DOUT(g_reductions, "Rank-" << d_myworld->myRank() << " Running Reduction Task: " << dtask->getName());

  Timers::Simple timer;

  timer.start();
  runReductionTask(dtask);
  timer.stop();

  mpi_info_[TotalReduce] += timer().seconds();
}

//______________________________________________________________________
//
void
MPIScheduler::runTask( DetailedTask * dtask
                     , int            iteration
                     )
{
  if (m_tracking_vars_print_location & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(dtask, SchedulerCommon::PRINT_BEFORE_EXEC);
  }
  std::vector<DataWarehouseP> plain_old_dws(m_dws.size());
  size_t num_dws = m_dws.size();
  for (size_t i = 0; i < num_dws; i++) {
    plain_old_dws[i] = m_dws[i].get_rep();
  }

  dtask->doit( d_myworld, m_dws, plain_old_dws );

  if (m_tracking_vars_print_location & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(dtask, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  postMPISends(dtask, iteration);

  dtask->done(m_dws);

  g_lb_mutex.lock();
  {
    // Do the global and local per task monitoring
    sumTaskMonitoringValues( dtask );
    
    double total_task_time = dtask->task_exec_time();
    if (g_exec_out) {
      m_exec_times[dtask->getTask()->getName()] += total_task_time;
    }
    // if I do not have a sub scheduler
    if (!dtask->getTask()->getHasSubScheduler()) {
      //add my task time to the total time
      mpi_info_[TotalTask] += total_task_time;
      if (!m_is_copy_data_timestep && dtask->getTask()->getType() != Task::Output) {
        // add contribution for patchlist
        m_loadBalancer->addContribution(dtask, total_task_time);
      }
    }
  }
  g_lb_mutex.unlock();

  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
  // ---------------------------------------------------------------------------
  // test a pending request
  auto ready_request = [](CommRequest const& r)->bool { return r.test(); };
  CommRequestPool::iterator comm_sends_iter = m_sends.find_any(ready_request);
  if (comm_sends_iter) {
    MPI_Status status;
    comm_sends_iter->finishedCommunication(d_myworld, status);
    m_sends.erase(comm_sends_iter);
  }
  //-----------------------------------

  // Add subscheduler timings to the parent scheduler and reset subscheduler timings
  if (m_parent_scheduler) {
    size_t num_elems = mpi_info_.size();
    for (size_t i = 0; i < num_elems; ++i) {
      m_parent_scheduler->mpi_info_[i] += mpi_info_[i];
    }
    mpi_info_.reset(0);
  }
}  // end runTask()

//______________________________________________________________________
//
void
MPIScheduler::runReductionTask( DetailedTask* dtask )
{
  const Task::Dependency* mod = dtask->getTask()->getModifies();
  ASSERT(!mod->m_next);

  OnDemandDataWarehouse* dw = m_dws[mod->mapDataWarehouse()].get_rep();
  ASSERT(dtask->getTask()->m_comm>=0);
  dw->reduceMPI(mod->m_var, mod->m_reduction_level, mod->m_matls, dtask->getTask()->m_comm);
  dtask->done(m_dws);
}

//______________________________________________________________________
//
void
MPIScheduler::postMPISends( DetailedTask * dtask
                          , int            iteration
                          )
{
  Timers::Simple send_timer;
  send_timer.start();

  int      my_rank = d_myworld->myRank();
  MPI_Comm my_comm = d_myworld->getComm();

  DOUT(g_dbg, "Rank-" << my_rank << " postMPISends - task " << *dtask);

  // Send data to dependents
  for (DependencyBatch* batch = dtask->getComputes(); batch != nullptr; batch = batch->m_comp_next) {

    // Prepare to send a message
#ifdef USE_PACKING
    PackBufferInfo mpibuff;
#else
    BufferInfo mpibuff;
#endif

    // Create the MPI type
    int to = batch->m_to_tasks.front()->getAssignedResourceIndex();
    ASSERTRANGE(to, 0, d_myworld->nRanks());

    for (DetailedDep* req = batch->m_head; req != nullptr; req = req->m_next) {

     if ((req->m_comm_condition == DetailedDep::FirstIteration && iteration > 0) || (req->m_comm_condition == DetailedDep::SubsequentIterations
         && iteration == 0) || (m_no_copy_data_vars.count(req->m_req->m_var->getName()) > 0)) {
       // See comment in DetailedDep about CommCondition
       DOUT(g_dbg, "Rank-" << my_rank << "   Ignoring conditional send for " << *req);
       continue;
     }

     // if we send/recv to an output task, don't send/recv if not an output timestep

     // ARS NOTE: Outputing and Checkpointing may be done out of snyc
     // now. I.e. turned on just before it happens rather than turned
     // on before the task graph execution.  As such, one should also
     // be checking:
     
     // m_application->activeReductionVariable( "outputInterval" );
     // m_application->activeReductionVariable( "checkpointInterval" );
      
     // However, if active the code below would be called regardless
     // if an output or checkpoint time step or not. Not sure that is
     // desired but not sure of the effect of not calling it and doing
     // an out of sync output or checkpoint.
     if (req->m_to_tasks.front()->getTask()->getType() == Task::Output &&
         !m_output->isOutputTimeStep() && !m_output->isCheckpointTimeStep()) {
       DOUT(g_dbg, "Rank-" << my_rank << "   Ignoring non-output-timestep send for " << *req);
       continue;
     }

      OnDemandDataWarehouse* dw = m_dws[req->m_req->mapDataWarehouse()].get_rep();

      DOUT(g_dbg, "Rank-" << my_rank << " --> sending " << *req << ", ghost type: " << "\""
               << Ghost::getGhostTypeName(req->m_req->m_gtype) << "\", " << "num req ghost "
               << Ghost::getGhostTypeName(req->m_req->m_gtype) << ": " << req->m_req->m_num_ghost_cells
               << ", Ghost::direction: " << Ghost::getGhostTypeDir(req->m_req->m_gtype)
               << ", from dw " << dw->getID());

      // the load balancer is used to determine where data was in the
      // old DW on the prev timestep, so pass it in if the particle
      // data is in the old DW
      const VarLabel        * posLabel;
      OnDemandDataWarehouse * posDW;

      if( !m_reloc_new_pos_label && m_parent_scheduler ) {
        posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
        posLabel = m_parent_scheduler->m_reloc_new_pos_label;
      }
      else {
        // on an output task (and only on one) we require particle
        // variables from the NewDW
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
          posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::NewDW)].get_rep();
        }
        else {
          posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::OldDW)].get_rep();
        }
        posLabel = m_reloc_new_pos_label;
      }

      MPIScheduler* top = this;
      while( top->m_parent_scheduler ) {
        top = top->m_parent_scheduler;
      }

      dw->sendMPI( batch, posLabel, mpibuff, posDW, req, m_loadBalancer );
    }

    // Post the send
    if (mpibuff.count() > 0) {
      ASSERT(batch->m_message_tag > 0);
      void* buf = nullptr;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, my_comm);
      mpibuff.pack(my_comm, count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif
      if (!buf) {
        printf("postMPISends() - ERROR, the send MPI buffer is nullptr\n");
        SCI_THROW( InternalError("The send MPI buffer is null", __FILE__, __LINE__) );
      }
      DOUT(g_mpi_dbg, "Rank-" << my_rank << " Posting send for message number " << batch->m_message_tag
                              << " to   rank-" << to << ", length: " << count << " (bytes)");

      m_num_messages++;
      int typeSize;

      Uintah::MPI::Type_size(datatype, &typeSize);

      {
        std::lock_guard<Uintah::MasterLock> msg_vol_lock(g_msg_vol_mutex);
        m_message_volume += count * typeSize;
      }

      //---------------------------------------------------------------------------
      // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
      //---------------------------------------------------------------------------
      CommRequestPool::iterator comm_sends_iter = m_sends.emplace(new SendHandle(mpibuff.takeSendlist()));
      Uintah::MPI::Isend(buf, count, datatype, to, batch->m_message_tag, my_comm, comm_sends_iter->request());
      comm_sends_iter.clear();
      //---------------------------------------------------------------------------

    }
  }  // end for (DependencyBatch* batch = task->getComputes())

  send_timer.stop();

  {
    std::lock_guard<Uintah::MasterLock> send_time_lock(g_send_time_mutex);
    mpi_info_[TotalSend] += send_timer().seconds();
  }

}  // end postMPISends();

//______________________________________________________________________
//
struct CompareDep {
    bool operator()( DependencyBatch* a, DependencyBatch* b )
    {
      return a->m_message_tag < b->m_message_tag;
    }
};

//______________________________________________________________________
//
void MPIScheduler::postMPIRecvs( DetailedTask * dtask
                               , bool           only_old_recvs
                               , int            abort_point
                               , int            iteration
                               )
{
  Timers::Simple recv_timer;
  recv_timer.start();

  int      my_rank = d_myworld->myRank();
  MPI_Comm my_comm = d_myworld->getComm();

  if (g_dbg) {
    DOUT(true, "Rank-" << my_rank << " postMPIRecvs - task " << *dtask);
  }

  if (m_tracking_vars_print_location & SchedulerCommon::PRINT_BEFORE_COMM) {
    printTrackedVars(dtask, SchedulerCommon::PRINT_BEFORE_COMM);
  }

  // sort the requires, so in case there is a particle send we receive it with the right message tag
  std::vector<DependencyBatch*> sorted_reqs;
  std::map<DependencyBatch*, DependencyBatch*>::const_iterator iter = dtask->getRequires().cbegin();
  for (; iter != dtask->getRequires().cend(); ++iter) {
    sorted_reqs.push_back(iter->first);
  }

  CompareDep comparator;
  std::sort(sorted_reqs.begin(), sorted_reqs.end(), comparator);

  // Need this until race condition on foreign variables is resolved - APH, 09/19/17
  std::lock_guard<Uintah::MasterLock> recv_lock(g_recv_mutex);
  {

    // Receive any of the foreign requires
    std::vector<DependencyBatch*>::const_iterator sorted_iter = sorted_reqs.cbegin();
    for (; sorted_iter != sorted_reqs.cend(); ++sorted_iter) {
      DependencyBatch* batch = *sorted_iter;

      dtask->incrementExternalDepCount();

      // The first thread that calls this on the batch will return true while subsequent threads
      // calling this will block and wait for that first thread to receive the data.
      if (!batch->makeMPIRequest()) {
        DOUT(g_dbg, "Someone else already receiving it");
        continue;
      }

      if (only_old_recvs) {
        DOUT(g_dbg, "abort analysis: " << batch->m_from_task->getTask()->getName()
                                       << ", so="
                                       << batch->m_from_task->getTask()->getSortedOrder()
                                       << ", abort_point=" << abort_point);

        if (batch->m_from_task->getTask()->getSortedOrder() <= abort_point) {
          DOUT(g_dbg, "posting MPI recv for pre-abort message " << batch->m_message_tag);
        }
        if (!(batch->m_from_task->getTask()->getSortedOrder() <= abort_point)) {
          continue;
        }
      }

      // Prepare to receive a message
      BatchReceiveHandler* pBatchRecvHandler = scinew BatchReceiveHandler(batch);
      PackBufferInfo* p_mpibuff = nullptr;

#ifdef USE_PACKING
      p_mpibuff = scinew PackBufferInfo();
      PackBufferInfo& mpibuff = *p_mpibuff;
#else
        BufferInfo mpibuff;
#endif

      // Create the MPI type
      for (DetailedDep* req = batch->m_head; req != nullptr; req = req->m_next) {

        OnDemandDataWarehouse* dw = m_dws[req->m_req->mapDataWarehouse()].get_rep();
        if ((req->m_comm_condition == DetailedDep::FirstIteration && iteration > 0)        ||
            (req->m_comm_condition == DetailedDep::SubsequentIterations && iteration == 0) ||
            (m_no_copy_data_vars.count(req->m_req->m_var->getName()) > 0)) {

          // See comment in DetailedDep about CommCondition
          DOUT(g_dbg, "Rank-" << my_rank << "   Ignoring conditional receive for " << *req);
          continue;
        }
        // if we send/recv to an output task, don't send/recv if not an output timestep

        // ARS NOTE: Outputing and Checkpointing may be done out of
        // snyc now. I.e. turned on just before it happens rather than
        // turned on before the task graph execution.  As such, one
        // should also be checking:
        
        // m_application->activeReductionVariable( "outputInterval" );
        // m_application->activeReductionVariable( "checkpointInterval" );
        
        // However, if active the code below would be called regardless
        // if an output or checkpoint time step or not. Not sure that is
        // desired but not sure of the effect of not calling it and doing
        // an out of sync output or checkpoint.
        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output && !m_output->isOutputTimeStep()
            && !m_output->isCheckpointTimeStep()) {
          DOUT(g_dbg, "Rank-" << my_rank << "   Ignoring non-output-timestep receive for " << *req);
          continue;
        }

        DOUT(g_dbg, "Rank-" << my_rank << " <-- receiving " << *req << ", ghost type: " << "\""
                            << Ghost::getGhostTypeName(req->m_req->m_gtype) << "\", " << "num req ghost "
                            << Ghost::getGhostTypeName(req->m_req->m_gtype) << ": " << req->m_req->m_num_ghost_cells
                            << ", Ghost::direction: " << Ghost::getGhostTypeDir(req->m_req->m_gtype) << ", into dw " << dw->getID());

        OnDemandDataWarehouse* posDW;

        // The load balancer is used to determine where data was in
        // the old dw on the prev timestep pass it in if the particle
        // data is on the old dw
        if (!m_reloc_new_pos_label && m_parent_scheduler) {
          posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
        }
        else {
          // On an output task (and only on one) we require particle
          // variables from the NewDW
          if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::NewDW)].get_rep();
          }
          else {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::OldDW)].get_rep();
          }
        }

        MPIScheduler* top = this;
        while (top->m_parent_scheduler) {
          top = top->m_parent_scheduler;
        }

        dw->recvMPI( batch, mpibuff, posDW, req, m_loadBalancer );

        if ( !req->isNonDataDependency() ) {
          m_task_graphs[m_current_task_graph]->getDetailedTasks()->setScrubCount(req->m_req, req->m_matl, req->m_from_patch, m_dws);
        }
      }

      // Post the receive
      if ( mpibuff.count() > 0 ) {

        ASSERT(batch->m_message_tag > 0);
        void* buf = nullptr;
        int count;
        MPI_Datatype datatype;

#ifdef USE_PACKING
        mpibuff.get_type(buf, count, datatype, my_comm);
#else
        mpibuff.get_type(buf, count, datatype);
#endif
        if (!buf) {
          printf("postMPIRecvs() - ERROR, the receive MPI buffer is nullptr\n");
          SCI_THROW( InternalError("The receive MPI buffer is nullptr", __FILE__, __LINE__) );
        }

        int from = batch->m_from_task->getAssignedResourceIndex();
        ASSERTRANGE(from, 0, d_myworld->nRanks());

        DOUT(g_mpi_dbg, "Rank-" << my_rank << " Posting recv for message number "
                                << batch->m_message_tag << " from rank-" << from
                                << ", length: " << count << " (bytes)");

        //---------------------------------------------------------------------------
        // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
        //---------------------------------------------------------------------------
        CommRequestPool::iterator comm_recvs_iter = m_recvs.emplace(new RecvHandle(p_mpibuff, pBatchRecvHandler));
        Uintah::MPI::Irecv(buf, count, datatype, from, batch->m_message_tag, my_comm, comm_recvs_iter->request());
        comm_recvs_iter.clear();
        //---------------------------------------------------------------------------

      }
      else {
        // Nothing really needs to be received, but let everyone else know that it has what is needed (nothing).
        batch->received(d_myworld);

#ifdef USE_PACKING
        // otherwise, these will be deleted after it receives and unpacks the data.
        delete p_mpibuff;
        delete pBatchRecvHandler;
#endif

      }
    }  // end for loop over requires

    recv_timer.stop();

  }


  {
    std::lock_guard<Uintah::MasterLock> recv_time_lock(g_recv_time_mutex);
    mpi_info_[TotalRecv] += recv_timer().seconds();
  }

}  // end postMPIRecvs()

//______________________________________________________________________
//
void MPIScheduler::processMPIRecvs( int test_type )
{
  if (m_recvs.size() == 0u) {
    return;
  }

  Timers::Simple process_recv_timer;
  process_recv_timer.start();

  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
  //---------------------------------------------------------------------------
  auto test_request    = [](CommRequest const& n)->bool { return n.test(); };
  auto wait_request    = [](CommRequest const& n)->bool { return n.wait(); };

  CommRequestPool::iterator comm_iter;

  switch (test_type) {

    case TEST :
    {
      RuntimeStats::TestTimer mpi_test_timer;
      comm_iter = m_recvs.find_any(test_request);
      if (comm_iter) {
        MPI_Status status;
        comm_iter->finishedCommunication(d_myworld, status);
        m_recvs.erase(comm_iter);
      }
      break;
    }

    case WAIT_ONCE :
    {
      RuntimeStats::WaitTimer mpi_wait_timer;
      comm_iter = m_recvs.find_any(wait_request);
      if (comm_iter) {
        MPI_Status status;
        comm_iter->finishedCommunication(d_myworld, status);
        m_recvs.erase(comm_iter);
      }
      break;
    }

    case WAIT_ALL :
    {
      RuntimeStats::WaitTimer mpi_wait_timer;
      while (m_recvs.size() != 0u) {
        comm_iter = m_recvs.find_any(wait_request);
        if (comm_iter) {
          MPI_Status status;
          comm_iter->finishedCommunication(d_myworld, status);
          m_recvs.erase(comm_iter);
        }
      }
      break;
    }

  }  // end switch
  process_recv_timer.stop();

  {
    std::lock_guard<Uintah::MasterLock> wait_time_lock(g_wait_time_mutex);
    mpi_info_[TotalWait] += process_recv_timer().seconds();
  }

}  // end processMPIRecvs()

//______________________________________________________________________
//

void
MPIScheduler::execute( int tgnum     /* = 0 */
                     , int iteration /* = 0 */
                     )
{
  // track total scheduler execution time across timesteps
  m_exec_timer.reset(true);

  RuntimeStats::initialize_timestep(m_task_graphs);

  ASSERTRANGE( tgnum, 0, static_cast<int>(m_task_graphs.size()) );
  TaskGraph* tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;

  // multi TG model - each graph needs have its dwmap reset here (even with the same tgnum)
  if (m_task_graphs.size() > 1) {
    tg->remapTaskDWs(m_dwmap);
  }

  DetailedTasks* dts = tg->getDetailedTasks();

  if (dts == nullptr) {
    proc0cout << "MPIScheduler skipping execute, no tasks" << std::endl;
    return;
  }

  int ntasks = dts->numLocalTasks();

  if( d_runtimeStats )
    (*d_runtimeStats)[NumTasks] += ntasks;
                   
  dts->initializeScrubs(m_dws, m_dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  int my_rank = d_myworld->myRank();

  // This only happens if "-emit_taskgraphs" is passed to sus
  makeTaskGraphDoc( dts, my_rank );

  mpi_info_.reset( 0 );

  DOUT(g_dbg, "Rank-" << my_rank << ", MPI Scheduler executing taskgraph: " << tgnum << ", timestep: " << m_application->getTimeStep()
                      << " with " << dts->numTasks() << " tasks (" << ntasks << " local)");

  if( m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr ) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, m_loadBalancer, m_reloc_new_pos_label, iteration);
  }

  bool abort       = false;
  int abort_point  = 987654;
  int numTasksDone = 0;
  int i            = 0;

  while ( numTasksDone < ntasks ) {

    i++;

    DetailedTask * dtask = dts->getNextInternalReadyTask();

    numTasksDone++;

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

    DOUT(g_task_dbg, "Rank-" << my_rank << " Initiating task:  " << *dtask);

    if ( dtask->getTask()->getType() == Task::Reduction ) {
      if (!abort) {
        initiateReduction( dtask );
      }
    }
    else {
      initiateTask( dtask, abort, abort_point, iteration );
      processMPIRecvs( WAIT_ALL );
      ASSERT( m_recvs.size() == 0u );
      runTask( dtask, iteration );

      DOUT(g_task_dbg, "Rank-" << d_myworld->myRank() << " Completed task:   " << *dtask);
      printTaskLevels( d_myworld, g_task_level, dtask );
    }

    // ARS - FIXME CHECK THE WAREHOUSE
    OnDemandDataWarehouseP dw = m_dws[m_dws.size() - 1];
    if (!abort && dw && dw->abortTimeStep()) {
      // TODO - abort might not work with external queue...
      abort = true;
      abort_point = dtask->getTask()->getSortedOrder();

      DOUT(true,  "Rank-" << d_myworld->myRank()
                          << "  WARNING: Aborting time step after task: "
                          << dtask->getTask()->getName());
    }

  } // end while( numTasksDone < ntasks )


  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
  // ---------------------------------------------------------------------------
  // wait on all pending requests
  auto ready_request = [](CommRequest const& r)->bool { return r.wait(); };
  CommRequestPool::handle find_handle;
  while ( m_sends.size() != 0u ) {
    CommRequestPool::iterator comm_sends_iter;
    if ((comm_sends_iter = m_sends.find_any(find_handle, ready_request))) {
      find_handle = comm_sends_iter;
      m_sends.erase(comm_sends_iter);
    }
    else {
      // TODO - make this a sleep? APH 07/20/16
    }
  }
  //---------------------------------------------------------------------------

  ASSERT(m_sends.size() == 0u);
  ASSERT(m_recvs.size() == 0u);

  finalizeTimestep();

  m_exec_timer.stop();
  
  // compute the net timings
  computeNetRuntimeStats();

  // only do on top-level scheduler
  if ( m_parent_scheduler == nullptr ) {

    // This seems like the best place to collect and save these runtime stats.
    // They are reported in outputTimingStats.
    if( d_runtimeStats ) {
      int numCells = 0, numParticles = 0;
      OnDemandDataWarehouseP dw = m_dws[m_dws.size() - 1];
      const GridP grid(const_cast<Grid*>(dw->getGrid()));
      const PatchSubset* myPatches = m_loadBalancer->getPerProcessorPatchSet(grid)->getSubset(my_rank);
      
      for (auto p = 0; p < myPatches->size(); p++) {
        const Patch* patch = myPatches->get(p);
        IntVector range = patch->getExtraCellHighIndex() - patch->getExtraCellLowIndex();
        numCells += range.x() * range.y() * range.z();
        
        // Go through all materials since getting an MPMMaterial
        // correctly would depend on MPM
        for (unsigned int m = 0; m < m_materialManager->getNumMatls(); m++) {
          if (dw->haveParticleSubset(m, patch)) {
            numParticles += dw->getParticleSubset(m, patch)->numParticles();
          }
        }
      }
      
      (*d_runtimeStats)[NumPatches]   = myPatches->size();
      (*d_runtimeStats)[NumCells]     = numCells;
      (*d_runtimeStats)[NumParticles] = numParticles;
    }    
    
    outputTimingStats( "MPIScheduler" );
  }

  RuntimeStats::report(d_myworld->getComm());

} // end execute()

//______________________________________________________________________
//
void
MPIScheduler::emitTime( const char* label, double dt )
{
   m_labels.push_back(label);
   m_times.push_back(dt);
}

//______________________________________________________________________
//
void
MPIScheduler::outputTimingStats( const char* label )
{
  int      my_rank      = d_myworld->myRank();
  int      my_comm_size = d_myworld->nRanks();
  MPI_Comm my_comm      = d_myworld->getComm();

  // for ExecTimes
  if (g_exec_out) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {
      std::ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", my_comm_size, my_rank);
      fout.open(filename);

      // Report which timesteps TaskExecTime values have been accumulated over
      fout << "Reported values are cumulative over 10 timesteps ("
           << m_application->getTimeStep()-9
           << " through "
           << m_application->getTimeStep()
           << ")" << std::endl;

      for (auto iter = m_exec_times.begin(); iter != m_exec_times.end(); ++iter) {
        fout << std::fixed<< "Rank-" << my_rank << ": TaskExecTime(s): " << iter->second << " Task:" << iter->first << std::endl;
      }
      fout.close();
      m_exec_times.clear();
    }
  }

  // for file-based MPI timings
  if (g_time_out) {

    m_labels.clear();
    m_times.clear();

    double  totalexec = m_exec_timer().seconds();

    if( d_runtimeStats ) {
      emitTime("NumPatches"  , (*d_runtimeStats)[NumPatches]);
      emitTime("NumCells"    , (*d_runtimeStats)[NumCells]);
      emitTime("NumParticles", (*d_runtimeStats)[NumParticles]);
    }
    
    emitTime("Total send time"  , mpi_info_[TotalSend]);
    emitTime("Total recv time"  , mpi_info_[TotalRecv]);
    emitTime("Total test time"  , mpi_info_[TotalTest]);
    emitTime("Total wait time"  , mpi_info_[TotalWait]);
    emitTime("Total reduce time", mpi_info_[TotalReduce]);
    emitTime("Total task time"  , mpi_info_[TotalTask]);
    emitTime("Total comm time"  , mpi_info_[TotalSend] + mpi_info_[TotalRecv] + mpi_info_[TotalTest] + mpi_info_[TotalWait] + mpi_info_[TotalReduce]);

    emitTime("Total execution time"   , totalexec );
    emitTime("Non-comm execution time", totalexec - mpi_info_[TotalSend] - mpi_info_[TotalRecv] - mpi_info_[TotalTest] - mpi_info_[TotalWait] - mpi_info_[TotalReduce]);

    std::vector<double> d_totaltimes(m_times.size());
    std::vector<double> d_maxtimes(m_times.size());
    std::vector<double> d_avgtimes(m_times.size());
    double avgTask = -1, maxTask = -1;
    double avgComm = -1, maxComm = -1;
    double avgCell = -1, maxCell = -1;

    MPI_Comm comm = d_myworld->getComm();
    Uintah::MPI::Reduce(&m_times[0], &d_totaltimes[0], static_cast<int>(m_times.size()), MPI_DOUBLE, MPI_SUM, 0, comm);
    Uintah::MPI::Reduce(&m_times[0], &d_maxtimes[0]  , static_cast<int>(m_times.size()), MPI_DOUBLE, MPI_MAX, 0, comm);

    double total    = 0;
    double avgTotal = 0;
    double maxTotal = 0;
    for (auto i = 0; i < (int)d_totaltimes.size(); i++) {
      d_avgtimes[i] = d_totaltimes[i] / my_comm_size;
      if (strcmp(m_labels[i], "Total task time") == 0) {
        avgTask = d_avgtimes[i];
        maxTask = d_maxtimes[i];
      }
      else if (strcmp(m_labels[i], "Total comm time") == 0) {
        avgComm = d_avgtimes[i];
        maxComm = d_maxtimes[i];
      }
      else if (strncmp(m_labels[i], "Num", 3) == 0) {
        if (strcmp(m_labels[i], "NumCells") == 0) {
          avgCell = d_avgtimes[i];
          maxCell = d_maxtimes[i];
        }
        // these are independent stats - not to be summed
        continue;
      }

      total    += m_times[i];
      avgTotal += d_avgtimes[i];
      maxTotal += d_maxtimes[i];
    }

    // to not duplicate the code
    std::vector<std::ofstream*> files;
    std::vector<std::vector<double>*> data;
    data.push_back(&m_times);

    if (my_rank == 0) {
      files.push_back(&m_avg_stats);
      files.push_back(&m_max_stats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (size_t file = 0; file < files.size(); ++file) {
      std::ofstream& out = *files[file];
      out << "TimeStep " << m_application->getTimeStep() << std::endl;
      for (size_t i = 0; i < (*data[file]).size(); i++) {
        out << label << ": " << m_labels[i] << ": ";
        int len = static_cast<int>(strlen(m_labels[i]) + strlen("MPIScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++)
          out << ' ';
        double percent;
        if (strncmp(m_labels[i], "Num", 3) == 0) {
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i] / d_totaltimes[i] * 100;
        }
        else {
          percent = (*data[file])[i] / total * 100;
        }
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << std::endl << std::endl;
    }

    if (my_rank == 0) {
      std::ostringstream message;
      message << "\n";
      message << "  avg exec: " << std::setw(12) << avgTask << ",   max exec: " << std::setw(12) << maxTask << "    load imbalance (exec)%:        " << std::setw(6) << (1 - avgTask / maxTask) * 100 << "\n";
      message << "  avg comm: " << std::setw(12) << avgComm << ",   max comm: " << std::setw(12) << maxComm << "    load imbalance (comm)%:        " << std::setw(6) << (1 - avgComm / maxComm) * 100 << "\n";
      message << "  avg  vol: " << std::setw(12) << avgCell << ",   max  vol: " << std::setw(12) << maxCell << "    load imbalance (theoretical)%: " << std::setw(6) << (1 - avgCell / maxCell) * 100 << "\n";
      DOUT(g_time_out, message.str());
    }
  } // end g_time_out

  // for MPISendStats
  if (g_send_stats) {
    unsigned int total_messages;
    unsigned int max_messages;
    double       total_volume;
    double       max_volume;

    // do SUM and MAX reduction for m_num_messages and m_message_volume
    Uintah::MPI::Reduce(&m_num_messages  , &total_messages, 1, MPI_UNSIGNED, MPI_SUM, 0, my_comm);
    Uintah::MPI::Reduce(&m_message_volume, &total_volume  , 1, MPI_DOUBLE  , MPI_SUM, 0, my_comm);
    Uintah::MPI::Reduce(&m_num_messages  , &max_messages  , 1, MPI_UNSIGNED, MPI_MAX, 0, my_comm);
    Uintah::MPI::Reduce(&m_message_volume, &max_volume    , 1, MPI_DOUBLE  , MPI_MAX, 0, my_comm);

    if (my_rank == 0) {
      std::ostringstream message;
      message << "MPISendStats: Num Send Messages   (avg): " << std::setw(12) << total_messages / (static_cast<double>(my_comm_size)) << "    (max):" << std::setw(12) << max_messages << "\n";
      message << "MPISendStats: Send Message Volume (avg): " << std::setw(12) << total_volume   / (static_cast<double>(my_comm_size)) << "    (max):" << std::setw(12) << max_volume   << "\n";
      DOUT(g_send_stats, message.str());
    }
  }
}

//______________________________________________________________________
//  Take the various timers and compute the net results
void MPIScheduler::computeNetRuntimeStats()
{
  if( d_runtimeStats ) {
    // don't count output time
    (*d_runtimeStats)[TaskExecTime      ] += mpi_info_[TotalTask] - (*d_runtimeStats)[TotalIOTime];
    (*d_runtimeStats)[TaskLocalCommTime ] += mpi_info_[TotalRecv] + mpi_info_[TotalSend];
    (*d_runtimeStats)[TaskWaitCommTime  ] += mpi_info_[TotalWait];
    (*d_runtimeStats)[TaskReduceCommTime] += mpi_info_[TotalReduce];
  }
}
