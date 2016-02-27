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

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SendState.h>
#include <CCA/Components/Schedulers/CommRecMPI.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Timers/Timers.hpp>
#include <Core/Util/DOUT.hpp>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI


#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif //UINTAH_ENABLE_KOKKOS

#include <chrono>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>

// Pack data into a buffer before sending -- testing to see if this
// works better and avoids certain problems possible when you allow
// tasks to modify data that may have a pending send.
#define USE_PACKING

using namespace Uintah;

DebugStream dbg(           "MPIDBG"        , false );
DebugStream dbgst(         "SendTiming"    , false );
DebugStream timeout(       "MPITimes"      , false );
DebugStream reductionout(  "ReductionTasks", false );
DebugStream taskorder(     "TaskOrder"     , false );
DebugStream waitout(       "WaitTimes"     , false );
DebugStream execout(       "ExecTimes"     , false );
DebugStream taskdbg(       "TaskDBG"       , false );
DebugStream taskLevel_dbg( "TaskLevel"     , false );
DebugStream mpidbg(        "MPIDBG"        , false );

std::map<std::string, std::atomic<uint64_t> > waittimes;
std::map<std::string, std::atomic<uint64_t> > exectimes;

namespace {

thread_local SendCommList::handle t_send_emplace;
thread_local SendCommList::handle t_send_find;

thread_local RecvCommList::handle t_recv_emplace;
thread_local RecvCommList::handle t_recv_find;

}

namespace {

std::mutex  s_cout_mutex;
std::mutex  s_cerr_mutex;

}

//______________________________________________________________________
//
MPIScheduler::MPIScheduler( const ProcessorGroup * myworld
                          , const Output         * oport
                          ,       MPIScheduler   * parentScheduler
                          )
  : SchedulerCommon(myworld, oport)
  ,  parentScheduler_{ parentScheduler }
  ,  log{ myworld, oport }
  ,  oport_{ oport }
  ,  numMessages_{ 0 }
  ,  messageVolume_{ 0 }
{
#ifdef UINTAH_ENABLE_KOKKOS
  Kokkos::initialize();
#endif //UINTAH_ENABLE_KOKKOS

  m_last_exec_timer.reset();

  reloc_new_posLabel_ = 0;

  // detailed MPI information, written to file per rank
  if (timeout.active()) {
    char filename[64];
    sprintf(filename, "timingStats.%d", d_myworld->myrank());
    timingStats.open(filename);
    if (d_myworld->myrank() == 0) {
      sprintf(filename, "timingStats.avg");
      avgStats.open(filename);
      sprintf(filename, "timingStats.max");
      maxStats.open(filename);
    }
  }

  std::string timeStr("seconds");

  mpi_info_.insert( TotalReduce,    std::string("TotalReduce"),    timeStr, 0 );
  mpi_info_.insert( TotalSend,      std::string("TotalSend"),      timeStr, 0 );
  mpi_info_.insert( TotalRecv,      std::string("TotalRecv"),      timeStr, 0 );
  mpi_info_.insert( TotalTask,      std::string("TotalTask"),      timeStr, 0 );
  mpi_info_.insert( TotalReduceMPI, std::string("TotalReduceMPI"), timeStr, 0 );
  mpi_info_.insert( TotalSendMPI,   std::string("TotalSendMPI"),   timeStr, 0 );
  mpi_info_.insert( TotalRecvMPI,   std::string("TotalRecvMPI"),   timeStr, 0 );
  mpi_info_.insert( TotalTestMPI,   std::string("TotalTestMPI"),   timeStr, 0 );
  mpi_info_.insert( TotalWaitMPI,   std::string("TotalWaitMPI"),   timeStr, 0 );
  mpi_info_.validate( MAX_TIMING_STATS );
}

//______________________________________________________________________
//
void
MPIScheduler::problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state )
{
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}

//______________________________________________________________________
//
MPIScheduler::~MPIScheduler()
{
  // detailed MPI information, written to file per rank
  if (timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      avgStats.close();
      maxStats.close();
    }
  }
#ifdef UINTAH_ENABLE_KOKKOS
  Kokkos::finalize();
#endif //UINTAH_ENABLE_KOKKOS
}

//______________________________________________________________________
//
SchedulerP
MPIScheduler::createSubScheduler()
{
  MPIScheduler* newsched = scinew MPIScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState = d_sharedState;
  return newsched;
}

//______________________________________________________________________
//
void MPIScheduler::verifyChecksum()
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

    if (mpidbg.active()) {
      s_cout_mutex.lock();
      mpidbg << d_myworld->myrank() << " (MPI_Allreduce) Checking checksum of " << checksum << '\n';
      s_cout_mutex.unlock();
    }

    int result_checksum;
    MPI_Allreduce(&checksum, &result_checksum, 1, MPI_INT, MPI_MIN, d_myworld->getComm());

    if (checksum != result_checksum) {
      std::cerr << "Failed task checksum comparison! Not all processes are executing the same taskgraph\n";
      std::cerr << "  Rank-" << d_myworld->myrank() << " of " << d_myworld->size() - 1 << ": has sum " << checksum
                << "  and global is " << result_checksum << '\n';
      MPI_Abort(d_myworld->getComm(), 1);
    }

    if (mpidbg.active()) {
      s_cout_mutex.lock();
      mpidbg << d_myworld->myrank() << " (MPI_Allreduce) Check succeeded\n";
      s_cout_mutex.unlock();
    }
  }
#endif
}

//______________________________________________________________________
//
void MPIScheduler::initiateTask( DetailedTask * task
                               , bool           only_old_recvs
                               , int            abort_point
                               , int            iteration
                               )
{
  postMPIRecvs(task, only_old_recvs, abort_point, iteration);
}

//______________________________________________________________________
//
void
MPIScheduler::initiateReduction( DetailedTask* task )
{
  Timers::Simple simple;
  {
    Timers::ThreadTrip< TotalReduceTag > tr;
    runReductionTask(task);
  }
  emitNode(task, 0.0, simple.seconds(), 0);
}

//______________________________________________________________________
//
void
MPIScheduler::runTask( DetailedTask* task,
                       int           iteration,
                       int           thread_id /*=0*/ )
{
  if (waitout.active()) {
    waittimes[task->getTask()->getName()].fetch_add( Timers::ThreadTrip< TotalWaitMPITag >::nanoseconds(), std::memory_order_relaxed );
  }

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }

  std::vector<DataWarehouseP> plain_old_dws(dws.size());
  for (int i = 0; i < (int)dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
  }


  m_task_exec_timer.reset();
  double task_start = m_task_exec_timer.seconds();
  {
    Timers::ThreadTrip< TotalTaskTag > task_timer;
    task->doit(d_myworld, dws, plain_old_dws);
  }
  uint64_t total_task_time = m_task_exec_timer.nanoseconds();


  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  dlbLock.lock();
  {
    if (execout.active()) {
      exectimes[task->getTask()->getName()].fetch_add( total_task_time, std::memory_order_relaxed );
    }

    // if I do not have a sub scheduler
    if (!task->getTask()->getHasSubScheduler()) {
      //add my task time to the total time
      if (!d_sharedState->isCopyDataTimestep() && task->getTask()->getType() != Task::Output) {
        //add contribution for patchlist
        getLoadBalancer()->addContribution(task, total_task_time * 1.0e-9);
      }
    }
  }
  dlbLock.unlock();

  postMPISends(task, iteration, thread_id);

  task->done(dws);  // should this be part of task execution time? - APH 09/16/15

  {
    Timers::ThreadTrip< TotalTestMPITag > test_mpi_timer;
    auto ready_request = [](SendCommNode const& n)->bool { return n.test(); };
    SendCommList::iterator iter = m_send_list.find_any(t_send_find, ready_request);
    if (iter) {
      t_send_find = iter;
      m_send_list.erase(iter);
    }
  }


  // Add subscheduler timings to the parent scheduler and reset subscheduler timings
  if (parentScheduler_) {
    for (size_t i = 0; i < mpi_info_.size(); ++i) {
      MPIScheduler::TimingStat e = (MPIScheduler::TimingStat)i;
      parentScheduler_->mpi_info_[e] += mpi_info_[e];
    }
    mpi_info_.reset(0);
  }

  emitNode(task, task_start, total_task_time * 1.0e-9, 0);

}  // end runTask()

//______________________________________________________________________
//
void
MPIScheduler::runReductionTask( DetailedTask* task )
{
  const Task::Dependency* mod = task->getTask()->getModifies();
  ASSERT(!mod->next);

  OnDemandDataWarehouse* dw = dws[mod->mapDataWarehouse()].get_rep();
  ASSERT(task->getTask()->d_comm>=0);
  dw->reduceMPI(mod->var, mod->reductionLevel, mod->matls, task->getTask()->d_comm);
  task->done(dws);
}

//______________________________________________________________________
//
void
MPIScheduler::postMPISends( DetailedTask* task,
                            int           iteration,
                            int           thread_id  /*=0*/ )
{
  Timers::ThreadTrip< TotalSendTag > send_timer;

  bool dbg_active = dbg.active();

  int me = d_myworld->myrank();
  if (dbg_active) {
    s_cerr_mutex.lock();
    dbg << "Rank-" << me << " postMPISends - task " << *task << '\n';
    s_cerr_mutex.unlock();
  }

  int numSend = 0;
  int volSend = 0;

  // Send data to dependents
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

    std::ostringstream ostr;
    ostr.clear();

    for (DetailedDep* req = batch->head; req != 0; req = req->next) {

      ostr << *req << ' '; // for CommRecMPI::add()

      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || (req->condition == DetailedDep::SubsequentIterations
          && iteration == 0) || (notCopyDataVars_.count(req->req->var->getName()) > 0)) {
        // See comment in DetailedDep about CommCondition
        if (dbg_active) {
          s_cerr_mutex.lock();
          dbg << "Rank-" << me << "   Ignoring conditional send for " << *req << "\n";
          s_cerr_mutex.unlock();
        }
        continue;
      }

      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep()
          && !oport_->isCheckpointTimestep()) {
        if (dbg_active) {
          s_cerr_mutex.lock();
          dbg << "Rank-" << me << "   Ignoring non-output-timestep send for " << *req << "\n";
          s_cerr_mutex.unlock();
        }
        continue;
      }

      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      if (dbg_active) {
        s_cerr_mutex.lock();
        {
          dbg << "Rank-" << me << " --> sending " << *req << ", ghost type: " << "\""
              << Ghost::getGhostTypeName(req->req->gtype) << "\", " << "num req ghost "
              << Ghost::getGhostTypeName(req->req->gtype) << ": " << req->req->numGhostCells
              << ", Ghost::direction: " << Ghost::getGhostTypeDir(req->req->gtype)
              << ", from dw " << dw->getID() << '\n';
        }
        s_cerr_mutex.unlock();
      }

      // the load balancer is used to determine where data was in the old dw on the prev timestep -
      // pass it in if the particle data is on the old dw
      const VarLabel* posLabel;
      OnDemandDataWarehouse* posDW;
      LoadBalancer* lb = 0;

      if( !reloc_new_posLabel_ && parentScheduler_ ) {
        posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
        posLabel = parentScheduler_->reloc_new_posLabel_;
      }
      else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output) {
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        }
        else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
        posLabel = reloc_new_posLabel_;
      }

      MPIScheduler* top = this;
      while( top->parentScheduler_ ) {
        top = top->parentScheduler_;
      }

      dw->sendMPI(batch, posLabel, mpibuff, posDW, req, lb);
    }

    // Post the send
    if (mpibuff.count() > 0) {
      ASSERT(batch->messageTag > 0);

      Timers::ThreadTrip< TotalSendMPITag > mpi_send_timer;

      void* buf;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
      mpibuff.pack(d_myworld->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      ++numMessages_;
      ++numSend;
      int typeSize;

      MPI_Type_size(datatype, &typeSize);
      messageVolume_ += count * typeSize;
      volSend += count * typeSize;

      SendCommList::iterator iter = m_send_list.emplace(t_send_emplace, mpibuff.takeSendlist());
      t_send_emplace = iter;

      MPI_Isend(buf, count, datatype, to, batch->messageTag, d_myworld->getComm(), iter->request());
    }
  }  // end for (DependencyBatch* batch = task->getComputes())

}  // end postMPISends();


//______________________________________________________________________
//
struct CompareDep {
    bool operator()( DependencyBatch* a,
                     DependencyBatch* b )
    {
      return a->messageTag < b->messageTag;
    }
};

//______________________________________________________________________
//
void MPIScheduler::postMPIRecvs( DetailedTask* task,
                                 bool          only_old_recvs,
                                 int           abort_point,
                                 int           iteration )
{
  Timers::ThreadTrip< TotalRecvTag > recv_timer;

  bool dbg_active = dbg.active();
  if (dbg_active) {
    s_cerr_mutex.lock();
    dbg << "Rank-" << d_myworld->myrank() << " postMPIRecvs - task " << *task << '\n';
    s_cerr_mutex.unlock();
  }

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_COMM) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_COMM);
  }

  // sort the requires, so in case there is a particle send we receive it with
  // the right message tag

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

    // The first thread that calls this on the batch will return true
    // while subsequent threads calling this will block and wait for
    // that first thread to receive the data.

    task->incrementExternalDepCount();
    if (!batch->makeMPIRequest()) {
      if (dbg_active) {
        s_cerr_mutex.lock();
        dbg << "Someone else already receiving it\n";
        s_cerr_mutex.unlock();
      }
      continue;
    }

    if (only_old_recvs) {
      if (dbg_active) {
        dbg << "abort analysis: " << batch->fromTask->getTask()->getName() << ", so="
            << batch->fromTask->getTask()->getSortedOrder() << ", abort_point=" << abort_point << '\n';
        if (batch->fromTask->getTask()->getSortedOrder() <= abort_point)
          dbg << "posting MPI recv for pre-abort message " << batch->messageTag << '\n';
      }
      if (!(batch->fromTask->getTask()->getSortedOrder() <= abort_point)) {
        continue;
      }
    }

    // Prepare to receive a message
    BatchReceiveHandler* pBatchRecvHandler = scinew BatchReceiveHandler(batch);
    PackBufferInfo* p_mpibuff = 0;

#ifdef USE_PACKING
    p_mpibuff = scinew PackBufferInfo();
    PackBufferInfo& mpibuff = *p_mpibuff;
#else
    BufferInfo mpibuff;
#endif

    std::ostringstream ostr;
    ostr.clear();

    // Create the MPI type
    for (DetailedDep* req = batch->head; req != 0; req = req->next) {

      ostr << *req << ' ';  // for CommRecMPI::add()

      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || (req->condition == DetailedDep::SubsequentIterations
          && iteration == 0)
          || (notCopyDataVars_.count(req->req->var->getName()) > 0)) {

        // See comment in DetailedDep about CommCondition
        if (dbg_active) {
          s_cerr_mutex.lock();
          dbg << "Rank-" << d_myworld->myrank() << "   Ignoring conditional receive for " << *req << std::endl;
          s_cerr_mutex.unlock();
        }
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep()
          && !oport_->isCheckpointTimestep()) {
        s_cerr_mutex.lock();
        dbg << "Rank-" << d_myworld->myrank() << "   Ignoring non-output-timestep receive for " << *req << std::endl;
        s_cerr_mutex.unlock();
        continue;
      }
      if (dbg_active) {
        s_cerr_mutex.lock();
        {
          dbg << "Rank-" << d_myworld->myrank() << " <-- receiving " << *req << ", ghost type: " << "\""
              << Ghost::getGhostTypeName(req->req->gtype) << "\", " << "num req ghost " << Ghost::getGhostTypeName(req->req->gtype)
              << ": " << req->req->numGhostCells << ", Ghost::direction: " << Ghost::getGhostTypeDir(req->req->gtype)
              << ", into dw " << dw->getID() << '\n';
        }
        s_cerr_mutex.unlock();
      }

      OnDemandDataWarehouse* posDW;

      // the load balancer is used to determine where data was in the old dw on the prev timestep
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;
      if (!reloc_new_posLabel_ && parentScheduler_) {
        posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output) {
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        } else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
      }

      MPIScheduler* top = this;
      while (top->parentScheduler_) {
        top = top->parentScheduler_;
      }

      dw->recvMPI(batch, mpibuff, posDW, req, lb);

      if (!req->isNonDataDependency()) {
        graphs[currentTG_]->getDetailedTasks()->setScrubCount(req->req, req->matl, req->fromPatch, dws);
      }
    }

    // Post the receive
    if (mpibuff.count() > 0) {

      ASSERT(batch->messageTag > 0);

      Timers::ThreadTrip< TotalRecvMPITag > mpi_recv_timer;

      void* buf;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      int from = batch->fromTask->getAssignedResourceIndex();
      ASSERTRANGE(from, 0, d_myworld->size());

      RecvCommList::iterator iter = m_recv_list.emplace(t_recv_emplace, p_mpibuff, pBatchRecvHandler);
      t_recv_emplace = iter;

      MPI_Irecv(buf, count, datatype, from, batch->messageTag, d_myworld->getComm(), iter->request());

    } else {
      // Nothing really need to be received, but let everyone else know
      // that it has what is needed (nothing).
      batch->received(d_myworld);
#ifdef USE_PACKING
      // otherwise, these will be deleted after it receives and unpacks the data.
      delete p_mpibuff;
      delete pBatchRecvHandler;
#endif

    }
  }  // end for loop over requires
}  // end postMPIRecvs()

//______________________________________________________________________
//
void MPIScheduler::processMPIRecvs( int how_much )
{
  if (m_recv_list.empty()) {
    return;
  }

  Timers::ThreadTrip < TotalWaitMPITag > mpi_wait_timer;

  auto ready_request = [](RecvCommNode const& n)->bool {return n.test();};
  auto finished_request = [](RecvCommNode const& n)->bool {return n.wait();};

  switch (how_much) {
    case TEST : {
      RecvCommList::iterator iter = m_recv_list.find_any(t_recv_find, ready_request);
      if (iter) {
        t_recv_find = iter;
        MPI_Status status;
        iter->finishedCommunication(d_myworld, status);
        m_recv_list.erase(iter);
      }
      break;
    }
    case WAIT_ONCE : {
      RecvCommList::iterator iter = m_recv_list.find_any(t_recv_find, finished_request);
      if (iter) {
        t_recv_find = iter;
        MPI_Status status;
        iter->finishedCommunication(d_myworld, status);
        m_recv_list.erase(iter);
      }
      break;
    }
    case WAIT_ALL : {
      while (!m_recv_list.empty()) {
        RecvCommList::iterator iter = m_recv_list.find_any(t_recv_find, finished_request);
        if (iter) {
          t_recv_find = iter;
          MPI_Status status;
          iter->finishedCommunication(d_myworld, status);
          m_recv_list.erase(iter);
        }
      }
      break;
    }
  }  // end switch

}  // end processMPIRecvs()

//______________________________________________________________________
//
void MPIScheduler::execute( int tgnum /* = 0 */, int iteration /* = 0 */ )
{
  ASSERTRANGE(tgnum, 0, (int )graphs.size());
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
      s_cerr_mutex.lock();
      std::cerr << "MPIScheduler skipping execute, no tasks\n";
      s_cerr_mutex.unlock();
    }
    return;
  }

  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  int me = d_myworld->myrank();
  makeTaskGraphDoc(dts, me);

  mpi_info_.reset( 0 );

  int numTasksDone = 0;

  if (dbg.active()) {
    s_cerr_mutex.lock();
    dbg << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)\n";
    s_cerr_mutex.unlock();
  }

  bool abort = false;
  int abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  int i = 0;
  while (numTasksDone < ntasks) {
    i++;

    //
    // The following checkMemoryUse() is commented out to allow for
    // maintaining the same functionality as before this commit...
    // In other words, so that memory highwater checking is only done
    // at the end of a timestep, and not between tasks... Once the
    // RT settles down we will uncomment this section and then
    // memory use checks will occur before every task.
    //
    // Note, the results (memuse, highwater, maxMemUse) from the following
    // checkMemoryUse call are not used... the call, however, records
    // the maxMemUse for future reference, and that is why we are calling
    // it.
    //
    //unsigned long memuse, highwater, maxMemUse;
    //checkMemoryUse( memuse, highwater, maxMemUse );

    DetailedTask * task = dts->getNextInternalReadyTask();

    numTasksDone++;

    if (taskorder.active()) {
      taskorder << d_myworld->myrank() << " Running task static order: " << task->getStaticOrder() << " , scheduled order: "
                << numTasksDone << std::endl;
    }

    if (taskdbg.active()) {
      taskdbg << me << " Initiating task:  \t";
      printTask(taskdbg, task);
      taskdbg << '\n';
    }

    if (task->getTask()->getType() == Task::Reduction) {
      if (!abort) {
        initiateReduction(task);
      }
    }
    else {
      initiateTask(task, abort, abort_point, iteration);
      processMPIRecvs(WAIT_ALL);
      ASSERT(m_recv_list.empty());
      runTask(task, iteration);

      if (taskdbg.active()) {
        taskdbg << d_myworld->myrank() << " Completed task:  \t";
        printTask(taskdbg, task);
        taskdbg << '\n';
        printTaskLevels(d_myworld, taskLevel_dbg, task);
      }
    }

    if(!abort && dws[dws.size()-1] && dws[dws.size()-1]->timestepAborted()){
      abort = true;
      abort_point = task->getTask()->getSortedOrder();
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
    }
  } // end while( numTasksDone < ntasks )

  // MUST call before computeNetRunTimeStats
  emitNetMPIStats();

  if( !parentScheduler_ ) { // If this scheduler is the root scheduler...
    computeNetRunTimeStats(d_sharedState->d_runTimeStats);
  }

  auto ready_request = [](SendCommNode const& n)->bool { return n.wait(); };
  while (!m_send_list.empty()) {
    SendCommList::iterator iter = m_send_list.find_any(t_send_find, ready_request);
    if (iter) {
      t_send_find = iter;
      m_send_list.erase(iter);
    }
  }

  ASSERT(m_send_list.empty());

  // Copy the restart flag to all processors
  reduceRestartFlag(tgnum);

  finalizeTimestep();
  log.finishTimestep();

  if ( !parentScheduler_ && (execout.active() || timeout.active() || waitout.active()) ) {  // only do on toplevel scheduler
    outputTimingStats("MPIScheduler");
  }

  if (dbg.active()) {
    s_cout_mutex.lock();
    dbg << me << " MPIScheduler finished\n";
    s_cout_mutex.unlock();
  }
}

//______________________________________________________________________
//
void
MPIScheduler::emitTime( const char* label )
{
  double elapsed = m_last_exec_timer.seconds();
  m_last_exec_timer.reset();
  emitTime(label, elapsed);
}

//______________________________________________________________________
//
void
MPIScheduler::emitTime( const char* label, double dt )
{
  d_labels.push_back(label);
  d_times.push_back(dt);
}

//______________________________________________________________________
//
void
MPIScheduler::emitNetMPIStats()
{
  mpi_info_[TotalWaitMPI]    = Timers::ThreadTrip< TotalWaitMPITag >::seconds();
  mpi_info_[TotalReduce]     = Timers::ThreadTrip< TotalReduceTag >::seconds();
  mpi_info_[TotalReduceMPI]  = Timers::ThreadTrip< TotalReduceTag >::seconds();
  mpi_info_[TotalSend]       = Timers::ThreadTrip< TotalSendTag >::seconds();
  mpi_info_[TotalSendMPI]    = Timers::ThreadTrip< TotalSendMPITag >::seconds();
  mpi_info_[TotalRecv]       = Timers::ThreadTrip< TotalRecvTag >::seconds();
  mpi_info_[TotalRecvMPI]    = Timers::ThreadTrip< TotalRecvMPITag >::seconds();
  mpi_info_[TotalTask]       = Timers::ThreadTrip< TotalTaskTag >::seconds();
  mpi_info_[TotalTestMPI]    = Timers::ThreadTrip< TotalTestMPITag >::seconds();

  Timers::ThreadTrip< TotalWaitMPITag >::reset();
  Timers::ThreadTrip< TotalReduceTag >::reset();
  Timers::ThreadTrip< TotalReduceTag >::reset();
  Timers::ThreadTrip< TotalSendTag >::reset();
  Timers::ThreadTrip< TotalSendMPITag >::reset();
  Timers::ThreadTrip< TotalRecvTag >::reset();
  Timers::ThreadTrip< TotalRecvMPITag >::reset();
  Timers::ThreadTrip< TotalTaskTag >::reset();
  Timers::ThreadTrip< TotalTestMPITag >::reset();


  if (timeout.active()) {

    d_labels.clear();
    d_times.clear();


    emitTime("Total task time"      , mpi_info_[TotalTask]);
    emitTime("MPI Send time"        , mpi_info_[TotalSendMPI]);
    emitTime("MPI Recv time"        , mpi_info_[TotalRecvMPI]);
    emitTime("MPI TestSome time"    , mpi_info_[TotalTestMPI]);
    emitTime("MPI Wait time"        , mpi_info_[TotalWaitMPI]);
    emitTime("MPI reduce time"      , mpi_info_[TotalReduceMPI]);
    emitTime("Total reduction time" , mpi_info_[TotalReduce] - mpi_info_[TotalReduceMPI]);
    emitTime("Total send time"      , mpi_info_[TotalSend]   - mpi_info_[TotalSendMPI] - mpi_info_[TotalTestMPI]);
    emitTime("Total recv time"      , mpi_info_[TotalRecv]   - mpi_info_[TotalRecvMPI] - mpi_info_[TotalWaitMPI]);
    emitTime("Total comm time"      , mpi_info_[TotalRecv]   + mpi_info_[TotalSend]    + mpi_info_[TotalReduce]);

    double totalexec = m_last_exec_timer.seconds();
    m_last_exec_timer.reset();

    emitTime("Other execution time", totalexec - mpi_info_[TotalSend] - mpi_info_[TotalRecv] - mpi_info_[TotalTask] - mpi_info_[TotalReduce]);
  }
}

//______________________________________________________________________
//
void
MPIScheduler::reduceRestartFlag( int task_graph_num )
{
  if (restartable && task_graph_num == static_cast<int>(graphs.size() - 1)) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size() - 1]->timestepRestarted();
    int netrestart;

    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR, d_myworld->getComm());

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
void
MPIScheduler::outputTimingStats( const char* label )
{
  if (timeout.active()) {
    // add number of cells, patches, and particles
    int numCells = 0, numParticles = 0;
    OnDemandDataWarehouseP dw = dws[dws.size() - 1];
    const GridP grid(const_cast<Grid*>(dw->getGrid()));
    const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(grid)->getSubset(d_myworld->myrank());
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

    emitTime("NumPatches", myPatches->size());
    emitTime("NumCells", numCells);
    emitTime("NumParticles", numParticles);
    std::vector<double> d_totaltimes(d_times.size());
    std::vector<double> d_maxtimes(d_times.size());
    std::vector<double> d_avgtimes(d_times.size());
    double avgTask = -1, maxTask = -1;
    double avgComm = -1, maxComm = -1;
    double avgCell = -1, maxCell = -1;

    MPI_Comm comm = d_myworld->getComm();
    MPI_Reduce(&d_times[0], &d_totaltimes[0], static_cast<int>(d_times.size()), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&d_times[0], &d_maxtimes[0], static_cast<int>(d_times.size()), MPI_DOUBLE, MPI_MAX, 0, comm);

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
    std::vector<std::ofstream*> files;
    std::vector<std::vector<double>*> data;
    files.push_back(&timingStats);
    data.push_back(&d_times);

    int me = d_myworld->myrank();

    if (me == 0) {
      files.push_back(&avgStats);
      files.push_back(&maxStats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (unsigned file = 0; file < files.size(); file++) {
      std::ofstream& out = *files[file];
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep() << std::endl;
      for (int i = 0; i < (int)(*data[file]).size(); i++) {
        out << label << ": " << d_labels[i] << ": ";
        int len = static_cast<int>(strlen(d_labels[i]) + strlen("MPIScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++)
          out << ' ';
        double percent;
        if (strncmp(d_labels[i], "Num", 3) == 0) {
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i] / d_totaltimes[i] * 100;
        } else {
          percent = (*data[file])[i] / total * 100;
        }
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << std::endl << std::endl;
    }

    if (me == 0) {
      timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1 - avgTask / maxTask) * 100
              << " load imbalance (exec)%\n";
      timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1 - avgComm / maxComm) * 100
              << " load imbalance (comm)%\n";
      timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1 - avgCell / maxCell) * 100
              << " load imbalance (theoretical)%\n";
    }
  }

  m_last_exec_timer.reset();

  if (execout.active()) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {
      std::ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);

      // Report which timesteps TaskExecTime values have been accumulated over
      fout << "Reported values are cumulative over 10 timesteps (" << d_sharedState->getCurrentTopLevelTimeStep() - 9 << " through "
           << d_sharedState->getCurrentTopLevelTimeStep() << ")" << std::endl;

      for (auto const& p : exectimes) {
        fout << std::fixed << d_myworld->myrank() << ": TaskExecTime(s): " << p.second * 1.0e-9 << " Task:" << p.first << std::endl;
      }

      fout.close();
      exectimes.clear();
    }
  }

  if (waitout.active()) {
    static int count = 0;

    //TODO reduce and print to screen

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {

      if (d_myworld->myrank() == 0 || d_myworld->myrank() == d_myworld->size() / 2
          || d_myworld->myrank() == d_myworld->size() - 1) {

        std::ofstream wout;
        char fname[100];
        sprintf(fname, "waittimes.%d.%d", d_myworld->size(), d_myworld->myrank());
        wout.open(fname);

        for (auto const& p : waittimes) {
          wout << std::fixed << d_myworld->myrank() << ":   TaskWaitTime(TO): " << p.second * 1.0e-9 << " Task:" << p.first
               << std::endl;
        }

        for (auto const& p : DependencyBatch::waittimes) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << p.second *1.0e-9 << " Task:" << p.first
               << std::endl;

        }

        wout.close();
      }

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }
}

//______________________________________________________________________
//  Take the various timers and compute the net results
void MPIScheduler::computeNetRunTimeStats(InfoMapper< SimulationState::RunTimeStat, double >& runTimeStats)
{
    runTimeStats[SimulationState::TaskExecTime]       += mpi_info_[TotalTask] - runTimeStats[SimulationState::OutputFileIOTime]  // don't count output time or bytes
                                                                              - runTimeStats[SimulationState::OutputFileIORate];

    runTimeStats[SimulationState::TaskLocalCommTime]  += mpi_info_[TotalRecv] + mpi_info_[TotalSend];
    runTimeStats[SimulationState::TaskWaitCommTime]   += mpi_info_[TotalWaitMPI];
    runTimeStats[SimulationState::TaskGlobalCommTime] += mpi_info_[TotalReduce];
}
