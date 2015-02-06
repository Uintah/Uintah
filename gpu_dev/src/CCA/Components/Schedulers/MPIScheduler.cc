/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SendState.h>
#include <CCA/Components/Schedulers/CommRecMPI.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Vampir.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

#include <sstream>
#include <iomanip>
#include <map>
#include <cstring>

// Pack data into a buffer before sending -- testing to see if this
// works better and avoids certain problems possible when you allow
// tasks to modify data that may have a pending send.
#define USE_PACKING

using namespace Uintah;
using namespace SCIRun;

// Used to sync cout/cerr so it is readable when output by multiple threads
extern SCIRun::Mutex coutLock;
extern SCIRun::Mutex cerrLock;

static DebugStream dbg(          "MPIScheduler_DBG",     false );
static DebugStream dbgst(        "SendTiming",           false );
static DebugStream timeout(      "MPIScheduler.timings", false );
static DebugStream reductionout( "ReductionTasks",       false );

DebugStream taskorder(     "TaskOrder", false );
DebugStream waitout(       "WaitTimes", false );
DebugStream execout(       "ExecTimes", false );
DebugStream taskdbg(       "TaskDBG",   false );
DebugStream taskLevel_dbg( "TaskLevel", false );
DebugStream mpidbg(        "MPIDBG",    false );

static double CurrentWaitTime = 0;

std::map<std::string, double> waittimes;
std::map<std::string, double> exectimes;

//______________________________________________________________________
//
MPIScheduler::MPIScheduler( const ProcessorGroup* myworld,
                            const Output*         oport,
                                  MPIScheduler*   parentScheduler)
  : SchedulerCommon(myworld, oport),
    parentScheduler_(parentScheduler),
    log(myworld, oport),
    oport_(oport),
    numMessages_(0),
    messageVolume_(0),
    recvLock("MPI receive lock"),
    sendLock("MPI send lock"),
    dlbLock("loadbalancer lock"),
    waittimesLock("waittimes lock")
{
  d_lasttime = Time::currentSeconds();
  reloc_new_posLabel_ = 0;

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
}

//______________________________________________________________________
//
void
MPIScheduler::problemSetup( const ProblemSpecP&     prob_spec,
                                  SimulationStateP& state )
{
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}

//______________________________________________________________________
//
MPIScheduler::~MPIScheduler()
{
  if (timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      avgStats.close();
      maxStats.close();
    }
  }
}

//______________________________________________________________________
//
SchedulerP
MPIScheduler::createSubScheduler()
{
  MPIScheduler* newsched = scinew MPIScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState=d_sharedState;
  return newsched;
}

//______________________________________________________________________
//
void
MPIScheduler::verifyChecksum()
{
#if SCI_ASSERTION_LEVEL >= 3
  if (Uintah::Parallel::usingMPI()) {
    TAU_PROFILE("MPIScheduler::verifyChecksum()", " ", TAU_USER);

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
      coutLock.lock();
      mpidbg << d_myworld->myrank() << " (MPI_Allreduce) Checking checksum of " << checksum << '\n';
      coutLock.unlock();
    }

    int result_checksum;
    MPI_Allreduce(&checksum, &result_checksum, 1, MPI_INT, MPI_MIN, d_myworld->getComm());

    if (checksum != result_checksum) {
      std::cerr << "Failed task checksum comparison! Not all processes are executing the same taskgraph\n";
      std::cerr << "  Rank: " << d_myworld->myrank() << " of " << d_myworld->size() - 1 << ": has sum " << checksum
                << "  and global is " << result_checksum << '\n';
      MPI_Abort(d_myworld->getComm(), 1);
    }

    if (mpidbg.active()) {
      coutLock.lock();
      mpidbg << d_myworld->myrank() << " (MPI_Allreduce) Check succeeded\n";
      coutLock.unlock();
    }
  }
#endif
}

//______________________________________________________________________
//
#ifdef USE_TAU_PROFILING
extern int create_tau_mapping( const string&      taskname,
                               const PatchSubset* patches );
#endif

//______________________________________________________________________
//
void MPIScheduler::initiateTask( DetailedTask* task,
                                 bool          only_old_recvs,
                                 int           abort_point,
                                 int           iteration )
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::initiateTask");
  TAU_PROFILE("MPIScheduler::initiateTask()", " ", TAU_USER);

  postMPIRecvs(task, only_old_recvs, abort_point, iteration);
  if (only_old_recvs) {
    return;
  }
}  // end initiateTask()

//______________________________________________________________________
//
void
MPIScheduler::initiateReduction( DetailedTask* task )
{
  TAU_PROFILE("MPIScheduler::initiateReduction()", " ", TAU_USER);

  if (reductionout.active() && d_myworld->myrank() == 0) {
    coutLock.lock();
    reductionout << "Running Reduction Task: " << task->getName() << std::endl;
    coutLock.unlock();
  }

  double reducestart = Time::currentSeconds();

  runReductionTask(task);

  double reduceend = Time::currentSeconds();

  emitNode(task, reducestart, reduceend - reducestart, 0);
  mpi_info_.totalreduce    += reduceend - reducestart;
  mpi_info_.totalreducempi += reduceend - reducestart;
}

//______________________________________________________________________
//
void
MPIScheduler::runTask( DetailedTask* task,
                       int           iteration,
                       int           thread_id /*=0*/ )
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::runTask");
  TAU_PROFILE("MPIScheduler::runTask()", " ", TAU_USER);

  if (waitout.active()) {
    waittimesLock.lock();
    waittimes[task->getTask()->getName()] += CurrentWaitTime;
    CurrentWaitTime = 0;
    waittimesLock.unlock();
  }

  double taskstart = Time::currentSeconds();

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }
  std::vector<DataWarehouseP> plain_old_dws(dws.size());
  for (int i = 0; i < (int)dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
  }

  {
    MALLOC_TRACE_TAG_SCOPE("MPIScheduler::runTask::doit(" + task->getName() + ")");
    task->doit(d_myworld, dws, plain_old_dws);
  }

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  double dtask = Time::currentSeconds() - taskstart;

  dlbLock.lock();
  {
    if (execout.active()) {
      exectimes[task->getTask()->getName()] += dtask;
    }

    // if I do not have a sub scheduler
    if (!task->getTask()->getHasSubScheduler()) {
      //add my task time to the total time
      mpi_info_.totaltask += dtask;
      if (!d_sharedState->isCopyDataTimestep() && task->getTask()->getType() != Task::Output) {
        //add contribution for patchlist
        getLoadBalancer()->addContribution(task, dtask);
      }
    }
  }
  dlbLock.unlock();

  postMPISends(task, iteration);

  task->done(dws);  // should this be timed with taskstart? - BJW
  double teststart = Time::currentSeconds();

  sends_[thread_id].testsome(d_myworld);

  mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;

  // Add subscheduler timings to the parent scheduler and reset subscheduler timings
  if (parentScheduler_) {
    parentScheduler_->mpi_info_.totaltask += mpi_info_.totaltask;
    parentScheduler_->mpi_info_.totaltestmpi += mpi_info_.totaltestmpi;
    parentScheduler_->mpi_info_.totalrecv += mpi_info_.totalrecv;
    parentScheduler_->mpi_info_.totalsend += mpi_info_.totalsend;
    parentScheduler_->mpi_info_.totalwaitmpi += mpi_info_.totalwaitmpi;
    parentScheduler_->mpi_info_.totalreduce += mpi_info_.totalreduce;
    mpi_info_.totalreduce    = 0;
    mpi_info_.totalsend      = 0;
    mpi_info_.totalrecv      = 0;
    mpi_info_.totaltask      = 0;
    mpi_info_.totalreducempi = 0;
    mpi_info_.totaltestmpi   = 0;
    mpi_info_.totalwaitmpi   = 0;
  }

  emitNode(task, taskstart, dtask, 0);

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
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::postMPISends");
  double sendstart = Time::currentSeconds();
  bool dbg_active = dbg.active();

  if (dbg_active) {
    cerrLock.lock();
    dbg << d_myworld->myrank() << " postMPISends - task " << *task << '\n';
    cerrLock.unlock();
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

      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || (req->condition == DetailedDep::SubsequentIterations
          && iteration == 0) || (notCopyDataVars_.count(req->req->var->getName()) > 0)) {
        // See comment in DetailedDep about CommCondition
        if (dbg_active) {
          cerrLock.lock();
          dbg << d_myworld->myrank() << "   Ignoring conditional send for " << *req << std::endl;
          cerrLock.unlock();
        }
        continue;
      }

      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep()
          && !oport_->isCheckpointTimestep()) {
        if (dbg_active) {
          cerrLock.lock();
          dbg << d_myworld->myrank() << "   Ignoring non-output-timestep send for " << *req << std::endl;
          cerrLock.unlock();
        }
        continue;
      }

      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      if (dbg.active()) {
        cerrLock.lock();
        ostr << *req << ' ';
        dbg << d_myworld->myrank() << " --> sending " << *req << ", ghost: " << req->req->gtype << ", " << req->req->numGhostCells
            << " from dw " << dw->getID() << '\n';
        cerrLock.unlock();
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
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;

#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
      mpibuff.pack(d_myworld->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      // TODO need to determine if this is actually true now - I don't think it is, APH - 01/07/15
      //only send message if size is greather than zero
      //we need this empty message to enforce modify after read dependencies 
      //if(count>0)
      //{

      if (dbg_active) {
        cerrLock.lock();
        dbg << d_myworld->myrank() << " Sending message number " << batch->messageTag << " to " << to << ": " << ostr.str() << "\n";
        cerrLock.unlock();
      }

      if (mpidbg.active()) {
        cerrLock.lock();
        mpidbg << d_myworld->myrank() << " Sending message number " << batch->messageTag << ", to " << to << ", length: " << count
               << "\n";
        cerrLock.unlock();
      }

      numMessages_++;
      numSend++;
      int typeSize;

      MPI_Type_size(datatype, &typeSize);
      messageVolume_ += count * typeSize;
      volSend += count * typeSize;

      MPI_Request requestid;
      MPI_Isend(buf, count, datatype, to, batch->messageTag, d_myworld->getComm(), &requestid);
      int bytes = count;

      // with multi-threaded schedulers (derived from MPIScheduler), this is written per thread
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // TODO - Somehow, with only the ThreadedMPI scheduler, a race condition exists on a member
      //        one of these CommRecMPI objects (deletion and access to this member)
      //        "sends_" contains per-threads objects so this is puzzling... for now, just lock it.
      //
      // NOTE:  This may have something to do with the PackBufferInfo leak in the Unified Scheduler
      //
      // APH - 01/24/15
      //
      sendLock.writeLock();
      sends_[thread_id].add(requestid, bytes, mpibuff.takeSendlist(), ostr.str(), batch->messageTag);
      sendLock.writeUnlock();

      mpi_info_.totalsendmpi += Time::currentSeconds() - start;

      //}
    }
  }  // end for (DependencyBatch* batch = task->getComputes())

  double dsend = Time::currentSeconds() - sendstart;
  mpi_info_.totalsend += dsend;
  if (dbgst.active() && numSend > 0) {
    if (d_myworld->myrank() == d_myworld->size() / 2) {
      if (dbgst.active()) {
        cerrLock.lock();
        dbgst << d_myworld->myrank() << " Time: " << Time::currentSeconds() << " , NumSend= " << numSend << " , VolSend: "
              << volSend << std::endl;
        cerrLock.unlock();
      }
    }
  }
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
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::postMPIRecvs");

  double recvstart = Time::currentSeconds();
  TAU_PROFILE("MPIScheduler::postMPIRecvs()", " ", TAU_USER);

  bool dbg_active = dbg.active();
  // Receive any of the foreign requires

  if (dbg_active) {
    cerrLock.lock();
    dbg << d_myworld->myrank() << " postMPIRecvs - task " << *task << '\n';
    cerrLock.unlock();
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

  recvLock.writeLock();
  {
    for (; sorted_iter != sorted_reqs.end(); sorted_iter++) {
      DependencyBatch* batch = *sorted_iter;

      // The first thread that calls this on the batch will return true
      // while subsequent threads calling this will block and wait for
      // that first thread to receive the data.

      task->incrementExternalDepCount();
      if (!batch->makeMPIRequest()) {
        if (dbg_active) {
          cerrLock.lock();
          dbg << "Someone else already receiving it\n";
          cerrLock.unlock();
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
        OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
        if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || (req->condition == DetailedDep::SubsequentIterations
            && iteration == 0) || (notCopyDataVars_.count(req->req->var->getName()) > 0)) {

          // See comment in DetailedDep about CommCondition
          if (dbg_active) {
            cerrLock.lock();
            dbg << d_myworld->myrank() << "   Ignoring conditional receive for " << *req << std::endl;
          }
          continue;
        }
        // if we send/recv to an output task, don't send/recv if not an output timestep
        if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep()
            && !oport_->isCheckpointTimestep()) {
          cerrLock.lock();
          dbg << d_myworld->myrank() << "   Ignoring non-output-timestep receive for " << *req << std::endl;
          cerrLock.unlock();
          continue;
        }
        if (dbg_active) {
          ostr << *req << ' ';
          cerrLock.lock();
          dbg  << d_myworld->myrank() << " <-- receiving " << *req << ", ghost: " << req->req->gtype << ", "
               << req->req->numGhostCells << " into dw " << dw->getID() << '\n';
          cerrLock.unlock();
        }

        OnDemandDataWarehouse* posDW;

        // the load balancer is used to determine where data was in the old dw on the prev timestep
        // pass it in if the particle data is on the old dw
        LoadBalancer* lb = 0;
        if (!reloc_new_posLabel_ && parentScheduler_) {
          posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
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
        double start = Time::currentSeconds();
        void* buf;
        int count;
        MPI_Datatype datatype;

#ifdef USE_PACKING
        mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
#else
        mpibuff.get_type(buf, count, datatype);
#endif

        //only receive message if size is greater than zero
        //we need this empty message to enforce modify after read dependencies
        //if(count>0)
        //{
        int from = batch->fromTask->getAssignedResourceIndex();
        ASSERTRANGE(from, 0, d_myworld->size());
        MPI_Request requestid;

        // TODO - do we both of these? (APH 01/22/15)
        if (dbg_active) {
          cerrLock.lock();
          dbg << d_myworld->myrank() << " Receiving message number " << batch->messageTag << " from " << from << ": " << ostr.str()
              << "\n";
          cerrLock.unlock();
        }
        if (mpidbg.active()) {
        cerrLock.lock();
        mpidbg << d_myworld->myrank() << " Posting receive for message number " << batch->messageTag << " from " << from
               << ", length=" << count << "\n";
        cerrLock.unlock();
        }

        MPI_Irecv(buf, count, datatype, from, batch->messageTag, d_myworld->getComm(), &requestid);
        int bytes = count;
        recvs_.add(requestid, bytes, scinew ReceiveHandler(p_mpibuff, pBatchRecvHandler), ostr.str(), batch->messageTag);
        mpi_info_.totalrecvmpi += Time::currentSeconds() - start;
        /*}
         else
         {
         //no message was sent so clean up buffer and handler
         delete p_mpibuff;
         delete pBatchRecvHandler;
         }*/
      }
      else {
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
  }
  recvLock.writeUnlock();

  double drecv = Time::currentSeconds() - recvstart;
  mpi_info_.totalrecv += drecv;

}  // end postMPIRecvs()

//______________________________________________________________________
//
void MPIScheduler::processMPIRecvs(int how_much)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::processMPIRecvs"); TAU_PROFILE("MPIScheduler::processMPIRecvs()", " ", TAU_USER);

  // Should only have external receives in the MixedScheduler version which
  // shouldn't use this function.
  // ASSERT(outstandingExtRecvs.empty());
  if (recvs_.numRequests() == 0) {
    return;
  }

  double start = Time::currentSeconds();

  recvLock.writeLock();
  {
    switch (how_much) {
      case TEST :
        recvs_.testsome(d_myworld);
        break;
      case WAIT_ONCE :
        mpidbg << d_myworld->myrank() << " Start waiting once...\n";
        recvs_.waitsome(d_myworld);
        mpidbg << d_myworld->myrank() << " Done  waiting once...\n";
        break;
      case WAIT_ALL :
        // This will allow some receives to be "handled" by their
        // AfterCommincationHandler while waiting for others.
        mpidbg << d_myworld->myrank() << "  Start waiting...\n";
        while ((recvs_.numRequests() > 0)) {
          bool keep_waiting = recvs_.waitsome(d_myworld);
          if (!keep_waiting) {
            break;
          }
        }
        mpidbg << d_myworld->myrank() << "  Done  waiting...\n";
        break;
    } // end switch
  }
  recvLock.writeUnlock();

  mpi_info_.totalwaitmpi += Time::currentSeconds() - start;
  CurrentWaitTime += Time::currentSeconds() - start;

}  // end processMPIRecvs()

//______________________________________________________________________
//

void
MPIScheduler::execute( int tgnum /*=0*/,
                       int iteration /*=0*/ )
{

  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::execute");

  TAU_PROFILE("MPIScheduler::execute()", " ", TAU_USER);
  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[MPIScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[MPIScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[MPIScheduler::execute()] " , TAU_USER);
  TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[MPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(testsometimer, "Test Some", "[MPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[MPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[MPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[MPIScheduler::execute()] ", TAU_USER);

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
      cerrLock.lock();
      std::cerr << "MPIScheduler skipping execute, no tasks\n";
      cerrLock.unlock();
    }
    return;
  }

  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  if (timeout.active()) {
    d_labels.clear();
    d_times.clear();
    //emitTime("time since last execute");
  }

  int me = d_myworld->myrank();

  // TODO determine exactly what this does and at what cost/benefit (APH 01/22/15)
  makeTaskGraphDoc(dts, me);

  //if(timeout.active())
  //emitTime("taskGraph output");

  mpi_info_.totalreduce = 0;
  mpi_info_.totalsend = 0;
  mpi_info_.totalrecv = 0;
  mpi_info_.totaltask = 0;
  mpi_info_.totalreducempi = 0;
  mpi_info_.totalsendmpi = 0;
  mpi_info_.totalrecvmpi = 0;
  mpi_info_.totaltestmpi = 0;
  mpi_info_.totalwaitmpi = 0;

  int numTasksDone = 0;

  if (dbg.active()) {
    cerrLock.lock();
    dbg << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)\n";
    cerrLock.unlock();
  }

  bool abort = false;
  int abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  TAU_PROFILE_TIMER(doittimer, "Task execution", "[MPIScheduler::execute() loop] ", TAU_USER); TAU_PROFILE_START(doittimer);

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
      taskorder << d_myworld->myrank() << " Running task static order: " << task->getSaticOrder() << " , scheduled order: "
                << numTasksDone << std::endl;
    }

    if (taskdbg.active()) {
      taskdbg << me << " Initiating task:  \t";
      printTask(taskdbg, task);
      taskdbg << '\n';
    }

#ifdef USE_TAU_PROFILING
    int id;
    const PatchSubset* patches = task->getPatches();
    id = create_tau_mapping( task->getTask()->getName(), patches );

    string phase_name = "no patches";
    if (patches && patches->size() > 0) {
      phase_name = "level";
      for(int i=0;i<patches->size();i++) {

        ostringstream patch_num;
        patch_num << patches->get(i)->getLevel()->getIndex();

        if (i == 0) {
          phase_name = phase_name + " " + patch_num.str();
        }
        else {
          phase_name = phase_name + ", " + patch_num.str();
        }
      }
    }

    static map<string,int> phase_map;
    static int unique_id = 99999;
    int phase_id;
    map<string,int>::iterator iter = phase_map.find( phase_name );
    if( iter != phase_map.end() ) {
      phase_id = (*iter).second;
    }
    else {
      TAU_MAPPING_CREATE( phase_name, "", (TauGroup_t) unique_id, "TAU_USER", 0 );
      phase_map[ phase_name ] = unique_id;
      phase_id = unique_id++;
    }
    // Task name
    TAU_MAPPING_OBJECT(tautimer)
    TAU_MAPPING_LINK(tautimer, (TauGroup_t)id);// EXTERNAL ASSOCIATION
    TAU_MAPPING_PROFILE_TIMER(doitprofiler, tautimer, 0)
    TAU_MAPPING_PROFILE_START(doitprofiler,0);
#endif

    if (task->getTask()->getType() == Task::Reduction) {
      if (!abort) {
        initiateReduction(task);
      }
    }
    else {
      initiateTask(task, abort, abort_point, iteration);
      processMPIRecvs(WAIT_ALL);
      ASSERT(recvs_.numRequests() == 0);
      runTask(task, iteration);

      if (taskdbg.active()) {
        taskdbg << d_myworld->myrank() << " Completed task:  \t";
        printTask(taskdbg, task);
        taskdbg << '\n';
        printTaskLevels(d_myworld, taskLevel_dbg, task);
      }
    }
  
      TAU_MAPPING_PROFILE_STOP(0);

    if(!abort && dws[dws.size()-1] && dws[dws.size()-1]->timestepAborted()){
      abort = true;
      abort_point = task->getTask()->getSortedOrder();
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
    }
  } // end while( numTasksDone < ntasks )

  TAU_PROFILE_STOP(doittimer);

  if (timeout.active()) {
    emitTime("MPI send time", mpi_info_.totalsendmpi);
    emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
    emitTime("Total send time", mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time", mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    emitTime("Total reduction time", mpi_info_.totalreduce - mpi_info_.totalreducempi);
    emitTime("Total comm time", mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

    double time = Time::currentSeconds();
    double totalexec = time - d_lasttime;

    d_lasttime = time;

    emitTime("Other execution time",
             totalexec - mpi_info_.totalsend - mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);
  }

  if( !parentScheduler_ ) { // If this scheduler is the root scheduler...
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime; // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
  }

  // Don't need to lock sends 'cause all threads are done at this point.
  sends_[0].waitall(d_myworld);

  ASSERT(sends_[0].numRequests() == 0);
  //if(timeout.active())
    //emitTime("final wait");
  if (restartable && tgnum == (int)graphs.size() - 1) {
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

  finalizeTimestep();
  log.finishTimestep();

  if (timeout.active() && !parentScheduler_) {  // only do on toplevel scheduler
    //emitTime("finalize");

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

    MPI_Reduce(&d_times[0], &d_totaltimes[0], static_cast<int>(d_times.size()), MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());
    MPI_Reduce(&d_times[0], &d_maxtimes[0],   static_cast<int>(d_times.size()), MPI_DOUBLE, MPI_MAX, 0, d_myworld->getComm());

    double total = 0, avgTotal = 0, maxTotal = 0;
    for (int i = 0; i < (int)d_totaltimes.size(); i++) {
      d_avgtimes[i] = d_totaltimes[i] / d_myworld->size();
      if (strcmp(d_labels[i], "Total task time") == 0) {
        avgTask = d_avgtimes[i];
        maxTask = d_maxtimes[i];
      }
      else if (strcmp(d_labels[i], "Total comm time") == 0) {
        avgComm = d_avgtimes[i];
        maxComm = d_maxtimes[i];
      }
      else if (strncmp(d_labels[i], "Num", 3) == 0) {
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
        out << "MPIScheduler: " << d_labels[i] << ": ";
        int len = (int)(strlen(d_labels[i]) + strlen("MPIScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++)
          out << ' ';
        double percent;
        if (strncmp(d_labels[i], "Num", 3) == 0) {
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i] / d_totaltimes[i] * 100;
        }
        else {
          percent = (*data[file])[i] / total * 100;
        }
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << std::endl << std::endl;
    }

    if (me == 0) {
      timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1 - avgTask / maxTask) * 100 << " load imbalance (exec)%\n";
      timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1 - avgComm / maxComm) * 100 << " load imbalance (comm)%\n";
      timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1 - avgCell / maxCell) * 100 << " load imbalance (theoretical)%\n";
    }

    double time = Time::currentSeconds();
    //double rtime=time-d_lasttime;
    d_lasttime = time;
    //timeout << "MPIScheduler: TOTAL                                    "
    //        << total << '\n';
    //timeout << "MPIScheduler: time sum reduction (one processor only): " 
    //        << rtime << '\n';
  }

  if (execout.active()) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {
      std::ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);

      for (std::map<std::string, double>::iterator iter = exectimes.begin(); iter != exectimes.end(); iter++) {
        fout << std::fixed << d_myworld->myrank() << ": TaskExecTime(s): " << iter->second << " Task:" << iter->first << std::endl;
      }
      fout.close();
      exectimes.clear();
    }
  }

  if (waitout.active()) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {

      if (d_myworld->myrank() == 0 || d_myworld->myrank() == d_myworld->size() / 2
          || d_myworld->myrank() == d_myworld->size() - 1) {

        std::ofstream wout;
        char fname[100];
        sprintf(fname, "waittimes.%d.%d", d_myworld->size(), d_myworld->myrank());
        wout.open(fname);

        for (std::map<std::string, double>::iterator iter = waittimes.begin(); iter != waittimes.end(); iter++) {
          wout << std::fixed << d_myworld->myrank() << ":   TaskWaitTime(TO): " << iter->second << " Task:" << iter->first << std::endl;
        }

        for (std::map<std::string, double>::iterator iter = DependencyBatch::waittimes.begin(); iter != DependencyBatch::waittimes.end();
            iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second << " Task:" << iter->first << std::endl;
        }

        wout.close();
      }

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }

  if (dbg.active()) {
    coutLock.lock();
    dbg << me << " MPIScheduler finished\n";
    coutLock.unlock();
  }
}

//______________________________________________________________________
//
void
MPIScheduler::emitTime( const char* label )
{
   double time = Time::currentSeconds();
   emitTime(label, time-d_lasttime);
   d_lasttime=time;
}

//______________________________________________________________________
//
void
MPIScheduler::emitTime( const char*  label,
                              double dt )
{
   d_labels.push_back(label);
   d_times.push_back(dt);
}

