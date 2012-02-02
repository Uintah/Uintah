/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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

#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

#include   <sstream>
#include   <iomanip>
#include   <map>
#include   <cstring>

// Pack data into a buffer before sending -- testing to see if this
// works better and avoids certain problems possible when you allow
// tasks to modify data that may have a pending send.
#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllimport)
#else
#define UINTAHSHARE
#endif
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern UINTAHSHARE SCIRun::Mutex       cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg("MPIScheduler",             false);
static DebugStream timeout("MPIScheduler.timings", false);
static DebugStream reductionout("ReductionTasks",  false);

DebugStream waitout("WaitTimes", false);
DebugStream execout("ExecTimes", false);
DebugStream taskdbg("TaskDBG",   false);
DebugStream taskLevel_dbg("TaskLevel", false);
DebugStream mpidbg("MPIDBG",false);

extern ofstream wout;
static double CurrentWaitTime=0;
map<string,double> waittimes;
map<string,double> exectimes;

 

MPIScheduler::MPIScheduler( const ProcessorGroup * myworld,
			          Output         * oport,
			          MPIScheduler   * parentScheduler) :
  SchedulerCommon( myworld, oport ),
  parentScheduler( parentScheduler ), log( myworld, oport ), oport_(oport)
{
  d_lasttime=Time::currentSeconds();
  reloc_new_posLabel_=0;

  // Sometimes it is easier to only see the spew on one processor...
  //if( d_myworld->myrank() == 0 ) {
  //  taskdbg.setActive( false );
  //}

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


void
MPIScheduler::problemSetup(const ProblemSpecP& prob_spec,
                           SimulationStateP& state)
{
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}

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

SchedulerP
MPIScheduler::createSubScheduler()
{
  MPIScheduler* newsched = scinew MPIScheduler(d_myworld, m_outPort, this);
  newsched->d_sharedState = d_sharedState;
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState=d_sharedState;
  return newsched;
}

void
MPIScheduler::verifyChecksum()
{
#if SCI_ASSERTION_LEVEL >= 3
  TAU_PROFILE("MPIScheduler::verifyChecksum()", " ", TAU_USER); 

  // Compute a simple checksum to make sure that all processes
  // are trying to execute the same graph.  We should do two
  // things in the future:
  //  - make a flag to turn this off
  //  - make the checksum more sophisticated
  int checksum = 0;
  for (unsigned i = 0; i < graphs.size(); i++)
    checksum += graphs[i]->getTasks().size();
  mpidbg << d_myworld->myrank() << " (Allreduce) Checking checksum of " << checksum << '\n';
  int result_checksum;
  MPI_Allreduce(&checksum, &result_checksum, 1, MPI_INT, MPI_MIN,
		d_myworld->getComm());
  if(checksum != result_checksum){
    cerr << "Failed task checksum comparison!\n";
    cerr << "Processor: " << d_myworld->myrank() << " of "
	 << d_myworld->size() << ": has sum " << checksum
	 << " and global is " << result_checksum << '\n';
    MPI_Abort(d_myworld->getComm(), 1);
  }
  mpidbg << d_myworld->myrank() << " (Allreduce) Check succeeded\n";
#endif
}

#ifdef USE_TAU_PROFILING
extern int create_tau_mapping( const string & taskname,
			       const PatchSubset * patches );  // ThreadPool.cc
#endif

void
MPIScheduler::wait_till_all_done()
{
  if( mixedDebug.active() ) {
    cerrLock.lock();mixedDebug << "MPIScheduler::wait_till_all_done()\n";
    cerrLock.unlock();
  }
  return;
}

void
MPIScheduler::initiateTask( DetailedTask          * task,
			    bool only_old_recvs, int abort_point, int iteration )
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::initiateTask");
  TAU_PROFILE("MPIScheduler::initiateTask()", " ", TAU_USER); 

  postMPIRecvs(task, only_old_recvs, abort_point, iteration);
  if(only_old_recvs) {
    return;
  }
} // end initiateTask()

void
MPIScheduler::initiateReduction( DetailedTask          * task)
{
  TAU_PROFILE("MPIScheduler::initiateReduction()", " ", TAU_USER); 
  {
    if(reductionout.active() && d_myworld->myrank()==0)
      reductionout << "Running Reduction Task: " << task->getName() << endl;

    double reducestart = Time::currentSeconds();
    
    runReductionTask(task);

    double reduceend = Time::currentSeconds();

    emitNode(task, reducestart, reduceend - reducestart, 0);
    mpi_info_.totalreduce += reduceend-reducestart;
    mpi_info_.totalreducempi += reduceend-reducestart;
  }
}

void
MPIScheduler::runTask( DetailedTask         * task, int iteration)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::runTask");
  TAU_PROFILE("MPIScheduler::runTask()", " ", TAU_USER); 

  if(waitout.active())
  {
    waittimes[task->getTask()->getName()]+=CurrentWaitTime;
    CurrentWaitTime=0;
  }

  double taskstart = Time::currentSeconds();
  
  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }
  vector<DataWarehouseP> plain_old_dws(dws.size());
  for(int i=0;i<(int)dws.size();i++)
    plain_old_dws[i] = dws[i].get_rep();
  //const char* tag = AllocatorSetDefaultTag(task->getTask()->getName());
 {
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::runTask::doit(" + task->getName() + ")" );
  task->doit(d_myworld, dws, plain_old_dws);
 }
  //AllocatorSetDefaultTag(tag);

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  double dtask = Time::currentSeconds()-taskstart;
 
  if(execout.active())
  {
    exectimes[task->getTask()->getName()]+=dtask;
  }

  //if i do not have a sub scheduler 
  if(!task->getTask()->getHasSubScheduler())
  {
    //add my task time to the total time
    mpi_info_.totaltask += dtask;  
    //if(d_myworld->myrank()==0)
    //  cout << "adding: " << dtask << " to counters, new total: " << mpi_info_.totaltask << endl;
    if(!d_sharedState->isCopyDataTimestep() && task->getTask()->getType()!=Task::Output)
    {
      //if(d_myworld->myrank()==0 && task->getPatches()!=0)
      //  cout << d_myworld->myrank() << " adding: " << task->getTask()->getName() << " to profile:" << dtask << " on patches:" << *(task->getPatches()) << endl;
      //add contribution for patchlist
      getLoadBalancer()->addContribution(task,dtask);
    }
  }

  postMPISends( task, iteration );
  task->done(dws); // should this be timed with taskstart? - BJW
  double teststart = Time::currentSeconds();

  sends_.testsome( d_myworld );

  mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;
 
  if(parentScheduler) //add my timings to the parent scheduler
  {
  //  if(d_myworld->myrank()==0)
  //    cout << "adding: " << mpi_info_.totaltask << " to parent counters, new total: " << parentScheduler->mpi_info_.totaltask << endl;
    parentScheduler->mpi_info_.totaltask+=mpi_info_.totaltask;
    parentScheduler->mpi_info_.totaltestmpi+=mpi_info_.totaltestmpi;
    parentScheduler->mpi_info_.totalrecv+=mpi_info_.totalrecv;
    parentScheduler->mpi_info_.totalsend+=mpi_info_.totalsend;
    parentScheduler->mpi_info_.totalwaitmpi+=mpi_info_.totalwaitmpi;
    parentScheduler->mpi_info_.totalreduce+=mpi_info_.totalreduce;
  }

  emitNode(task, taskstart, dtask, 0);
  
} // end runTask()

void
MPIScheduler::runReductionTask( DetailedTask         * task)
{
  const Task::Dependency* mod = task->getTask()->getModifies();
  ASSERT(!mod->next);
  
  OnDemandDataWarehouse* dw = dws[mod->mapDataWarehouse()].get_rep();
  ASSERT(task->getTask()->d_comm>=0);
  dw->reduceMPI(mod->var, mod->reductionLevel, mod->matls, task->getTask()->d_comm);
  task->done(dws);

  cerrLock.lock(); taskdbg << d_myworld->myrank() << " Completed: \t";
  printTask(taskdbg, task); taskdbg << '\n'; cerrLock.unlock();

}

void
MPIScheduler::postMPISends( DetailedTask         * task, int iteration )
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::postMPISends");
  double sendstart = Time::currentSeconds();
  if( dbg.active()) {
    cerrLock.lock();dbg << d_myworld->myrank() << " postMPISends - task " << *task << '\n';
    cerrLock.unlock();
  }

  // Send data to dependendents
  for(DependencyBatch* batch = task->getComputes();
      batch != 0; batch = batch->comp_next){

    // Prepare to send a message
#ifdef USE_PACKING
    PackBufferInfo mpibuff;
#else
    BufferInfo mpibuff;
#endif
    // Create the MPI type
    int to = batch->toTasks.front()->getAssignedResourceIndex();
    ASSERTRANGE(to, 0, d_myworld->size());
    ostringstream ostr;
    ostr.clear();
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || 
          (req->condition == DetailedDep::SubsequentIterations && iteration == 0)) {
        // See comment in DetailedDep about CommCondition
        dbg << d_myworld->myrank() << "   Ignoring conditional send for " << *req << endl;
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && 
          !oport_->isOutputTimestep() && !oport_->isCheckpointTimestep()) {
        dbg << d_myworld->myrank() << "   Ignoring non-output-timestep send for " << *req << endl;
        continue;
      }
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();

      //dbg.setActive(req->req->lookInOldTG);
      if (dbg.active()) {
        ostr << *req << ' ';
        //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
        dbg << d_myworld->myrank() << " --> sending " << *req << ", ghost: " << req->req->gtype << ", " << req->req->numGhostCells << " from dw " << dw->getID() << '\n';
      }
      const VarLabel* posLabel;
      OnDemandDataWarehouse* posDW;

      // the load balancer is used to determine where data was in the old dw on the prev timestep -
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;

      if(!reloc_new_posLabel_ && parentScheduler){
	posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
	posLabel = parentScheduler->reloc_new_posLabel_;
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output)
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
	posLabel = reloc_new_posLabel_;
      }
      MPIScheduler* top = this;
      while(top->parentScheduler) top = top->parentScheduler;

      dw->sendMPI(batch, posLabel, mpibuff, posDW, req, lb);
    }
    // Post the send
    if(mpibuff.count()>0){
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

      //only send message if size is greather than zero
      //we need this empty message to enforce modify after read dependencies 
      //if(count>0)
      //{
        if( dbg.active()) {
          cerrLock.lock();
          //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
          dbg << d_myworld->myrank() << " Sending message number " << batch->messageTag 
            << " to " << to << ": " << ostr.str() << "\n"; cerrLock.unlock();
          //dbg.setActive(false);
        }
        //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
        mpidbg <<d_myworld->myrank() << " Sending message number " << batch->messageTag << ", to " << to << ", length: " << count << "\n"; 

        numMessages_++;
        int typeSize;

        MPI_Type_size(datatype,&typeSize);
        messageVolume_+=count*typeSize;

        MPI_Request requestid;
        MPI_Isend(buf, count, datatype, to, batch->messageTag,
            d_myworld->getComm(), &requestid);
        int bytes = count;

        sends_.add( requestid, bytes, mpibuff.takeSendlist(), ostr.str(), batch->messageTag );
        mpi_info_.totalsendmpi += Time::currentSeconds() - start;
      //}
    }
  } // end for (DependencyBatch * batch = task->getComputes() )
  double dsend = Time::currentSeconds()-sendstart;
  mpi_info_.totalsend += dsend;

} // end postMPISends();


struct CompareDep {
bool operator()(DependencyBatch* a, DependencyBatch* b)
{
  return a->messageTag < b->messageTag;
}
};
void
MPIScheduler::postMPIRecvs( DetailedTask * task, bool only_old_recvs, int abort_point, int iteration)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::postMPIRecvs");
  double recvstart = Time::currentSeconds();
  TAU_PROFILE("MPIScheduler::postMPIRecvs()", " ", TAU_USER); 

  // Receive any of the foreign requires

  if( dbg.active()) {
    cerrLock.lock();dbg << d_myworld->myrank() << " postMPIRecvs - task " << *task << '\n';
    cerrLock.unlock();
  }

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_COMM)
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_COMM);

  // sort the requires, so in case there is a particle send we receive it with
  // the right message tag

  vector<DependencyBatch*> sorted_reqs;
  map<DependencyBatch*, DependencyBatch*>::const_iterator iter = 
    task->getRequires().begin();
  for( ; iter != task->getRequires().end(); iter++) {
    sorted_reqs.push_back(iter->first);
  }
  CompareDep comparator;;
  sort(sorted_reqs.begin(), sorted_reqs.end(), comparator);
  vector<DependencyBatch*>::iterator sorted_iter = sorted_reqs.begin();
  for( ; sorted_iter != sorted_reqs.end(); sorted_iter++) {
    DependencyBatch* batch = *sorted_iter;

    // The first thread that calls this on the batch will return true
    // while subsequent threads calling this will block and wait for
    // that first thread to receive the data.

    task->incrementExternalDepCount();
    //cout << d_myworld->myrank() << " Add dep count to task " << *task << " for ext: " << *batch->fromTask << ": " << task->getExternalDepCount() << endl;
    if( !batch->makeMPIRequest() ) {
      //externalRecvs.push_back( batch ); // no longer necessary

      if( mixedDebug.active() ) {
        cerrLock.lock();mixedDebug << "Someone else already receiving it\n";
        cerrLock.unlock();
      }

      continue;
    }

    if(only_old_recvs){
      if(dbg.active()){
        dbg << "abort analysis: " << batch->fromTask->getTask()->getName()
          << ", so=" << batch->fromTask->getTask()->getSortedOrder()
          << ", abort_point=" << abort_point << '\n';
        if(batch->fromTask->getTask()->getSortedOrder() <= abort_point)
          dbg << "posting MPI recv for pre-abort message " 
            << batch->messageTag << '\n';
      }
      if(!(batch->fromTask->getTask()->getSortedOrder() <= abort_point)) {
        continue;
      }
    }

    // Prepare to receive a message
    BatchReceiveHandler* pBatchRecvHandler = 
      scinew BatchReceiveHandler(batch);
    PackBufferInfo* p_mpibuff = 0;
#ifdef USE_PACKING
    p_mpibuff = scinew PackBufferInfo();
    PackBufferInfo& mpibuff = *p_mpibuff;
#else
    BufferInfo mpibuff;
#endif

    ostringstream ostr;
    ostr.clear();
    // Create the MPI type
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      //dbg.setActive(req->req->lookInOldTG );
      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || 
          (req->condition == DetailedDep::SubsequentIterations && iteration == 0)) {
        // See comment in DetailedDep about CommCondition

        dbg << d_myworld->myrank() << "   Ignoring conditional receive for " << *req << endl;
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep() && !oport_->isCheckpointTimestep()) {
        dbg << d_myworld->myrank() << "   Ignoring non-output-timestep receive for " << *req << endl;
        continue;
      }
      if (dbg.active()) {
        ostr << *req << ' ';
        dbg << d_myworld->myrank() << " <-- receiving " << *req << ", ghost: " << req->req->gtype << ", " << req->req->numGhostCells << " into dw " << dw->getID() << '\n';
      }

      OnDemandDataWarehouse* posDW;
      const VarLabel* posLabel;

      // the load balancer is used to determine where data was in the old dw on the prev timestep
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;
      if(!reloc_new_posLabel_ && parentScheduler){
        posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
        posLabel = parentScheduler->reloc_new_posLabel_;
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output)
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
        posLabel = reloc_new_posLabel_;
      }

      MPIScheduler* top = this;
      while(top->parentScheduler) top = top->parentScheduler;

      dw->recvMPI(batch, mpibuff, posDW, req, lb);

      if (!req->isNonDataDependency()) {
        graphs[currentTG_]->getDetailedTasks()->setScrubCount(req->req, req->matl, req->fromPatch, dws);
      }

    }

    // Post the receive
    if(mpibuff.count()>0){

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
      //only recieve message if size is greater than zero
      //we need this empty message to enforce modify after read dependencies 
      //if(count>0)
      //{
        int from = batch->fromTask->getAssignedResourceIndex();
        ASSERTRANGE(from, 0, d_myworld->size());
        MPI_Request requestid;

        if( dbg.active()) {
          cerrLock.lock();
          //if (d_myworld->myrank() == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && from == 43)
          dbg << d_myworld->myrank() << " Recving message number " << batch->messageTag 
            << " from " << from << ": " << ostr.str() << "\n"; cerrLock.unlock();
          //dbg.setActive(false);
        }

        //if (d_myworld->myrank() == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && from == 43) 
        mpidbg << d_myworld->myrank() << " Posting receive for message number " << batch->messageTag << " from " << from << ", length=" << count << "\n";
        MPI_Irecv(buf, count, datatype, from, batch->messageTag,
            d_myworld->getComm(), &requestid);
        int bytes = count;
        recvs_.add(requestid, bytes,
            scinew ReceiveHandler(p_mpibuff, pBatchRecvHandler),
            ostr.str(), batch->messageTag);
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
      // otherwise, these will be deleted after it receives and unpacks
      // the data.
      delete p_mpibuff;
      delete pBatchRecvHandler;
#endif	        
    }
  } // end for

  double drecv = Time::currentSeconds() - recvstart;
  mpi_info_.totalrecv += drecv;


} // end postMPIRecvs()

void
MPIScheduler::processMPIRecvs(int how_much)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::processMPIRecvs");
  TAU_PROFILE("MPIScheduler::processMPIRecvs()", " ", TAU_USER);

  // Should only have external receives in the MixedScheduler version which
  // shouldn't use this function.
  // ASSERT(outstandingExtRecvs.empty());
  double start = Time::currentSeconds();

  switch (how_much) {
  case TEST:
    if (recvs_.numRequests() > 0)
      recvs_.testsome(d_myworld);
    break;
  case WAIT_ONCE:
    if(recvs_.numRequests() > 0) {
      mpidbg << d_myworld->myrank() << " Start waiting once...\n";
      recvs_.waitsome(d_myworld);
      mpidbg << d_myworld->myrank() << " Done  waiting once...\n";
    }
    break;
  case WAIT_ALL:
    // This will allow some receives to be "handled" by their
    // AfterCommincationHandler while waiting for others.  
    mpidbg << d_myworld->myrank() << "  Start waiting...\n";
    while( (recvs_.numRequests() > 0)) {
      bool keep_waiting = recvs_.waitsome(d_myworld);
      if (!keep_waiting)
        break;
    }
    mpidbg << d_myworld->myrank() << "  Done  waiting...\n";
  }
  mpi_info_.totalwaitmpi+=Time::currentSeconds()-start;
  CurrentWaitTime+=Time::currentSeconds()-start;

} // end processMPIRecvs()

void
MPIScheduler::execute(int tgnum /*=0*/, int iteration /*=0*/)
{

  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::execute");
  TAU_PROFILE("MPIScheduler::execute()", " ", TAU_USER); 
  
  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[MPIScheduler::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[MPIScheduler::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[MPIScheduler::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[MPIScheduler::execute()] ", 
                    TAU_USER); 
  TAU_PROFILE_TIMER(testsometimer, "Test Some", "[MPIScheduler::execute()] ", 
                    TAU_USER); 
  TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[MPIScheduler::execute()] ", 
                    TAU_USER); 
  TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[MPIScheduler::execute()] ", 
                    TAU_USER); 
  TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[MPIScheduler::execute()] ", 
                    TAU_USER); 

  ASSERTRANGE(tgnum, 0, (int)graphs.size());
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  DetailedTasks* dts = tg->getDetailedTasks();

  if(dts == 0){
    if (d_myworld->myrank() == 0)
      cerr << "MPIScheduler skipping execute, no tasks\n";
    return;
  }

  //ASSERT(pg_ == 0);
  //pg_ = pg;
  
  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++)
    dts->localTask(i)->resetDependencyCounts();

  if(timeout.active()){
    d_labels.clear();
    d_times.clear();
    //emitTime("time since last execute");
  }

  int me = d_myworld->myrank();
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

  if( dbg.active()) {
    cerrLock.lock();
    dbg << me << " Executing " << dts->numTasks() << " tasks (" 
	       << ntasks << " local)\n";
    cerrLock.unlock();
  }

  bool abort=false;
  int abort_point = 987654;

  int i = 0;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0)
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);

  TAU_PROFILE_TIMER(doittimer, "Task execution", 
		    "[MPIScheduler::execute() loop] ", TAU_USER); 
  TAU_PROFILE_START(doittimer);

  while( numTasksDone < ntasks) {
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
      taskdbg << me << " Initiating task:  \t"; printTask(taskdbg, task); taskdbg << '\n';

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
          } else {
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
      } else {
        TAU_MAPPING_CREATE( phase_name, "",
			    (TauGroup_t) unique_id, "TAU_USER", 0 );
        phase_map[ phase_name ] = unique_id;
        phase_id = unique_id++;
      }
      // Task name
      TAU_MAPPING_OBJECT(tautimer)
      TAU_MAPPING_LINK(tautimer, (TauGroup_t)id);  // EXTERNAL ASSOCIATION
      TAU_MAPPING_PROFILE_TIMER(doitprofiler, tautimer, 0)
      TAU_MAPPING_PROFILE_START(doitprofiler,0);
#endif


      if (task->getTask()->getType() == Task::Reduction){
        if(!abort)
          initiateReduction(task);
      }
      else {
        initiateTask( task, abort, abort_point, iteration );
        processMPIRecvs(WAIT_ALL);
        ASSERT(recvs_.numRequests() == 0);
        runTask(task, iteration);
        
        //  output to help application developers
        taskdbg << d_myworld->myrank() << " Completed task:  \t";
        printTask(taskdbg, task); taskdbg << '\n';
        
        printTaskLevels( d_myworld, taskLevel_dbg, task );
      }
  
      TAU_MAPPING_PROFILE_STOP(0);

    if(!abort && dws[dws.size()-1] && dws[dws.size()-1]->timestepAborted()){
      abort = true;
      abort_point = task->getTask()->getSortedOrder();
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
    }
  } // end while( numTasksDone < ntasks )
  TAU_PROFILE_STOP(doittimer);

  // wait for all tasks to finish -- i.e. MixedScheduler
  // MPIScheduler will just continue.
  wait_till_all_done();
  
  //if (d_generation > 2)
  //dws[dws.size()-2]->printParticleSubsets();

  if(timeout.active()){
    emitTime("MPI send time", mpi_info_.totalsendmpi);
    //emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
    emitTime("Total send time", 
             mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time", 
             mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    //emitTime("Total reduction time", 
    //         mpi_info_.totalreduce - mpi_info_.totalreducempi);
    emitTime("Total comm time", 
             mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

    double time      = Time::currentSeconds();
    double totalexec = time - d_lasttime;
    
    d_lasttime = time;

    emitTime("Other excution time", totalexec - mpi_info_.totalsend -
             mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);
  }

  if (d_sharedState != 0) { // subschedulers don't have a sharedState
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime; // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
  }

  // Don't need to lock sends 'cause all threads are done at this point.
  sends_.waitall(d_myworld);
  ASSERT(sends_.numRequests() == 0);
  //if(timeout.active())
    //emitTime("final wait");
  if(restartable && tgnum == (int) graphs.size() -1) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size()-1]->timestepRestarted();
    int netrestart;
    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR,
                  d_myworld->getComm());
    if(netrestart) {
      dws[dws.size()-1]->restartTimestep();
      if (dws[0])
        dws[0]->setRestarted();
    }
  }

  finalizeTimestep();
  

  log.finishTimestep();
  if(timeout.active() && !parentScheduler){ // only do on toplevel scheduler
    //emitTime("finalize");

    // add number of cells, patches, and particles
    int numCells = 0, numParticles = 0;
    OnDemandDataWarehouseP dw = dws[dws.size()-1];
    const GridP grid(const_cast<Grid*>(dw->getGrid()));
    const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(grid)->getSubset(d_myworld->myrank());
    for (int p = 0; p < myPatches->size(); p++) {
      const Patch* patch = myPatches->get(p);
      IntVector range = patch->getExtraCellHighIndex() - patch->getExtraCellLowIndex();
      numCells += range.x()*range.y()*range.z();

      // go through all materials since getting an MPMMaterial correctly would depend on MPM
      for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
        if (dw->haveParticleSubset(m, patch))
          numParticles += dw->getParticleSubset(m, patch)->numParticles();
      }
    }
    d_labels.push_back("NumPatches");
    d_labels.push_back("NumCells");
    d_labels.push_back("NumParticles");
    d_times.push_back(myPatches->size());
    d_times.push_back(numCells);
    d_times.push_back(numParticles);
    vector<double> d_totaltimes(d_times.size());
    vector<double> d_maxtimes(d_times.size());
    vector<double> d_avgtimes(d_times.size());
    double maxTask = -1, avgTask = -1;
    double maxComm = -1, avgComm = -1;
    double maxCell = -1, avgCell = -1;
    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE,
               MPI_SUM, 0, d_myworld->getComm());
    MPI_Reduce(&d_times[0], &d_maxtimes[0], (int)d_times.size(), MPI_DOUBLE,
               MPI_MAX, 0, d_myworld->getComm());

    double total = 0, avgTotal = 0, maxTotal = 0;
    for(int i=0;i<(int)d_totaltimes.size();i++) {
      d_avgtimes[i] = d_totaltimes[i]/d_myworld->size();
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

      total+= d_times[i];
      avgTotal += d_avgtimes[i];
      maxTotal += d_maxtimes[i];
    }

    // to not duplicate the code
    vector <ofstream*> files;
    vector <vector<double>* > data;
    files.push_back(&timingStats);
    data.push_back(&d_times);

    if (me == 0) {
      files.push_back(&avgStats);
      files.push_back(&maxStats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (unsigned file = 0; file < files.size(); file++) {
      ofstream& out = *files[file];
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep() << endl;
      for(int i=0;i<(int)(*data[file]).size();i++){
        out << "MPIScheduler: " << d_labels[i] << ": ";
        int len = (int)(strlen(d_labels[i])+strlen("MPIScheduler: ")+strlen(": "));
        for(int j=len;j<55;j++)
          out << ' ';
        double percent;
        if (strncmp(d_labels[i], "Num", 3) == 0)
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i]/d_totaltimes[i]*100;
        else
          percent = (*data[file])[i]/total*100;
        out << (*data[file])[i] <<  " (" << percent << "%)\n";
      }
      out << endl << endl;
    }

    if (me == 0) {
      timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1-avgTask/maxTask)*100 << " load imbalance (exec)%\n";
      timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1-avgComm/maxComm)*100 << " load imbalance (comm)%\n";
      timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1-avgCell/maxCell)*100 << " load imbalance (theoretical)%\n";
    }
    double time = Time::currentSeconds();
    //double rtime=time-d_lasttime;
    d_lasttime=time;
    //timeout << "MPIScheduler: TOTAL                                    "
    //        << total << '\n';
    //timeout << "MPIScheduler: time sum reduction (one processor only): " 
    //        << rtime << '\n';
  }

  if(execout.active())
  {
    static int count=0;
      
    if(++count%10==0)
    {
      ofstream fout;
      char filename[100];
      sprintf(filename,"exectimes.%d.%d",d_myworld->size(),d_myworld->myrank());
      fout.open(filename);
      
      for(map<string,double>::iterator iter=exectimes.begin();iter!=exectimes.end();iter++)
        fout << fixed << d_myworld->myrank() << ": TaskExecTime: " << iter->second << " Task:" << iter->first << endl;
      fout.close();
      //exectimes.clear();
    }
  }
  if(waitout.active())
  {
    static int count=0;

    //only output the wait times every so many timesteps
    if(++count%100==0)
    {
      char fname[100];
      sprintf(fname,"WaitTimes.%d.%d",d_myworld->size(),d_myworld->myrank());
      wout.open(fname);

      for(map<string,double>::iterator iter=waittimes.begin();iter!=waittimes.end();iter++)
        wout << fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second << " Task:" << iter->first << endl;

      for(map<string,double>::iterator iter=DependencyBatch::waittimes.begin();iter!=DependencyBatch::waittimes.end();iter++)
        wout << fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second << " Task:" << iter->first << endl;
     
      wout.close();
      //waittimes.clear();
      //DependencyBatch::waittimes.clear();
    }
  }

  if( dbg.active()) {
    dbg << me << " MPIScheduler finished\n";
  }
  //pg_ = 0;
}

void
MPIScheduler::emitTime(const char* label)
{
   double time = Time::currentSeconds();
   emitTime(label, time-d_lasttime);
   d_lasttime=time;
}

void
MPIScheduler::emitTime(const char* label, double dt)
{
   d_labels.push_back(label);
   d_times.push_back(dt);
}

void 
MPIScheduler::addToSendList(const MPI_Request& request, int bytes, AfterCommunicationHandler* handler, const string& var)
{
  sends_.add(request, bytes, handler, var, 0);
}

