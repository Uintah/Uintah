
#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <sgi_stl_warnings_off.h>
#include <iomanip>
#include <map>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Parallel/Vampir.h>
#ifdef USE_PERFEX_COUNTERS
#include "counters.h"
#endif

// Pack data into a buffer before sending -- testing to see if this
// works better and avoids certain problems possible when you allow
// tasks to modify data that may have a pending send.
#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg("MPIScheduler", false);
static DebugStream timeout("MPIScheduler.timings", false);

Mutex sendsLock( "sendsLock" );

static
void
printTask( ostream& out, DetailedTask* task )
{
  out << task->getTask()->getName();
  if(task->getPatches()){
    out << " on patches ";
    const PatchSubset* patches = task->getPatches();
    for(int p=0;p<patches->size();p++){
      if(p != 0)
	out << ", ";
      out << patches->get(p)->getID();
    }
  }
}

MPIScheduler::MPIScheduler( const ProcessorGroup * myworld,
			          Output         * oport,
			          MPIScheduler   * parentScheduler) :
  SchedulerCommon( myworld, oport ),
  log( myworld, oport ), parentScheduler( parentScheduler )
{
  d_lasttime=Time::currentSeconds();
  ss_ = 0;
  pg_ = 0;
  reloc_new_posLabel_=0;
}


void
MPIScheduler::problemSetup(const ProblemSpecP& prob_spec)
{
   log.problemSetup(prob_spec);
}

MPIScheduler::~MPIScheduler()
{
  if( ss_ )
    delete ss_;
}

SchedulerP
MPIScheduler::createSubScheduler()
{
  MPIScheduler* newsched = new MPIScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
MPIScheduler::verifyChecksum()
{
  // Compute a simple checksum to make sure that all processes
  // are trying to execute the same graph.  We should do two
  // things in the future:
  //  - make a flag to turn this off
  //  - make the checksum more sophisticated
  int checksum = (int)graph.getTasks().size();
  dbg << "Checking checksum of " << checksum << '\n';
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
  dbg << "Checksum succeeded\n";
}

void
MPIScheduler::actuallyCompile(const ProcessorGroup* pg)
{
  TAU_PROFILE("MPIScheduler::actuallyCompile()", " ", TAU_USER); 

  dbg << pg->myrank() << " MPIScheduler starting compile\n";
  if( dts_ )
    delete dts_;

  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  dts_ = graph.createDetailedTasks( pg, lb, useInternalDeps() );

  if(dts_->numTasks() == 0)
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  
  lb->assignResources(*dts_, d_myworld);
  graph.createDetailedDependencies(dts_, lb, pg);
  releasePort("load balancer");

  dts_->assignMessageTags(pg->myrank());

  verifyChecksum();

  dbg << pg->myrank() << " MPIScheduler finished compile\n";
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
			    bool only_old_recvs, int abort_point )
{
  long long start_total_comm_flops = mpi_info_.totalcommflops;
  long long start_total_exec_flops = mpi_info_.totalexecflops;

  double recvstart = Time::currentSeconds();
#ifdef USE_PERFEX_COUNTERS
  long long dummy, recv_flops;
  start_counters(0, 19);
#endif  
  CommRecMPI recvs;
  list<DependencyBatch*> externalRecvs;
  postMPIRecvs( task, recvs, externalRecvs, only_old_recvs, abort_point );
  processMPIRecvs( task, recvs, externalRecvs );
  if(only_old_recvs)
    return;
#ifdef USE_PERFEX_COUNTERS
  read_counters(0, &dummy, 19, &recv_flops);
  mpi_info_.totalcommflops += recv_flops;
#endif
  
  double drecv = Time::currentSeconds() - recvstart;
  mpi_info_.totalrecv += drecv;

  double start_total_send = mpi_info_.totalsend;
  double start_total_task = mpi_info_.totaltask;
  runTask(task);

  double dsend = mpi_info_.totalsend - start_total_send;
  double dtask = mpi_info_.totaltask - start_total_task;
  emitNode(task, Time::currentSeconds(), dsend+dtask+drecv, dtask,
	   mpi_info_.totalexecflops - start_total_exec_flops,
	   mpi_info_.totalcommflops - start_total_comm_flops);
} // end initiateTask()

void
MPIScheduler::initiateReduction( DetailedTask          * task )
{
  {
#ifdef USE_PERFEX_COUNTERS
    start_counters(0, 19);
#endif
    double reducestart = Time::currentSeconds();

    runReductionTask(task);

    double reduceend = Time::currentSeconds();
    long long flop_count=0;
#ifdef USE_PERFEX_COUNTERS
    long long dummy;
    read_counters(0, &dummy, 19, &flop_count);
#endif
    emitNode(task, reducestart, reduceend - reducestart, 0, 0, flop_count);
    mpi_info_.totalreduce += reduceend-reducestart;
    mpi_info_.totalreducempi += reduceend-reducestart;
  }
}

void
MPIScheduler::runTask( DetailedTask         * task )
{
#ifdef USE_PERFEX_COUNTERS
  long long dummy, exec_flops, send_flops;
#endif
#ifdef USE_TAU_PROFILING
  int id;
  const PatchSubset* patches = task->getPatches();
  id = create_tau_mapping( task->getTask()->getName(), patches );
#endif
  // Should this be here?
  TAU_PROFILE_TIMER(doittimer, "Task execution", 
		    "[MPIScheduler::initiateTask()] ", TAU_USER); 

  TAU_MAPPING_OBJECT(tautimer)
  TAU_MAPPING_LINK(tautimer, (TauGroup_t)id);  // EXTERNAL ASSOCIATION
  TAU_MAPPING_PROFILE_TIMER(doitprofiler, tautimer, 0)
  TAU_PROFILE_START(doittimer);
  TAU_MAPPING_PROFILE_START(doitprofiler,0);

  double taskstart = Time::currentSeconds();
  
#ifdef USE_PERFEX_COUNTERS
  start_counters(0, 19);
#endif
  // TODO - make this not reallocated for each task...
  vector<DataWarehouseP> plain_old_dws(dws.size());
  for(int i=0;i<(int)dws.size();i++)
    plain_old_dws[i] = dws[i].get_rep();
  task->doit(pg_, dws, plain_old_dws);
#ifdef USE_PERFEX_COUNTERS
  read_counters(0, &dummy, 19, &exec_flops);
  mpi_info_.totalexecflops += exec_flops;
  start_counters(0, 19);
#endif
  
  TAU_MAPPING_PROFILE_STOP(0);
  TAU_PROFILE_STOP(doittimer);

  double sendstart = Time::currentSeconds();
  postMPISends( task );
  task->done(dws);
  double stop = Time::currentSeconds();

  sendsLock.lock(); // Dd... could do better?
  sends_.testsome( pg_ );
  sendsLock.unlock(); // Dd... could do better?


  mpi_info_.totaltestmpi += Time::currentSeconds() - stop;

  double dsend = Time::currentSeconds()-sendstart;
  double dtask = sendstart-taskstart;

#ifdef USE_PERFEX_COUNTERS
  long long send_flops;
  read_counters(0, &dummy, 19, &send_flops);
  mpi_info_.totalcommflops += send_flops;
#endif
  
  dbg << pg_->myrank() << " Completed task: ";
  printTask(dbg, task); dbg << '\n';

  mpi_info_.totalsend += dsend;
  mpi_info_.totaltask += dtask;
}

void
MPIScheduler::runReductionTask( DetailedTask         * task )
{
  const Task::Dependency* comp = task->getTask()->getComputes();
  ASSERT(!comp->next);
  
  OnDemandDataWarehouse* dw = dws[comp->mapDataWarehouse()].get_rep();
  dw->reduceMPI(comp->var, comp->reductionLevel, comp->matls, d_myworld);
  task->done(dws);
}

void
MPIScheduler::postMPISends( DetailedTask         * task )
{
  if( mixedDebug.active() ) {
    cerrLock.lock();mixedDebug << "postMPISends - task " << *task << '\n';
    cerrLock.unlock();
  }

  // Send data to dependendents
  for(DependencyBatch* batch = task->getComputes();
      batch != 0; batch = batch->comp_next){

    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << "batch: " << batch << '\n';
      cerrLock.unlock();
    }

    // Prepare to send a message
#ifdef USE_PACKING
    PackBufferInfo mpibuff;
#else
    BufferInfo mpibuff;
#endif
    // Create the MPI type
    int to = batch->toTasks.front()->getAssignedResourceIndex();
    ASSERTRANGE(to, 0, d_myworld->size());
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << " --> sending " << *req << '\n';
	cerrLock.unlock();
      }

      dbg << pg_->myrank() << " --> sending " << *req << " from dw " << dw->getID() << '\n';
      const VarLabel* posLabel;
      OnDemandDataWarehouse* posDW;
      if(!reloc_new_posLabel_ && parentScheduler){
	posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
	posLabel = parentScheduler->reloc_new_posLabel_;
      } else {
	posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
	posLabel = reloc_new_posLabel_;
      }
      MPIScheduler* top = this;
      while(top->parentScheduler) top = top->parentScheduler;
      dw->sendMPI(*top->ss_, batch, pg_, posLabel, mpibuff, posDW, req);
    }
    // Post the send
    if(mpibuff.count()>0){
      ASSERT(batch->messageTag > 0);
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;
     
#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, pg_->getComm());
      mpibuff.pack(pg_->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      dbg << d_myworld->myrank() << " Sending message number " << batch->messageTag 
	  << " to " << to << ", length=" << count << "\n";
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "Sending message number " << batch->messageTag 
		   << " to " << to << "\n"; cerrLock.unlock();
      }

      MPI_Request requestid;
      MPI_Isend(buf, count, datatype, to, batch->messageTag,
		pg_->getComm(), &requestid);
      int bytes = count;
#ifdef USE_PACKING
      MPI_Pack_size(count, datatype, pg_->getComm(), &bytes);
#endif
      sendsLock.lock(); // Dd: ??
      sends_.add(requestid, bytes, mpibuff.takeSendlist());
      sendsLock.unlock(); // Dd: ??
      mpi_info_.totalsendmpi += Time::currentSeconds() - start;
    }
  } // end for (DependencyBatch * batch = task->getComputes() )

} // end postMPISends();

void
MPIScheduler::postMPIRecvs( DetailedTask * task, CommRecMPI& recvs,
			    list<DependencyBatch*>& externalRecvs,
			    bool only_old_recvs, int abort_point)
{
  TAU_PROFILE("MPIScheduler::postMPIRecvs()", " ", TAU_USER); 

  // Receive any of the foreign requires

  if( mixedDebug.active() ) {
    cerrLock.lock();mixedDebug << "postMPIRecvs - task " << *task << '\n';
    cerrLock.unlock();
  }


  map<DependencyBatch*, DependencyBatch*>::const_iterator iter = 
    task->getRequires().begin();
  for( ; iter != task->getRequires().end(); iter++) {
    DependencyBatch* batch = (*iter).first;

    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << "Handle Batch: " 
				 << *(batch->fromTask) << "\n";
      cerrLock.unlock();
    }


    // The first thread that calls this on the batch will return true
    // while subsequent threads calling this will block and wait for
    // that first thread to receive the data.

    if( !batch->makeMPIRequest() ) {
      externalRecvs.push_back( batch );

      if( mixedDebug.active() ) {
	cerrLock.lock();mixedDebug << "Someone else already receiving it\n";
	cerrLock.unlock();
      }

      continue;
    }

    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << "postMPIRecvs: requesting batch message " 
				 << batch->messageTag << "\n";
      cerrLock.unlock();
    }

    if(only_old_recvs){
      if(mixedDebug.active()){
	mixedDebug << "abort analysis: " << batch->fromTask->getTask()->getName()
		   << ", so=" << batch->fromTask->getTask()->getSortedOrder()
		   << ", abort_point=" << abort_point << '\n';
	if(batch->fromTask->getTask()->getSortedOrder() <= abort_point)
	  mixedDebug << "posting MPI recv for pre-abort message " 
		     << batch->messageTag << '\n';
      }
      if(!(batch->fromTask->getTask()->getSortedOrder() <= abort_point))
	continue;
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

    // Create the MPI type
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      dbg << pg_->myrank() << " <-- receiving " << *req << " into dw " << dw->getID() << '\n';

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "-- scheduling receive for " << *req << '\n';
	mixedDebug << "mpibuff size: " << mpibuff.count()<< "\n";
	cerrLock.unlock();
      }

      OnDemandDataWarehouse* posDW;
      if(!reloc_new_posLabel_ && parentScheduler){
	posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
      } else {
	posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
      }
      dw->recvMPI(mpibuff, batch, pg_, posDW, req);
      if (!req->isNonDataDependency()) {
	dts_->setScrubCount(req->req->var, req->matl, req->fromPatch,
			    req->req->mapDataWarehouse(), dws);
      }

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "now mpibuff size: " << mpibuff.count()<< "\n";
	cerrLock.unlock();
      }

    }
    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << "MPIScheduler: here\n";cerrLock.unlock();
    }

    // Post the receive
    if(mpibuff.count()>0){

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "mpibuff.count: " << mpibuff.count() << "\n";
	cerrLock.unlock();
      }

      ASSERT(batch->messageTag > 0);
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;
#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, pg_->getComm());
#else
      mpibuff.get_type(buf, count, datatype);
#endif
      int from = batch->fromTask->getAssignedResourceIndex();
      ASSERTRANGE(from, 0, d_myworld->size());
      MPI_Request requestid;
      dbg << d_myworld->myrank() << " Posting receive for message number " << batch->messageTag << " from " << from << ", length=" << count << "\n";      
      MPI_Irecv(buf, count, datatype, from, batch->messageTag,
		pg_->getComm(), &requestid);
      int bytes = count;
#ifdef USE_PACKING
      MPI_Pack_size(count, datatype, pg_->getComm(), &bytes);
#endif
      recvs.add(requestid, bytes,
		scinew ReceiveHandler(p_mpibuff, pBatchRecvHandler));
      mpi_info_.totalrecvmpi += Time::currentSeconds() - start;

      if( mixedDebug.active() ) {
	cerrLock.lock();mixedDebug << "MPIScheduler: HERE\n";cerrLock.unlock();
      }

    }
    else {
      // Nothing really need to be received, but let everyone else know
      // that it has what is needed (nothing).
      batch->received(pg_);
#ifdef USE_PACKING
      // otherwise, these will be deleted after it receives and unpacks
      // the data.
      delete p_mpibuff;
      delete pBatchRecvHandler;
#endif	        
    }
  } // end for

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Requested " << recvs.numRequests() << " receives\n";
    cerrLock.unlock();  
  }
} // end postMPIRecvs()

void
MPIScheduler::processMPIRecvs( DetailedTask *, CommRecMPI& recvs,
			       list<DependencyBatch*>& outstandingExtRecvs )
{
  TAU_PROFILE("MPIScheduler::processMPIRecvs()", " ", TAU_USER);

  // Should only have external receives in the MixedScheduler version which
  // shouldn't use this function.
  ASSERT(outstandingExtRecvs.empty());
  double start = Time::currentSeconds();

  // This will allow some receives to be "handled" by their
  // AfterCommincationHandler while waiting for others.  
  while( (recvs.numRequests() > 0)) {
    bool keep_waiting = recvs.waitsome(pg_);
    if (!keep_waiting)
      break;
  }

  mpi_info_.totalwaitmpi+=Time::currentSeconds()-start;

} // end processMPIRecvs()

void
MPIScheduler::execute(const ProcessorGroup * pg)
{
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

  if(dts_ == 0){
    cerr << "MPIScheduler skipping execute, no tasks\n";
    return;
  }

  ASSERT(pg_ == 0);
  pg_ = pg;
  
  int ntasks = dts_->numLocalTasks();
  if(dbg.active()){
    dbg << "MPIScheduler executing " << dts_->numTasks() << " tasks (" << ntasks << " local), ";
    if (numOldDWs != 0){
      dbg << "from DWs: ";
      for(int i=0;i<numOldDWs;i++){
	if(dws[i])
	  dbg << dws[i]->getID() << ", ";
	else
	  dbg << "Null, ";
      }
    }
    if(dws.size()-numOldDWs>1){
      dbg << "intermediate DWs: ";
      for(unsigned int i=numOldDWs;i<dws.size()-1;i++)
	dbg << dws[i]->getID() << ", ";
    }
    if(dws[dws.size()-1])
      dbg << " to DW: " << dws[dws.size()-1]->getID();
    else
      dbg << " to DW: Null";
    dbg << "\n";
  }
  dts_->initializeScrubs(dws);
  dts_->initTimestep();

  d_labels.clear();
  d_times.clear();
  emitTime("time since last execute");
  if( ss_ )
    delete ss_;
  ss_ = scinew SendState;
  // We do not use many Bsends, so this doesn't need to be
  // big.  We make it moderately large anyway - memory is cheap.
  void* old_mpibuffer;
  int old_mpibuffersize;
  MPI_Buffer_detach(&old_mpibuffer, &old_mpibuffersize);
#define MPI_BUFSIZE (10000+MPI_BSEND_OVERHEAD)
  char* mpibuffer = scinew char[MPI_BUFSIZE];
  MPI_Buffer_attach(mpibuffer, MPI_BUFSIZE);

  int me = pg_->myrank();
  makeTaskGraphDoc(dts_, me);

  emitTime("taskGraph output");

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

  // TEMPORARY
//  cerrLock.lock();
//  cerr << "Executing " << dts_->numTasks() << " tasks (" << ntasks << " local)\n";  
//  cerrLock.unlock();
  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Executing " << dts_->numTasks() << " tasks (" 
	       << ntasks << " local)\n";
    cerrLock.unlock();
  }

  bool abort=false;
  int abort_point = 987654;
  while( numTasksDone < ntasks ) {

    if( mixedDebug.active() ) {
      cerrLock.lock();
      mixedDebug << "Num tasks done: " << numTasksDone << "\n";
      cerrLock.unlock();
    }

    DetailedTask * task = dts_->getNextInternalReadyTask();

    //cerr << "Got task: " << task->getTask()->getName() << "\n";

    if( mixedDebug.active() ) {
      cerrLock.lock();  
      mixedDebug << "Got task: " << task->getTask()->getName() << "\n";
      cerrLock.unlock();
    }

    numTasksDone++;

    switch(task->getTask()->getType()){
    case Task::Reduction:
      dbg << me << ": Performing reduction: ";
      printTask(dbg, task);
      dbg << '\n';
      
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "Initiating reduction: ";
	printTask( mixedDebug, task );
	mixedDebug << "\n";
	cerrLock.unlock();
      }
      if(!abort)
	initiateReduction(task);
      break;
    case Task::Normal:
    case Task::InitialSend:
      {
	dbg << me << " Initiating task: "; printTask(dbg, task); dbg << '\n';
	if( mixedDebug.active() ) {
	  cerrLock.lock();  
	  mixedDebug<< " Initiating task: ";
	  printTask(mixedDebug, task); mixedDebug << '\n';
	  cerrLock.unlock();  
	}

	initiateTask( task, abort, abort_point );

	if( mixedDebug.active() ) {
	  cerrLock.lock();
	  mixedDebug << "task initiated(MixedSchd) or done(MPIScheduler)\n";
	  cerrLock.unlock();
	}

      } // end case Task::InitialSend or Task::Normal
      break;
    default:
      SCI_THROW(InternalError("Unknown task type"));
    } // end switch( task->getTask()->getType() )

    if(!abort && dws[dws.size()-1] && dws[dws.size()-1]->timestepAborted()){
      abort = true;
      abort_point = task->getTask()->getSortedOrder();
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
    }
  } // end while( numTasksDone < ntasks )

  // wait for all tasks to finish -- i.e. MixedScheduler
  // MPIScheduler will just continue.
  wait_till_all_done();

  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Done with all tasks...\n";
    cerrLock.unlock();
  }

  emitTime("MPI send time", mpi_info_.totalsendmpi);
  emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
  emitTime("Total send time", 
	   mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
  emitTime("MPI recv time", mpi_info_.totalrecvmpi);
  emitTime("MPI wait time", mpi_info_.totalwaitmpi);
  emitTime("Total recv time", 
	   mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
  emitTime("Total task time", mpi_info_.totaltask);
  emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
  emitTime("Total reduction time", 
	   mpi_info_.totalreduce - mpi_info_.totalreducempi);

  double time      = Time::currentSeconds();
  double totalexec = time - d_lasttime;

  d_lasttime = time;

  emitTime("Other excution time", totalexec - mpi_info_.totalsend -
	   mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);


  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Waiting for all final communications to finish\n";
    cerrLock.unlock();
  }

  // Don't need to lock sends 'cause all threads are done at this point.
  sends_.waitall(pg_);
  emitTime("final wait");

  finalizeTimestep();

  int junk;
  MPI_Buffer_detach(&mpibuffer, &junk);
  delete[] mpibuffer;
  if(old_mpibuffersize)
    MPI_Buffer_attach(old_mpibuffer, old_mpibuffersize);

  log.finishTimestep();
  emitTime("finalize");
  vector<double> d_totaltimes(d_times.size());
  MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE,
	     MPI_SUM, 0, d_myworld->getComm());
  if(me == 0){
    double total=0;
    for(int i=0;i<(int)d_totaltimes.size();i++)
      total+= d_totaltimes[i];
    for(int i=0;i<(int)d_totaltimes.size();i++){
      timeout << "MPIScheduler: " << d_labels[i] << ": ";
      int len = (int)(strlen(d_labels[i])+strlen("MPIScheduler: ")+strlen(": "));
      for(int j=len;j<55;j++)
	timeout << ' ';
      double percent=d_totaltimes[i]/total*100;
      timeout << d_totaltimes[i] << " seconds (" << percent << "%)\n";
    }
    double time = Time::currentSeconds();
    double rtime=time-d_lasttime;
    d_lasttime=time;
    timeout << "MPIScheduler: TOTAL                                    "
	    << total << '\n';
    timeout << "MPIScheduler: time sum reduction (one processor only): " 
	    << rtime << '\n';
  }

  dbg << "MPIScheduler finished\n";
  pg_ = 0;
}

void
MPIScheduler::scheduleParticleRelocation(const LevelP& level,
					 const VarLabel* old_posLabel,
					 const vector<vector<const VarLabel*> >& old_labels,
					 const VarLabel* new_posLabel,
					 const vector<vector<const VarLabel*> >& new_labels,
					 const VarLabel* particleIDLabel,
					 const MaterialSet* matls)
{
  reloc_new_posLabel_ = new_posLabel;
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  reloc_.scheduleParticleRelocation( this, d_myworld, lb, level,
				     old_posLabel, old_labels,
				     new_posLabel, new_labels,
				     particleIDLabel, matls );
  releasePort("load balancer");
}

void
MPIScheduler::emitTime(char* label)
{
   double time = Time::currentSeconds();
   emitTime(label, time-d_lasttime);
   d_lasttime=time;
}

void
MPIScheduler::emitTime(char* label, double dt)
{
   d_labels.push_back(label);
   d_times.push_back(dt);
}
