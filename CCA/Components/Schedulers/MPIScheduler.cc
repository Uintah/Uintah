
#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
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
#include <iomanip>
#include <map>
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

MPIScheduler::MPIScheduler(const ProcessorGroup* myworld, Output* oport)
   : SchedulerCommon(myworld, oport), log(myworld, oport)
{
  d_generation = 0;
  d_lasttime=Time::currentSeconds();
}


void
MPIScheduler::problemSetup(const ProblemSpecP& prob_spec)
{
   log.problemSetup(prob_spec);
}

MPIScheduler::~MPIScheduler()
{
}

SchedulerP
MPIScheduler::createSubScheduler()
{
  MPIScheduler* newsched = new MPIScheduler(d_myworld, m_outPort);
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
}

void
MPIScheduler::compile(const ProcessorGroup* pg, bool init_timestep)
{
  TAU_PROFILE("MPIScheduler::compile()", " ", TAU_USER); 

  dbg << "MPIScheduler starting compile\n";
  if( dts_ )
    delete dts_;

  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  if( useInternalDeps() )
    dts_ = graph.createDetailedTasks( pg, lb, true );
  else
    dts_ = graph.createDetailedTasks( pg, lb, false );

  if(dts_->numTasks() == 0)
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  
  lb->assignResources(*dts_, d_myworld);
  graph.createDetailedDependencies(dts_, lb, pg);
  releasePort("load balancer");

  dts_->assignMessageTags(graph.getTasks());
  int me=pg->myrank();
  dts_->computeLocalTasks(me);
  dts_->createScrublists(init_timestep);

  verifyChecksum();

  dbg << "MPIScheduler finished compile\n";
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
MPIScheduler::initiateTask( const ProcessorGroup  * pg, 
			    DetailedTask          * task,
			    mpi_timing_info_s     & mpi_info,
			    SendRecord            & sends,
			    SendState             & ss,
			    OnDemandDataWarehouse * dws[2],
			    const VarLabel        * reloc_label )
{
  long long communication_flops = 0;
  long long execution_flops = 0;
#ifdef USE_PERFEX_COUNTERS
  long long dummy;
  start_counters(0, 19);
#endif
  
  double recvstart = Time::currentSeconds();
  
  recvMPIData( pg, task, mpi_info, dws );

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
  read_counters(0, &dummy, 19, &communication_flops);
  start_counters(0, 19);
#endif  
  task->doit(pg, dws_[Task::OldDW], dws_[Task::NewDW]);
#ifdef USE_PERFEX_COUNTERS
  read_counters(0, &dummy, 19, &execution_flops);
  start_counters(0, 19);
#endif
  
  TAU_MAPPING_PROFILE_STOP(0);
  TAU_PROFILE_STOP(doittimer);

  task->done();

  double sendstart = Time::currentSeconds();
  sendMPIData( pg, task, mpi_info, sends, ss, dws, reloc_label );

  scrub( task );

  double start = Time::currentSeconds();

  sendsLock.lock(); // Dd... could do better?
  sends.testsome( pg->getComm(), pg->myrank() );
  sendsLock.unlock(); // Dd... could do better?


  mpi_info.totaltestmpi += Time::currentSeconds() - start;

  double dsend = Time::currentSeconds()-sendstart;
  double dtask = sendstart-taskstart;
  double drecv = taskstart-recvstart;

#ifdef USE_PERFEX_COUNTERS
  long long end_communication_flops;
  read_counters(0, &dummy, 19, &end_communication_flops);
  communication_flops += end_communication_flops;
#endif
  
  dbg << pg->myrank() << " Completed task: ";
  printTask(dbg, task); dbg << '\n';

  emitNode(task, Time::currentSeconds(), dsend+dtask+drecv, dtask, execution_flops,
	   communication_flops);

  mpi_info.totalsend += dsend;
  mpi_info.totaltask += dtask;
  mpi_info.totalrecv += drecv;
} // end initiateTask()


void
MPIScheduler::sendMPIData( const ProcessorGroup * pg,
			   DetailedTask         * task,
			   mpi_timing_info_s    & mpi_info,
			   SendRecord & sends,
			   SendState  & ss,
			   OnDemandDataWarehouse * dws[2],
			   const VarLabel        * reloc_label )
{
  if( mixedDebug.active() ) {
    cerrLock.lock();mixedDebug << "sendMPIData - task " << *task << '\n';
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
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      OnDemandDataWarehouse* dw = dws[req->req->dw];
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << " --> sending " << *req << '\n';
	cerrLock.unlock();
      }

      dbg << pg->myrank() << " --> sending " << *req << '\n';
      dw->sendMPI(ss, batch, pg, reloc_label,
		  mpibuff, dws[Task::OldDW], req);
    }
    // Post the send
    if(mpibuff.count()>0){
      ASSERT(batch->messageTag > 0);
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;
#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, pg->getComm());
      mpibuff.pack(pg->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif
      dbg << "Sending message number " << batch->messageTag 
	  << " to " << to << "\n";
      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "Sending message number " << batch->messageTag 
		   << " to " << to << "\n"; cerrLock.unlock();
      }

      MPI_Request requestid;
      MPI_Isend(buf, count, datatype, to, batch->messageTag,
		pg->getComm(), &requestid);
      sendsLock.lock(); // Dd: ??
      sends.add(requestid, mpibuff.takeSendlist());
      sendsLock.unlock(); // Dd: ??
      mpi_info.totalsendmpi += Time::currentSeconds() - start;
    }
  } // end for (DependencyBatch * batch = task->getComputes() )

} // end sendMPIData();

void
MPIScheduler::recvMPIData( const ProcessorGroup * pg,
			   DetailedTask * task, 
			   mpi_timing_info_s & mpi_info,
			   OnDemandDataWarehouse * dws[2] )
{
  TAU_PROFILE("MPIScheduler::recvMPIData()", " ", TAU_USER); 

  RecvRecord recvs;
  // Receive any of the foreign requires

  if( mixedDebug.active() ) {
    cerrLock.lock();mixedDebug << "recvMPIData - task " << *task << '\n';
    cerrLock.unlock();
  }


  list<DependencyBatch*> outstanding_mpi_requests;

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
      outstanding_mpi_requests.push_back( batch );

      if( mixedDebug.active() ) {
	cerrLock.lock();mixedDebug << "Someone else already receiving it\n";
	cerrLock.unlock();
      }

      continue;
    }

    if( mixedDebug.active() ) {
      cerrLock.lock();mixedDebug << "recvMPIData: requesting batch message " 
				 << batch->messageTag << "\n";
      cerrLock.unlock();
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
      OnDemandDataWarehouse* dw = dws[req->req->dw];
      dbg << pg->myrank() << " <-- receiving " << *req << '\n';

      if( mixedDebug.active() ) {
	cerrLock.lock();
	mixedDebug << "-- scheduling receive for " << *req << '\n';
	mixedDebug << "mpibuff size: " << mpibuff.count()<< "\n";
	cerrLock.unlock();
      }

      dw->recvMPI(mpibuff, batch, pg, dws[Task::OldDW], req);

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
      mpibuff.get_type(buf, count, datatype, pg->getComm());
#else
      mpibuff.get_type(buf, count, datatype);
#endif
      int from = batch->fromTask->getAssignedResourceIndex();
      MPI_Request requestid;
      MPI_Irecv(buf, count, datatype, from, batch->messageTag,
		pg->getComm(), &requestid);
      recvs.add(requestid, 
		scinew ReceiveHandler(p_mpibuff, pBatchRecvHandler));
      mpi_info.totalrecvmpi += Time::currentSeconds() - start;

      if( mixedDebug.active() ) {
	cerrLock.lock();mixedDebug << "MPIScheduler: HERE\n";cerrLock.unlock();
      }

    }
    else {
      // Nothing really need to be received, but let everyone else know
      // that it has what is needed (nothing).
      batch->received();
#ifdef USE_PACKING
      // otherwise, it will be deleted after it receives and unpacks
      // the data.
      delete p_mpibuff;
#endif	        
    }
  } // end for

  double start = Time::currentSeconds();

  // This will allow some receives to be "handled" by their
  // AfterCommincationHandler while waiting for others.
  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Requested " << recvs.ids.size() << " receives\n";
    cerrLock.unlock();  
  }

  while( (recvs.ids.size() > 0) && 
	 recvs.waitsome(pg->getComm(), pg->myrank()) )
    {
      if( mixedDebug.active() ) {
	cerrLock.lock();mixedDebug << "waitsome called\n";
	cerrLock.unlock();  
      }
    }

  mpi_info.totalwaitmpi+=Time::currentSeconds()-start;

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "recvMPIData task " <<*task<< " waiting for all receives\n";
    cerrLock.unlock();
  }

  for(list<DependencyBatch*>::iterator iter = outstanding_mpi_requests.begin();
      iter != outstanding_mpi_requests.end(); iter++ ) {
    if( mixedDebug.active() ) {
      cerrLock.lock();
      mixedDebug << "Waiting for other's request for batch message " 
		 << (*iter)->messageTag << " from task " 
		 << *(*iter)->fromTask << endl;
      cerrLock.unlock();
    }

    bool requestNotMade = (*iter)->waitForMPIRequest();
    ASSERT(!requestNotMade);
  }

} // end recvMPIData()

void
MPIScheduler::execute(const ProcessorGroup * pg )
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
  dbg << "MPIScheduler executing\n";
  dts_->initTimestep();

  d_labels.clear();
  d_times.clear();
  emitTime("time since last execute");
  SendState ss;
  // We do not use many Bsends, so this doesn't need to be
  // big.  We make it moderately large anyway - memory is cheap.
  void* old_mpibuffer;
  int old_mpibuffersize;
  MPI_Buffer_detach(&old_mpibuffer, &old_mpibuffersize);
#define MPI_BUFSIZE (10000+MPI_BSEND_OVERHEAD)
  char* mpibuffer = scinew char[MPI_BUFSIZE];
  MPI_Buffer_attach(mpibuffer, MPI_BUFSIZE);

  SendRecord sends;

  int me = pg->myrank();
  makeTaskGraphDoc(dts_, me);

  emitTime("taskGraph output");

  mpi_timing_info_s mpi_info;

  mpi_info.totalreduce = 0;
  mpi_info.totalsend = 0;
  mpi_info.totalrecv = 0;
  mpi_info.totaltask = 0;
  mpi_info.totalreducempi = 0;
  mpi_info.totalsendmpi = 0;
  mpi_info.totalrecvmpi = 0;
  mpi_info.totaltestmpi = 0;
  mpi_info.totalwaitmpi = 0;

  int ntasks = dts_->numLocalTasks();
  int numTasksDone = 0;

  dbg << "Executing " << dts_->numTasks() << " tasks (" << ntasks << " local)\n";

  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Executing " << dts_->numTasks() << " tasks (" 
	       << ntasks << " local)\n";
    cerrLock.unlock();
  }

  while( numTasksDone < ntasks ) {

    if( mixedDebug.active() ) {
      cerrLock.lock();
      mixedDebug << "Num tasks done: " << numTasksDone << "\n";
      cerrLock.unlock();
    }

    DetailedTask * task = dts_->getNextInternalReadyTask();

    dbg << "Got task: " << task->getTask()->getName() << "\n";

    if( mixedDebug.active() ) {
      cerrLock.lock();  
      mixedDebug << "Got task: " << task->getTask()->getName() << "\n";
      cerrLock.unlock();
    }

    numTasksDone++;

    switch(task->getTask()->getType()){
    case Task::Reduction:
      {
	dbg << me << ": Performing reduction: ";
	printTask(dbg, task);
	dbg << '\n';

	if( mixedDebug.active() ) {
	  cerrLock.lock();
	  mixedDebug << "Performing reduction: ";
	  printTask( mixedDebug, task );
	  mixedDebug << "\n";
	  cerrLock.unlock();
	}

#ifdef USE_PERFEX_COUNTERS
	start_counters(0, 19);
#endif
	double reducestart = Time::currentSeconds();
	const Task::Dependency* comp = task->getTask()->getComputes();
	ASSERT(!comp->next);
	OnDemandDataWarehouse* dw = this->dws_[Task::NewDW];
	dw->reduceMPI(comp->var, comp->matls /*task->getMaterials() */,
		      d_myworld);
	double reduceend = Time::currentSeconds();
	long long flop_count=0;
#ifdef USE_PERFEX_COUNTERS
	long long dummy;
	read_counters(0, &dummy, 19, &flop_count);
#endif
	emitNode(task, reducestart, reduceend - reducestart, 0, 0, flop_count);
	mpi_info.totalreduce += reduceend-reducestart;
	mpi_info.totalreducempi += reduceend-reducestart;
      }
      task->done();
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

	initiateTask( pg, task, mpi_info, sends, ss, 
		      this->dws_, reloc_new_posLabel_ );

	if( mixedDebug.active() ) {
	  cerrLock.lock();
	  mixedDebug << "task initiated(MixedSchd) or done(MPIScheduler)\n";
	  cerrLock.unlock();
	}

      } // end case Task::InitialSend or Task::Normal
      break;
    default:
      throw InternalError("Unknown task type");
    } // end switch( task->getTask()->getType() )
  } // end while( numTasksDone < ntasks )

  // wait for all tasks to finish -- i.e. MixedScheduler
  // MPIScheduler will just continue.
  wait_till_all_done();

  
  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Done with all tasks...\n";
    cerrLock.unlock();
  }

  emitTime("MPI send time", mpi_info.totalsendmpi);
  emitTime("MPI Testsome time", mpi_info.totaltestmpi);
  emitTime("Total send time", 
	   mpi_info.totalsend - mpi_info.totalsendmpi - mpi_info.totaltestmpi);
  emitTime("MPI recv time", mpi_info.totalrecvmpi);
  emitTime("MPI wait time", mpi_info.totalwaitmpi);
  emitTime("Total recv time", 
	   mpi_info.totalrecv - mpi_info.totalrecvmpi - mpi_info.totalwaitmpi);
  emitTime("Total task time", mpi_info.totaltask);
  emitTime("Total MPI reduce time", mpi_info.totalreducempi);
  emitTime("Total reduction time", 
	   mpi_info.totalreduce - mpi_info.totalreducempi);

  double time      = Time::currentSeconds();
  double totalexec = time - d_lasttime;

  d_lasttime = time;

  emitTime("Other excution time", totalexec - mpi_info.totalsend -
	   mpi_info.totalrecv - mpi_info.totaltask - mpi_info.totalreduce);


  if( mixedDebug.active() ) {
    cerrLock.lock();
    mixedDebug << "Waiting for all final communications to finish\n";
    cerrLock.unlock();
  }

  // Don't need to lock sends 'cause all threads are done at this point.
  sends.waitall(pg->getComm(), me);
  emitTime("final wait");

  dws_[Task::NewDW]->finalize(); // Dd: I think this is NewDW
  finalizeNodes(me);
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
