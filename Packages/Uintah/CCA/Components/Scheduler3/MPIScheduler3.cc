#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/Scheduler3/MPIScheduler3.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/DetailedTasks3.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
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

static DebugStream dbg("MPIScheduler", false);
static DebugStream timeout("MPIScheduler.timings", false);
static Mutex sendsLock( "sendsLock" );

static
void
printTask( ostream& out, DetailedTask3* task )
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

MPIScheduler3::MPIScheduler3( const ProcessorGroup * myworld,
			          Output         * oport,
			          MPIScheduler3   * parentScheduler) :
  Scheduler3Common( myworld, oport ),
  log( myworld, oport ), parentScheduler( parentScheduler )
{
  d_lasttime=Time::currentSeconds();
  ss_ = 0;
  rs_ = 0;
  reloc_new_posLabel_=0;
  d_logTimes = 0;
}


void
MPIScheduler3::problemSetup(const ProblemSpecP& prob_spec)
{
  log.problemSetup(prob_spec);
  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if(params){
    params->getWithDefault("logTimes", d_logTimes, false);
  } else {
    d_logTimes = false;
  }
}

MPIScheduler3::~MPIScheduler3()
{
  if( ss_ )
    delete ss_;
  if( rs_ )
    delete rs_;
}

SchedulerP
MPIScheduler3::createSubScheduler()
{
  MPIScheduler3* newsched = new MPIScheduler3(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
MPIScheduler3::verifyChecksum()
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
MPIScheduler3::actuallyCompile()
{
  TAU_PROFILE("MPIScheduler3::actuallyCompile()", " ", TAU_USER); 

  dbg << d_myworld->myrank() << " MPIScheduler3 starting compile\n";
  if( dts_ )
    delete dts_;

  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  dts_ = graph.createDetailedTasks(lb, useInternalDeps() );

  if(dts_->numTasks() == 0)
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  
  lb->assignResources(*dts_);
  graph.createDetailedDependencies(dts_, lb);
  //releasePort("load balancer");

  dts_->assignMessageTags(d_myworld->myrank());

  verifyChecksum();

  dbg << d_myworld->myrank() << " MPIScheduler3 finished compile\n";
}

#ifdef USE_TAU_PROFILING
extern int create_tau_mapping( const string & taskname,
			       const PatchSubset * patches );  // ThreadPool.cc
#endif

void
MPIScheduler3::wait_till_all_done()
{
  if( dbg.active() ) {
    cerrLock.lock();dbg << "MPIScheduler3::wait_till_all_done()\n";
    cerrLock.unlock();
  }
  return;
}

void
MPIScheduler3::initiateTask( DetailedTask3          * task,
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
MPIScheduler3::initiateReduction( DetailedTask3          * task )
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
MPIScheduler3::runTask( DetailedTask3         * task )
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
		    "[MPIScheduler3::initiateTask()] ", TAU_USER); 

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
  task->doit(d_myworld, dws, plain_old_dws);
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
  sends_.testsome( d_myworld );
  sendsLock.unlock(); // Dd... could do better?


  mpi_info_.totaltestmpi += Time::currentSeconds() - stop;

  double dsend = Time::currentSeconds()-sendstart;
  double dtask = sendstart-taskstart;

#ifdef USE_PERFEX_COUNTERS
  long long send_flops;
  read_counters(0, &dummy, 19, &send_flops);
  mpi_info_.totalcommflops += send_flops;
#endif
  
  dbg << d_myworld->myrank() << " Completed task: ";
  printTask(dbg, task); dbg << '\n';

  mpi_info_.totalsend += dsend;
  mpi_info_.totaltask += dtask;
}

void
MPIScheduler3::runReductionTask( DetailedTask3         * task )
{
  const Task::Dependency* comp = task->getTask()->getComputes();
  ASSERT(!comp->next);
  
  PatchBasedDataWarehouse3* dw = dws[comp->mapDataWarehouse()].get_rep();
  dw->reduceMPI(comp->var, comp->reductionLevel, comp->matls);
  task->done(dws);

  dbg << d_myworld->myrank() << " Completed task: ";
  printTask(dbg, task); dbg << '\n';

}

void
MPIScheduler3::postMPISends( DetailedTask3         * task )
{
} // end postMPISends();


void
MPIScheduler3::postMPIRecvs( DetailedTask3 * task, CommRecMPI& recvs,
			    list<DependencyBatch*>& externalRecvs,
			    bool only_old_recvs, int abort_point)
{
} // end postMPIRecvs()

void
MPIScheduler3::processMPIRecvs( DetailedTask3 *, CommRecMPI& recvs,
			       list<DependencyBatch*>& outstandingExtRecvs )
{

} // end processMPIRecvs()

void
MPIScheduler3::execute()
{
   TAU_PROFILE("MPIScheduler3::execute()", " ", TAU_USER); 

   TAU_PROFILE_TIMER(reducetimer, "Reductions", "[MPIScheduler3::execute()] " , TAU_USER); 
   TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[MPIScheduler3::execute()] " , TAU_USER); 
   TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[MPIScheduler3::execute()] " , TAU_USER); 
   TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[MPIScheduler3::execute()] ", 
	TAU_USER); 
   TAU_PROFILE_TIMER(testsometimer, "Test Some", "[MPIScheduler3::execute()] ", 
	TAU_USER); 
   TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[MPIScheduler3::execute()] ", 
	TAU_USER); 
   TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[MPIScheduler3::execute()] ", 
	TAU_USER); 
   TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[MPIScheduler3::execute()] ", 
	TAU_USER); 

  if(dts_ == 0){
    cerr << "MPIScheduler3 skipping execute, no tasks\n";
    return;
  }

  //ASSERT(pg_ == 0);
  //pg_ = pg;
  
  int ntasks = dts_->numLocalTasks();
  dts_->initializeScrubs(dws);
  dts_->initTimestep();

  if(d_logTimes){
    d_labels.clear();
    d_times.clear();
    emitTime("time since last execute");
  }
  if( ss_ )
    delete ss_;
  ss_ = scinew SendState;
  if( rs_ )
    delete rs_;
  rs_ = scinew SendState;
  // We do not use many Bsends, so this doesn't need to be
  // big.  We make it moderately large anyway - memory is cheap.
  void* old_mpibuffer;
  int old_mpibuffersize;
  MPI_Buffer_detach(&old_mpibuffer, &old_mpibuffersize);
#define MPI_BUFSIZE (10000+MPI_BSEND_OVERHEAD)
  char* mpibuffer = scinew char[MPI_BUFSIZE];
  MPI_Buffer_attach(mpibuffer, MPI_BUFSIZE);

  int me = d_myworld->myrank();
  makeTaskGraphDoc(dts_, me);

  if(d_logTimes)
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

  if( dbg.active() ) {
    cerrLock.lock();
    dbg << me << " Executing " << dts_->numTasks() << " tasks (" 
	       << ntasks << " local)\n";
    cerrLock.unlock();
  }

  bool abort=false;

  int abort_point = 987654;
  while( numTasksDone < ntasks ) {

    DetailedTask3 * task = dts_->getNextInternalReadyTask();

    //cerr << "Got task: " << task->getTask()->getName() << "\n";

    numTasksDone++;
    dbg << me << " Initiating task: "; printTask(dbg, task); dbg << '\n';

    switch(task->getTask()->getType()){
    case Task::Reduction:
      if(!abort)
	initiateReduction(task);
      break;
    case Task::Normal:
    case Task::Output:
    case Task::InitialSend:
      {
	initiateTask( task, abort, abort_point );

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

  // wait for all tasks to finish -- i.e. MixedScheduler3
  // MPIScheduler3 will just continue.
  wait_till_all_done();

  if(d_logTimes){
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
  }

  // Don't need to lock sends 'cause all threads are done at this point.
  sends_.waitall(d_myworld);
  ASSERT(sends_.numRequests() == 0);
  if(d_logTimes)
    emitTime("final wait");
  if(restartable){
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size()-1]->timestepRestarted();
    int netrestart;
    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR,
                  d_myworld->getComm());
    if(netrestart)
      dws[dws.size()-1]->restartTimestep();
  }

  finalizeTimestep();

  int junk;
  MPI_Buffer_detach(&mpibuffer, &junk);
  delete[] mpibuffer;
  if(old_mpibuffersize)
    MPI_Buffer_attach(old_mpibuffer, old_mpibuffersize);

  log.finishTimestep();
  if(d_logTimes){
    emitTime("finalize");
    vector<double> d_totaltimes(d_times.size());
    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE,
               MPI_SUM, 0, d_myworld->getComm());
    if(me == 0){
      double total=0;
      for(int i=0;i<(int)d_totaltimes.size();i++)
        total+= d_totaltimes[i];
      for(int i=0;i<(int)d_totaltimes.size();i++){
        timeout << "MPIScheduler3: " << d_labels[i] << ": ";
        int len = (int)(strlen(d_labels[i])+strlen("MPIScheduler3: ")+strlen(": "));
        for(int j=len;j<55;j++)
          timeout << ' ';
        double percent=d_totaltimes[i]/total*100;
        timeout << d_totaltimes[i] << " seconds (" << percent << "%)\n";
      }
      double time = Time::currentSeconds();
      double rtime=time-d_lasttime;
      d_lasttime=time;
      timeout << "MPIScheduler3: TOTAL                                    "
              << total << '\n';
      timeout << "MPIScheduler3: time sum reduction (one processor only): " 
              << rtime << '\n';
    }
  }

  dbg << me << " MPIScheduler3 finished\n";
  //pg_ = 0;
}

void
MPIScheduler3::scheduleParticleRelocation(const LevelP& level,
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
MPIScheduler3::emitTime(char* label)
{
   double time = Time::currentSeconds();
   emitTime(label, time-d_lasttime);
   d_lasttime=time;
}

void
MPIScheduler3::emitTime(char* label, double dt)
{
   d_labels.push_back(label);
   d_times.push_back(dt);
}
