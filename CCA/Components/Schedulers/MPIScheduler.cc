
#include <TauProfilerForSCIRun.h>

#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/Core/Grid/PackBufferInfo.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <mpi.h>
#include <iomanip>
#include <map>
#include <Packages/Uintah/Core/Parallel/Vampir.h>

// Pack data into a buffer before sending -- testing to see if this
// works better and avoids certain problems possible when you allow
// tasks to modify data that may have a pending send.
#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg("MPIScheduler", false);
static DebugStream timeout("MPIScheduler.timings", false);

namespace Uintah {
  // An MPI_CommunicationRecord keeps track of MPI_Requests and a
  // AfterCommunicationHandler for each of these requests.  By calling
  // testsome or waitall, it will call finishedCommunication(MPI_Comm)
  // on the handlers of all finish requests and then delete these handlers.
  template <class AfterCommunicationHandler>
  struct MPI_CommunicationRecord {
    // returns true while there are more tests to wait for.
    bool testsome(MPI_Comm comm, int me);
    void waitall(MPI_Comm comm, int me);
    void add(MPI_Request id, AfterCommunicationHandler* handler) {
      ids.push_back(id);
      handlers.push_back(handler);
    }
    vector<MPI_Request> ids;
    vector<MPI_Status> statii;
    vector<int> indices;
    vector<AfterCommunicationHandler*> handlers;
  };
  typedef MPI_CommunicationRecord<Sendlist> SendRecord;
  typedef MPI_CommunicationRecord<PackBufferInfo> RecvRecord;

template <class AfterCommunicationHandler>
bool MPI_CommunicationRecord<AfterCommunicationHandler>::
testsome(MPI_Comm comm, int me)
{
  if(ids.size() == 0)
    return false; // no more to test
  statii.resize(ids.size());
  indices.resize(ids.size());
  dbg << me << " Calling testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Testsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  dbg << me << " Done calling testsome with " << ids.size() 
      << " waiters and got " << donecount << " done\n";
  for(int i=0;i<donecount;i++){
    int idx=indices[i];
    if(handlers[idx]){
      handlers[idx]->finishedCommunication(comm);
      delete handlers[idx];
      handlers[idx]=0;
    }
  }
  if(donecount == (int)ids.size() || donecount == MPI_UNDEFINED){
    ids.clear();
    handlers.clear();
    return false; // no more to test
  }
  return true; // more to test
}

template <class AfterCommunicationHandler>
void MPI_CommunicationRecord<AfterCommunicationHandler>::
waitall(MPI_Comm comm, int me)
{
  if(ids.size() == 0)
    return;
  statii.resize(ids.size());
  dbg << me << " Calling waitall with " << ids.size() << " waiters\n";
  MPI_Waitall((int)ids.size(), &ids[0], &statii[0]);
  dbg << me << " Done calling waitall with " 
      << ids.size() << " waiters\n";
  for(int i=0;i<ids.size();i++){
    if(handlers[i]) {
      handlers[i]->finishedCommunication(comm);
      delete handlers[i];
    }
  }
  ids.clear();
  handlers.clear();
}

}

static void printTask(ostream& out, DetailedTask* task)
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

void
MPIScheduler::compile(const ProcessorGroup* pg, bool init_timestep)
{
  dbg << "MPIScheduler starting compile\n";
  if(dt)
    delete dt;
  dt = graph.createDetailedTasks(pg);

  if(dt->numTasks() == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  lb->assignResources(*dt, d_myworld);

  graph.createDetailedDependencies(dt, lb, pg);
  releasePort("load balancer");

  dt->assignMessageTags();
  int me=pg->myrank();
  dt->computeLocalTasks(me);
  dt->createScrublists(init_timestep);
  // Compute a simple checksum to make sure that all processes
  // are trying to execute the same graph.  We should do two
  // things in the future:
  //  - make a flag to turn this off
  //  - make the checksum more sophisticated
  int checksum = dt->getMaxMessageTag();
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
  dbg << "MPIScheduler finished compile\n";
}

#ifdef USE_TAU_PROFILING

map<string,int> taskname_to_id_map;
int unique_id = 9999;

int
create_tau_mapping( const string & taskname )
{
  map<string,int>::iterator iter = taskname_to_id_map.find( taskname );
  if( iter != taskname_to_id_map.end() )
    {
      return (*iter).second;
    }
  else
    {
      string name = taskname;
      TAU_MAPPING_CREATE( name, "[MPIScheduler::execute()]",
			  (TauGroup_t) unique_id, name.c_str(), 0 );
      taskname_to_id_map[ taskname ] = unique_id;
      unique_id++;
      return (unique_id - 1);
    }
}
#endif

void
MPIScheduler::execute(const ProcessorGroup * pg )
{
  ASSERT(dt != 0);
  dbg << "MPIScheduler executing\n";
  VT_begin(VT_EXECUTE);

   TAU_PROFILE("MPIScheduler::execute()", " ", TAU_USER); 
   TAU_PROFILE_TIMER(doittimer, "Task execution", "[MPIScheduler::execute()] " , TAU_USER); 
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

  VT_begin(VT_SEND_PARTICLES);

  SendRecord sends;

  int me = pg->myrank();
  makeTaskGraphDoc(dt, me == 0);

  emitTime("taskGraph output");
  double totalreduce=0;
  double totalsend=0;
  double totalrecv=0;
  double totaltask=0;
  double totalreducempi=0;
  double totalsendmpi=0;
  double totalrecvmpi=0;
  double totaltestmpi=0;
  double totalwaitmpi=0;
  int ntasks = dt->numLocalTasks();
  dbg << "Executing " << dt->numTasks() << " tasks (" << ntasks << " local)\n";

  for(int i=0;i<ntasks;i++){
    DetailedTask* task = dt->localTask(i);
    dbg << "Task " << i << " of " << ntasks << '\n';
    switch(task->getTask()->getType()){
    case Task::Reduction:
      {
	dbg << me << " Performing reduction: ";
	printTask(dbg, task);
	dbg << '\n';
	double reducestart = Time::currentSeconds();
	const Task::Dependency* comp = task->getTask()->getComputes();
	ASSERT(!comp->next);
	OnDemandDataWarehouse* dw = this->dw[Task::NewDW];
	dw->reduceMPI(comp->var, comp->matls /*task->getMaterials() */,
		      d_myworld);
	double reduceend = Time::currentSeconds();
	emitNode(task, reducestart, reduceend - reducestart);
	totalreduce += reduceend-reducestart;
	totalreducempi += reduceend-reducestart;
      }
      break;
    case Task::Normal:
    case Task::InitialSend:
      {
	dbg << me << " Performing task: ";
	printTask(dbg, task);
	dbg << '\n';
	double recvstart = Time::currentSeconds();
	{
	  VT_begin(VT_RECV_DEPENDENCIES);
	  RecvRecord recvs;
	  // Receive any of the foreign requires
	  for(DependencyBatch* batch = task->getRequires();
	      batch != 0; batch = batch->req_next){
	    // Prepare to receive a message
#ifdef USE_PACKING
	    PackBufferInfo* p_mpibuff = scinew PackBufferInfo();
	    PackBufferInfo& mpibuff = *p_mpibuff;
#else
	    BufferInfo mpibuff;
#endif
	    // Create the MPI type
	    for(DetailedDep* req = batch->head; req != 0; req = req->next){
	      OnDemandDataWarehouse* dw = this->dw[req->req->dw];
	      dbg << me << " <-- receiving " << *req << '\n';
	      dw->recvMPI(mpibuff, batch, pg, this->dw[Task::OldDW], req);
	    }
	    // Post the receive
	    if(mpibuff.count()>0){
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
#ifdef USE_PACKING
	      recvs.add(requestid, p_mpibuff);
#else
	      recvs.add(requestid, 0);
#endif	      
	      totalrecvmpi+=Time::currentSeconds()-start;
	    }
#ifdef USE_PACKING
	    else {
	      // otherwise, it will be deleted after it receives and unpacks
	      // the data.
	      delete p_mpibuff;
	    }
#endif	        
	  }

	  double start = Time::currentSeconds();
#ifdef USE_PACKING
	  // This will allow some receives to start unpacking while
	  // waiting for other receives.
	  while (recvs.testsome(pg->getComm(), me))
		 ;
#else
	  recvs.waitall(pg->getComm(), me);
#endif
	  totalwaitmpi+=Time::currentSeconds()-start;
	}
	
	VT_end(VT_RECV_DEPENDENCIES);

	VT_begin(VT_PERFORM_TASK);
	dbg << me << " Starting task: ";
	printTask(dbg, task);
	dbg << '\n';

#ifdef USE_TAU_PROFILING
	int id;
	id = create_tau_mapping( task->getTask()->getName() );
#endif

  TAU_MAPPING_OBJECT(tautimer)
  TAU_MAPPING_LINK(tautimer, (TauGroup_t)id);  // EXTERNAL ASSOCIATION

  TAU_MAPPING_PROFILE_TIMER(doitprofiler, tautimer, 0)

  TAU_PROFILE_START(doittimer);

  TAU_MAPPING_PROFILE_START(doitprofiler,0);

	double taskstart = Time::currentSeconds();
	task->doit(pg, dw[Task::OldDW], dw[Task::NewDW]);
	double sendstart = Time::currentSeconds();
  TAU_MAPPING_PROFILE_STOP(0);

  TAU_PROFILE_STOP(doittimer);
	
	VT_end(VT_PERFORM_TASK);
	VT_begin(VT_SEND_COMPUTES);

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
	  int to = batch->toTask->getAssignedResourceIndex();
	  for(DetailedDep* req = batch->head; req != 0; req = req->next){
	    OnDemandDataWarehouse* dw = this->dw[req->req->dw];
	    dbg << me << " --> sending " << *req << '\n';
	    dw->sendMPI(ss, batch, pg, reloc_new_posLabel,
			mpibuff, this->dw[Task::OldDW], req);
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
	    MPI_Request requestid;
	    MPI_Isend(buf, count, datatype, to, batch->messageTag,
		      pg->getComm(), &requestid);
	    sends.add(requestid, mpibuff.takeSendlist());
	    totalsendmpi+=Time::currentSeconds()-start;
	  }
	}

	scrub(task);

	double start = Time::currentSeconds();
	sends.testsome(pg->getComm(), me);
	totaltestmpi += Time::currentSeconds()-start;
	VT_end(VT_SEND_COMPUTES);

	double dsend = Time::currentSeconds()-sendstart;
	double dtask = sendstart-taskstart;
	double drecv = taskstart-recvstart;
	dbg << me << " Completed task: ";
	printTask(dbg, task);
	dbg << '\n';
	emitNode(task, Time::currentSeconds(), dsend+dtask+drecv);
	totalsend+=dsend;
	totaltask+=dtask;
	totalrecv+=drecv;
      }
      break;
    default:
      throw InternalError("Unknown task type");
    }
  }
  emitTime("MPI send time", totalsendmpi);
  emitTime("MPI Testsome time", totaltestmpi);
  emitTime("Total send time", totalsend-totalsendmpi-totaltestmpi);
  emitTime("MPI recv time", totalrecvmpi);
  emitTime("MPI wait time", totalwaitmpi);
  emitTime("Total recv time", totalrecv-totalrecvmpi-totalwaitmpi);
  emitTime("Total task time", totaltask);
  emitTime("Total MPI reduce time", totalreducempi);
  emitTime("Total reduction time", totalreduce-totalreducempi);
  double time = Time::currentSeconds();
  double totalexec = time-d_lasttime;
  d_lasttime=time;
  emitTime("Other excution time",
	   totalexec-totalsend-totalrecv-totaltask-totalreduce);

  sends.waitall(pg->getComm(), me);
  emitTime("final wait");

  dw[1]->finalize();
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

  VT_end(VT_EXECUTE);
  dbg << "MPIScheduler finished\n";
}

void
MPIScheduler::scheduleParticleRelocation(const LevelP& level,
					 const VarLabel* old_posLabel,
					 const vector<vector<const VarLabel*> >& old_labels,
					 const VarLabel* new_posLabel,
					 const vector<vector<const VarLabel*> >& new_labels,
					 const MaterialSet* matls)
{
  reloc_new_posLabel = new_posLabel;
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  reloc.scheduleParticleRelocation(this, d_myworld, lb, level,
				   old_posLabel, old_labels,
				   new_posLabel, new_labels, matls);
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
