

#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/ScatterGatherBase.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <mpi.h>
#include <iomanip>
#include <set>

#ifdef USE_VAMPIR
#include <Packages/Uintah/Core/Parallel/Vampir.h>
#endif //USE_VAMPIR

using namespace std;
using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg("MPIScheduler", false);
static DebugStream timeout("MPIScheduler.timings", false);
#define PARTICLESET_TAG		0x1000000
#define RECV_BUFFER_SIZE_TAG	0x2000000

struct DestType {
   // A unique destination for sending particle sets
   const Patch* patch;
   int matlIndex;
   int dest;
   DestType(int matlIndex, const Patch* patch, int dest)
      : patch(patch), matlIndex(matlIndex), dest(dest)
   {
   }
   bool operator<(const DestType& c) const {
      if(patch < c.patch)
	 return true;
      else if(patch == c.patch){
	 if(matlIndex < c.matlIndex)
	    return true;
	 else if(matlIndex == c.matlIndex)
	    return dest < c.dest;
      }
      // Never reached, but SGI compile complaines?
      return false;
   }
};

struct VarDestType {
   // A unique destination for sending particle sets
   const VarLabel* var;
   const Patch* patch;
   int matlIndex;
   int dest;
   VarDestType(const VarLabel* var, int matlIndex, const Patch* patch, int dest)
      : var(var), patch(patch), matlIndex(matlIndex), dest(dest)
   {
   }
   bool operator<(const VarDestType& c) const {
      VarLabel::Compare comp;
      if(comp(var, c.var))
	 return true;
      else if(!(comp(c.var, var))){
	 if(patch < c.patch)
	    return true;
	 else if(patch == c.patch){
	    if(matlIndex < c.matlIndex)
	       return true;
	    else if(matlIndex == c.matlIndex)
	       return dest < c.dest;
	 }
      }
      return false;
   }
};

static const TypeDescription* specialType;

MPIScheduler::MPIScheduler(const ProcessorGroup* myworld, Output* oport)
   : UintahParallelComponent(myworld), Scheduler(oport), log(myworld, oport)
{
  d_generation = 0;
  if(!specialType)
     specialType = scinew TypeDescription(TypeDescription::ScatterGatherVariable,
				       "DataWarehouse::specialInternalScatterGatherType", false, -1);
  scatterGatherVariable = scinew VarLabel("DataWarehouse::scatterGatherVariable",
				       specialType, VarLabel::Internal);
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
MPIScheduler::initialize()
{
   graph.initialize();
}

void
MPIScheduler::execute(const ProcessorGroup * pc,
		      DataWarehouseP   & old_dw,
		      DataWarehouseP   & dw )
{
#ifdef USE_VAMPIR
   VT_begin(VT_EXECUTE);
#endif

   d_labels.clear();
   d_times.clear();
   emitTime("time since last execute");
   SendState ss;
   // We do not use many Bsends, so this doesn't need to be
   // big.  We make it moderately large anyway.
   void* old_mpibuffer;
   int old_mpibuffersize;
   MPI_Buffer_detach(&old_mpibuffer, &old_mpibuffersize);
#define MPI_BUFSIZE (10000+MPI_BSEND_OVERHEAD)
   char* mpibuffer = new char[MPI_BUFSIZE];
   MPI_Buffer_attach(mpibuffer, MPI_BUFSIZE);

   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   lb->assignResources(graph, d_myworld);
   releasePort("load balancer");

   graph.assignSerialNumbers();

   emitTime("assignSerialNumbers");


#ifdef USE_VAMPIR
   VT_begin(VT_SEND_PARTICLES);
#endif

   // Send particle sets
   int me = pc->myrank();
   vector<Task*>& presort_tasks = graph.getTasks();
   set<DestType> sent;
   for(vector<Task*>::iterator iter = presort_tasks.begin();
       iter != presort_tasks.end(); iter++){
      Task* task = *iter;
      if(task->getAssignedResourceIndex() == me)
	 continue;
      if(task->getType() != Task::Normal)
	 continue;

      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){

	 if(dep->d_dw->isFinalized() && dep->d_patch 
	    && dep->d_var->typeDescription()->getType() == TypeDescription::ParticleVariable){
	    int dest = task->getAssignedResourceIndex();
	    if(dep->d_dw->haveParticleSubset(dep->d_matlIndex, dep->d_patch)){
	       DestType ddest(dep->d_matlIndex, dep->d_patch, dest);
	       if(sent.find(ddest) == sent.end()){
		  int size;
		  ParticleSubset* pset = dep->d_dw->getParticleSubset(dep->d_matlIndex, dep->d_patch);
		  OnDemandDataWarehouse* newdw = dynamic_cast<OnDemandDataWarehouse*>(dw.get_rep());
		  if(!dw)
		     throw InternalError("Wrong Datawarehouse?");
		  newdw->sendParticleSubset(ss, pset, reloc_new_posLabel,
					    dep, task->getPatch(), d_myworld, &size);
		  log.logSend(dep, size, "particleSet size");
		  sent.insert(ddest);
	       }
	    }
	 }
      }
   }

#ifdef USE_VAMPIR
   VT_end(VT_SEND_PARTICLES);
   VT_begin(VT_RECV_PARTICLES);
#endif

   // Recv particle sets
   for(vector<Task*>::iterator iter = presort_tasks.begin();
       iter != presort_tasks.end(); iter++){
      Task* task = *iter;
      if(task->getAssignedResourceIndex() != me)
	 continue;
      if(task->getType() != Task::Normal)
	 continue;

      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){
	 if(dep->d_dw->isFinalized() && dep->d_patch 
	    && dep->d_var->typeDescription()->getType() == TypeDescription::ParticleVariable){
	    if(!dep->d_dw->haveParticleSubset(dep->d_matlIndex, dep->d_patch)){
	       int numParticles;
	       MPI_Status status;
	       ASSERT(dep->d_serialNumber >= 0);
	       MPI_Recv(&numParticles, 1, MPI_INT, MPI_ANY_SOURCE, PARTICLESET_TAG|dep->d_serialNumber, d_myworld->getComm(), &status);
	       dep->d_dw->createParticleSubset(numParticles, dep->d_matlIndex, dep->d_patch);
	    }
	 }
      }
   }

#ifdef USE_VAMPIR
   VT_end(VT_RECV_PARTICLES);
   VT_begin(VT_SEND_INITDATA);
#endif

   emitTime("send/recv particles sets");

   set<VarDestType> varsent;
   // send initial data from old datawarehouse
   vector<MPI_Request> send_ids;
   vector<MPI_Status> send_statii;
   vector<int> indices;
   for(vector<Task*>::iterator iter = presort_tasks.begin();
       iter != presort_tasks.end(); iter++){
      Task* task = *iter;
      if(task->getAssignedResourceIndex() == me)
	 continue;
      if(task->getType() != Task::Normal)
	 continue;

      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){
	 if(dep->d_dw->isFinalized() && dep->d_patch){
	    if(dep->d_dw->exists(dep->d_var, dep->d_matlIndex, dep->d_patch)){
	       VarDestType ddest(dep->d_var, dep->d_matlIndex, dep->d_patch, task->getAssignedResourceIndex());
	       if(varsent.find(ddest) == varsent.end()){
		  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
		  if(!dw)
		     throw InternalError("Wrong Datawarehouse?");
		  MPI_Request requestid;
		  ASSERT(dep->d_serialNumber >= 0);
		  dbg << me << " --> sending initial " << dep->d_var->getName() << " serial " << dep->d_serialNumber << ", to " << dep->d_task->getAssignedResourceIndex() << '\n';
		  int size;
		  dw->sendMPI(ss, dep->d_var, dep->d_matlIndex,
			      dep->d_patch, d_myworld, dep,
			      dep->d_task->getAssignedResourceIndex(),
			      dep->d_serialNumber, &size, &requestid);
		  if(size != -1){
		     log.logSend(dep, size);
		     send_ids.push_back(requestid);
		  }
		  varsent.insert(ddest);
	       }
	    }
	 }
      }
   }
   if(send_ids.size() > 0){
      send_statii.resize(send_ids.size());
      indices.resize(send_ids.size());
      dbg << me << " Calling send Testsome with " << send_ids.size() << " waiters\n";
      int donecount;
      MPI_Testsome((int)send_ids.size(), &send_ids[0], &donecount,
		   &indices[0], &send_statii[0]);
      dbg << me << " Done calling send Testsome with " << send_ids.size() << " waiters and got " << donecount << " done\n";
      if(donecount == (int)send_ids.size() || donecount == MPI_UNDEFINED){
	 send_ids.clear();
      }
   }

#ifdef USE_VAMPIR
   VT_end(VT_SEND_INITDATA);
   VT_begin(VT_RECV_INITDATA);
#endif

   // recv initial data from old datawarhouse
   vector<MPI_Request> recv_ids;
   for(vector<Task*>::iterator iter = presort_tasks.begin();
       iter != presort_tasks.end(); iter++){
      Task* task = *iter;
      if(task->getAssignedResourceIndex() != me)
	 continue;
      if(task->getType() != Task::Normal)
	 continue;

      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){
	 if(dep->d_dw->isFinalized() && dep->d_patch){
	    if(!dep->d_dw->exists(dep->d_var, dep->d_matlIndex, dep->d_patch)){
	       OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
	       if(!dw)
		  throw InternalError("Wrong Datawarehouse?");
	       MPI_Request requestid;
	       ASSERT(dep->d_serialNumber >= 0);
	       dbg << me << " <-- receiving initial " << dep->d_var->getName() << " serial " << dep->d_serialNumber << '\n';
	       int size;
	       dw->recvMPI(ss, old_dw, dep->d_var, dep->d_matlIndex,
			   dep->d_patch, d_myworld, dep,
			   MPI_ANY_SOURCE,
			   dep->d_serialNumber, &size, &requestid);
	       if(size != -1){
		  log.logRecv(dep, size);
		  recv_ids.push_back(requestid);
	       }
	    }
	 }
      }
   }
   if(recv_ids.size() > 0){
      vector<MPI_Status> statii(recv_ids.size());
      dbg << me << " Calling recv waitall with " << recv_ids.size() << " waiters\n";
      MPI_Waitall((int)recv_ids.size(), &recv_ids[0], &statii[0]);
      dbg << me << " Done calling recv waitall with " << recv_ids.size() << " waiters\n";
   }

#ifdef USE_VAMPIR
   VT_end(VT_RECV_INITDATA);
#endif

   emitTime("send initial data");

   vector<Task*> tasks;
   graph.topologicalSort(tasks);

   int ntasks = (int)tasks.size();
   if(ntasks == 0){
      cerr << "WARNING: Scheduler executed, but no tasks\n";
   }

   emitTime("topologicalSort:");

#ifdef USE_VAMPIR
   VT_begin(VT_CHECKSUM);
#endif
   // Compute a simple checksum to make sure that all processes
   // are trying to execute the same graph.  We should do two
   // things in the future:
   //  - make a flag to turn this off
   //  - make the checksum more sophisticated
#if 0
   int checksum = graph.getMaxSerialNumber();
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
#ifdef USE_VAMPIR
   VT_end(VT_CHECKSUM);
#endif
#endif

   emitTime("checksum reduction");

   dbg << "Executing " << ntasks << " tasks\n";

   makeTaskGraphDoc(tasks, me == 0);

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
   for(int i=0;i<ntasks;i++){
      Task* task = tasks[i];
      switch(task->getType()){
      case Task::Reduction:
	 {
	    double reducestart = Time::currentSeconds();
	    const Task::compType& comps = task->getComputes();
	    ASSERTEQ(comps.size(), 1);
	    const Task::Dependency* dep = &comps[0];
	    OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
	    double reducestart2 = Time::currentSeconds();
	    dw->reduceMPI(dep->d_var, dep->d_matlIndex, d_myworld);
	    double reduceend = Time::currentSeconds();
	    time_t t(0);
	    emitNode(tasks[i], t, reduceend - reducestart);
	    totalreduce += reduceend-reducestart;
	    totalreducempi += reduceend-reducestart2;
	 }
	 break;
      case Task::Scatter:
      case Task::Gather:
      case Task::Normal:
	 if(task->getAssignedResourceIndex() == me){

	    dbg << me << " Performing task: " << task->getName();
	    if(task->getPatch())
	       dbg << " on patch " << task->getPatch()->getID();
	    dbg << '\n';
	    double recvstart = Time::currentSeconds();
	    if(task->getType() != Task::Gather){
#ifdef USE_VAMPIR
	      VT_begin(VT_RECV_DEPENDENCIES);
#endif
	       // Receive any of the foreign requires
	       const Task::reqType& reqs = task->getRequires();
	       vector<MPI_Request> recv_ids;
	       for(int r=0;r<(int)reqs.size();r++){
		  const Task::Dependency* req = &reqs[r];
		  if(!req->d_dw->isFinalized()){
		     const Task::Dependency* dep = graph.getComputesForRequires(req);
		     OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(req->d_dw);
		     if(!dep->d_task->isReductionTask() 
			&& dep->d_task->getAssignedResourceIndex() != me
			&& !dw->exists(req->d_var, req->d_matlIndex, req->d_patch)){
			if(dw->isFinalized())
			   throw InternalError("Should not receive from finalized dw!");
			if(!dw)
			   throw InternalError("Wrong Datawarehouse?");
			MPI_Request requestid;
			ASSERT(req->d_serialNumber >= 0);
			dbg << me << " <-- receiving " << req->d_var->getName() << " serial " << req->d_serialNumber << ' ' << *req << '\n';
			int size;
			double start = Time::currentSeconds();
			dw->recvMPI(ss, old_dw, req->d_var, req->d_matlIndex,
				    req->d_patch, d_myworld, req,
				    MPI_ANY_SOURCE,
				    req->d_serialNumber, &size, &requestid);
			totalrecvmpi+=Time::currentSeconds()-start;
			if(size != -1){
			   log.logRecv(req, size);
			   recv_ids.push_back(requestid);
			}
			dbg << "there are now " << recv_ids.size() << " waiters\n";
		     }
		  }
	       }
	       vector<MPI_Status> statii(recv_ids.size());
	       if(recv_ids.size() > 0){
		  dbg << me << " Calling recv(2) waitall with " << recv_ids.size() << " waiters\n";
		  double start = Time::currentSeconds();
		  MPI_Waitall((int)recv_ids.size(), &recv_ids[0], &statii[0]);
		  totalwaitmpi+=Time::currentSeconds()-start;
		  dbg << me << " Done calling recv(2) waitall with " << recv_ids.size() << " waiters\n";
	       }
	    }

#ifdef USE_VAMPIR
	    VT_end(VT_RECV_DEPENDENCIES);
#endif

	    if(task->getType() == Task::Scatter){
	       const Task::compType& comps = task->getComputes();
	       ASSERTEQ(comps.size(), 1);
	       const Task::Dependency* cmp = &comps[0];
	       vector<const Task::Dependency*> reqs;
	       graph.getRequiresForComputes(cmp, reqs);
	       
	       sgargs.dest.resize(reqs.size());
	       sgargs.tags.resize(reqs.size());
	       for(int r=0;r<(int)reqs.size();r++){
		  const Task::Dependency* dep = reqs[r];
		  sgargs.dest[r] = dep->d_task->getAssignedResourceIndex();
		  sgargs.tags[r] = dep->d_serialNumber;
	       }
	    } else if(task->getType() == Task::Gather){
	       const Task::reqType& reqs = task->getRequires();
	       sgargs.dest.resize(reqs.size());
	       sgargs.tags.resize(reqs.size());
	       for(int r=0;r<(int)reqs.size();r++){
		  const Task::Dependency* req = &reqs[r];
		  const Task::Dependency* cmp = graph.getComputesForRequires(req);
		  sgargs.dest[r] = cmp->d_task->getAssignedResourceIndex();
		  sgargs.tags[r] = req->d_serialNumber;
	       }
	    } else {
	       sgargs.dest.resize(0);
	       sgargs.tags.resize(0);
	    }

#ifdef USE_VAMPIR
	    VT_begin(VT_PERFORM_TASK);
#endif
	    dbg << me << " Starting task: " << task->getName();
	    if(task->getPatch())
	       dbg << " on patch " << task->getPatch()->getID() << '\n';
	    double taskstart = Time::currentSeconds();
	    //AuditAllocator(default_allocator);
	    task->doit(pc);
	    //AuditAllocator(default_allocator);
	    double sendstart = Time::currentSeconds();
	    
#ifdef USE_VAMPIR
	    VT_end(VT_PERFORM_TASK);
	    VT_begin(VT_SEND_COMPUTES);
#endif	    

	    if(task->getType() != Task::Scatter){
	       //cerr << me << " !scatter\n";
	       // Send all of the productions
	       const Task::compType& comps = task->getComputes();
	       //cerr << me << " comps.size=" << comps.size() << '\n';
	       for(int c=0;c<(int)comps.size();c++){
		  //cerr << me << " c=" << c << '\n';
		  vector<const Task::Dependency*> reqs;
		  graph.getRequiresForComputes(&comps[c], reqs);
		  set<VarDestType> varsent;
		  //cerr << me << " reqs size=" << reqs.size() << '\n';
		  for(int r=0;r<(int)reqs.size();r++){
		     //cerr << me << "r=" << r << '\n';
		     const Task::Dependency* dep = reqs[r];
		     //cerr << me << "resource=" << dep->d_task->getAssignedResourceIndex() << '\n';
		     //cerr << me << "red=" << dep->d_task->isReductionTask() << '\n';
		     if(dep->d_task->getAssignedResourceIndex() != me
			&& !dep->d_task->isReductionTask()){

			VarDestType ddest(dep->d_var, dep->d_matlIndex, dep->d_patch, dep->d_task->getAssignedResourceIndex());
			if(varsent.find(ddest) == varsent.end()){
			   //cerr << me << " not sent!\n";
			   OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
			   if(!dw)
			      throw InternalError("Wrong Datawarehouse?");
			   MPI_Request requestid;
			   ASSERT(dep->d_serialNumber >= 0);
			   dbg << me << " --> sending " << dep->d_var->getName() << " serial " << dep->d_serialNumber << ", to " << dep->d_task->getAssignedResourceIndex() << " " << *dep << '\n';
			   int size;
			   double start = Time::currentSeconds();
			   dw->sendMPI(ss, dep->d_var, dep->d_matlIndex,
				       dep->d_patch, d_myworld, dep,
				       dep->d_task->getAssignedResourceIndex(),
				       dep->d_serialNumber,
				       &size, &requestid);
			   totalsendmpi += Time::currentSeconds()-start;
			   if(size != -1){
			      log.logSend(dep, size);
			      send_ids.push_back(requestid);
			   }
			   varsent.insert(ddest);
			}
		     }
		  }
	       }
	       if(send_ids.size() > 0){
		  send_statii.resize(send_ids.size());
		  indices.resize(send_ids.size());
		  dbg << me << " Calling send Testsome(2) with " << send_ids.size() << " waiters\n";
		  int donecount;
	   	  double start = Time::currentSeconds();
		  MPI_Testsome((int)send_ids.size(), &send_ids[0], &donecount,
			       &indices[0], &send_statii[0]);
		  totaltestmpi += Time::currentSeconds()-start;
		  dbg << me << " Done calling send Testsome with " << send_ids.size() << " waiters and got " << donecount << " done\n";
		  if(donecount == (int)send_ids.size() || donecount == MPI_UNDEFINED){
		     send_ids.clear();
		  }
	       }
	    }

#ifdef USE_VAMPIR
	    VT_end(VT_SEND_COMPUTES);
#endif	    

	    double dsend = Time::currentSeconds()-sendstart;
	    double dtask = sendstart-taskstart;
	    double drecv = taskstart-recvstart;
	    dbg << me << " Completed task: " << task->getName();
	    if(task->getPatch())
	       dbg << " on patch " << task->getPatch()->getID();
	    dbg << " (recv: " << drecv << " seconds, task: " << dtask << " seconds, send: " << dsend << " seconds)\n";
	    time_t t = time(0);
	    emitNode(tasks[i], t, dsend+dtask+drecv);
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
   emitTime("Other excution time", totalexec-totalsend-totalrecv-totaltask-totalreduce);

   if(send_ids.size() > 0){
      vector<MPI_Status> statii(send_ids.size());
      dbg << me << " Calling send(2) waitall with " << send_ids.size() << " waiters\n";
      MPI_Waitall((int)send_ids.size(), &send_ids[0], &statii[0]);
      dbg << me << " Done calling send(2) waitall with " << send_ids.size() << " waiters\n";
   }
   emitTime("final wait");

   dw->finalize();
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

#ifdef USE_VAMPIR
   VT_end(VT_EXECUTE);
#endif
}

void
MPIScheduler::addTask(Task* task)
{
   graph.addTask(task);
}

DataWarehouseP
MPIScheduler::createDataWarehouse( DataWarehouseP& parent_dw )
{
  int generation = d_generation++;
  return scinew OnDemandDataWarehouse(d_myworld, generation, parent_dw);
}

void
MPIScheduler::scheduleParticleRelocation(const LevelP& level,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw,
					 const VarLabel* old_posLabel,
					 const vector<vector<const VarLabel*> >& old_labels,
					 const VarLabel* new_posLabel,
					 const vector<vector<const VarLabel*> >& new_labels,
					 int numMatls)
{
   reloc_old_posLabel = old_posLabel;
   reloc_old_labels = old_labels;
   reloc_new_posLabel = new_posLabel;
   reloc_new_labels = new_labels;
   reloc_numMatls = numMatls;
   for (int m = 0; m < numMatls; m++ )
     ASSERTEQ(reloc_new_labels[m].size(), reloc_old_labels[m].size());
   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;

      Task* t = scinew Task("MPIScheduler::scatterParticles",
			    patch, old_dw, new_dw,
			    this, &MPIScheduler::scatterParticles);
      for(int m=0;m < numMatls;m++){
	 t->requires( new_dw, old_posLabel, m, patch, Ghost::None);
	 for(int i=0;i<(int)old_labels[m].size();i++)
	    t->requires( new_dw, old_labels[m][i], m, patch, Ghost::None);
      }
      t->computes(new_dw, scatterGatherVariable, 0, patch);
      t->setType(Task::Scatter);
      addTask(t);

      Task* t2 = scinew Task("MPIScheduler::gatherParticles",
			     patch, old_dw, new_dw,
			     this, &MPIScheduler::gatherParticles);
      // Particles are only allowed to be one cell out
      IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
      IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
      Level::selectType neighbors;
      level->selectPatches(l, h, neighbors);
      for(int i=0;i<(int)neighbors.size();i++)
	 t2->requires(new_dw, scatterGatherVariable, 0, neighbors[i], Ghost::None);
      for(int m=0;m < numMatls;m++){
	 t2->computes( new_dw, new_posLabel, m, patch);
	 for(int i=0;i<(int)new_labels[m].size();i++)
	    t2->computes(new_dw, new_labels[m][i], m, patch);
      }
      t2->setType(Task::Gather);
      addTask(t2);
   }
}

namespace Uintah {
   struct MPIScatterMaterialRecord {
      ParticleSubset* relocset;
      vector<ParticleVariableBase*> vars;
   };

   struct MPIScatterRecord : public ScatterGatherBase {
      vector<MPIScatterMaterialRecord*> matls;
   };
} // End namespace Uintah

void
MPIScheduler::scatterParticles(const ProcessorGroup* pc,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
   const Level* level = patch->getLevel();

   // Particles are only allowed to be one cell out
   IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
   IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
   Level::selectType neighbors;
   level->selectPatches(l, h, neighbors);

   vector<MPIScatterRecord*> sr(neighbors.size());
   for(int i=0;i<(int)sr.size();i++)
      sr[i]=0;
   for(int m = 0; m < reloc_numMatls; m++){
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);
      ParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* relocset = scinew ParticleSubset(pset->getParticleSet(),
						    false, -1, 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;
	 if(!patch->getBox().contains(px[idx])){
	    relocset->addParticle(idx);
	 }
      }
      if(relocset->numParticles() > 0){
	 // Figure out where they went...
	 for(ParticleSubset::iterator iter = relocset->begin();
	     iter != relocset->end(); iter++){
	    particleIndex idx = *iter;
	    // This loop should change - linear searches are not good!
	    int i;
	    for(i=0;i<(int)neighbors.size();i++){
	       if(neighbors[i]->getBox().contains(px[idx])){
		  break;
	       }
	    }
	    if(i == (int)neighbors.size()){
	       // Make sure that the particle left the world
	       if(level->containsPoint(px[idx]))
		  throw InternalError("Particle fell through the cracks!");
	    } else {
	       if(!sr[i]){
		  sr[i] = scinew MPIScatterRecord();
		  sr[i]->matls.resize(reloc_numMatls);
		  for(int m=0;m<reloc_numMatls;m++){
		     sr[i]->matls[m]=0;
		  }
	       }
	       if(!sr[i]->matls[m]){
		  MPIScatterMaterialRecord* smr=scinew MPIScatterMaterialRecord();
		  sr[i]->matls[m]=smr;
		  smr->vars.push_back(new_dw->getParticleVariable(reloc_old_posLabel, pset));
		  for(int v=0;v<(int)reloc_old_labels[m].size();v++)
		     smr->vars.push_back(new_dw->getParticleVariable(reloc_old_labels[m][v], pset));
		  smr->relocset = scinew ParticleSubset(pset->getParticleSet(),
						     false, -1, 0);
		  smr->relocset->addReference();
	       }
	       sr[i]->matls[m]->relocset->addParticle(idx);
	    }
	 }
      }
      delete relocset;
   }

   int me = pc->myrank();
   ASSERTEQ(sr.size(), sgargs.dest.size());
   ASSERTEQ(sr.size(), sgargs.tags.size());
   for(int i=0;i<(int)sr.size();i++){
      if(sgargs.dest[i] == me){
	 new_dw->scatter(sr[i], patch, neighbors[i]);
      } else {
	 if(sr[i]){
	    int sendsize = 0;
	    for(int j=0;j<(int)sr[i]->matls.size();j++){
	      int size;
	      MPI_Pack_size(1, MPI_INT, pc->getComm(), &size);
	      sendsize+=size;

	      if (sr[i]->matls[j]) {
		MPIScatterMaterialRecord* mr = sr[i]->matls[j];
		int numP = mr->relocset->numParticles();
		for(int v=0;v<(int)mr->vars.size();v++){
		  ParticleVariableBase* var = mr->vars[v];
		  ParticleVariableBase* var2 = var->cloneSubset(mr->relocset);
		  var2->packsizeMPI(&sendsize, pc, 0, numP);
		  delete var2;
		}
	      }
	    } 
	    char* buf = scinew char[sendsize];
	    int position = 0;
	    for(int j=0;j<(int)sr[i]->matls.size();j++){
	      if (sr[i]->matls[j]) {
		MPIScatterMaterialRecord* mr = sr[i]->matls[j];
		int numP = mr->relocset->numParticles();
		MPI_Pack(&numP, 1, MPI_INT, buf, sendsize, &position, pc->getComm());
		for(int v=0;v<(int)mr->vars.size();v++){
		  ParticleVariableBase* var = mr->vars[v];
		  ParticleVariableBase* var2 = var->cloneSubset(mr->relocset);
		  int numP = mr->relocset->numParticles();
		  var2->packMPI(buf, sendsize, &position, pc, 0, numP);
		  delete var2;
		}
	      } else {
		int numP = 0;
		MPI_Pack(&numP, 1, MPI_INT, buf, sendsize, &position, pc->getComm());
              }   
	    }
	    // This may not be a valid assertion on some systems
	    //ASSERTEQ(position, sendsize);
	    MPI_Send(buf, sendsize, MPI_PACKED, sgargs.dest[i],
		     sgargs.tags[i], pc->getComm());
	    log.logSend(0, sizeof(int), "scatter");
	    delete[] buf;
	 } else {
	    MPI_Send(NULL, 0, MPI_PACKED, sgargs.dest[i],
	        sgargs.tags[i], pc->getComm());
	    log.logSend(0, sizeof(int), "scatter");
	 }
      }
   }
}

void
MPIScheduler::gatherParticles(const ProcessorGroup* pc,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw)
{
   const Level* level = patch->getLevel();

   // Particles are only allowed to be one cell out
   IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
   IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
   Level::selectType neighbors;
   level->selectPatches(l, h, neighbors);

   vector<MPIScatterRecord*> sr;
   vector<int> recvsize(neighbors.size());
   int me = d_myworld->myrank();
   ASSERTEQ((int)sgargs.dest.size(), (int)neighbors.size());
   ASSERTEQ((int)sgargs.tags.size(), (int)neighbors.size());
   vector<char*> recvbuf(neighbors.size());
   vector<int> recvpos(neighbors.size());
   for(int i=0;i<(int)neighbors.size();i++){
      if(patch != neighbors[i]){
	 if(sgargs.dest[i] == me){
	    ScatterGatherBase* sgb = new_dw->gather(neighbors[i], patch);
	    if(sgb != 0){
	       MPIScatterRecord* srr = dynamic_cast<MPIScatterRecord*>(sgb);
	       ASSERT(srr != 0);
	       sr.push_back(srr);
	    }
	 } else {
	    MPI_Status stat;
	    MPI_Probe(sgargs.dest[i], sgargs.tags[i], pc->getComm(), &stat);
	    MPI_Get_count(&stat, MPI_PACKED, &recvsize[i]);
	    log.logRecv(0, sizeof(int), "sg_buffersize");
	    recvpos[i] = 0;
	    recvbuf[i] = scinew char[recvsize[i]];
	    MPI_Recv(recvbuf[i], recvsize[i], MPI_PACKED,
		     sgargs.dest[i], sgargs.tags[i],
		     pc->getComm(), &stat);
	    log.logRecv(0, recvsize[i], "gather");
	 }
      }
   }
   for(int m=0;m<reloc_numMatls;m++){
      // Compute the new particle subset
      vector<ParticleSubset*> subsets;
      vector<ParticleVariableBase*> posvars;

      // Get the local subset without the deleted particles...
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);
      ParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* keepset = scinew ParticleSubset(pset->getParticleSet(),
						   false, -1, 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;
	 if(patch->getBox().contains(px[idx]))
	    keepset->addParticle(idx);
      }
      subsets.push_back(keepset);
      particleIndex totalParticles = keepset->numParticles();
      ParticleVariableBase* pos = new_dw->getParticleVariable(reloc_old_posLabel, pset);
      posvars.push_back(pos);

      // Get the subsets from the neighbors
      particleIndex recvParticles = 0;
      for(int i=0;i<(int)sr.size();i++){
	 if(sr[i]->matls[m]){
	    subsets.push_back(sr[i]->matls[m]->relocset);
	    posvars.push_back(sr[i]->matls[m]->vars[0]);
	    totalParticles += sr[i]->matls[m]->relocset->numParticles();
	 }
      }
      vector<int> counts(neighbors.size());
      for(int i=0;i<(int)neighbors.size();i++){
	 if(sgargs.dest[i] != me){
	   if(recvsize[i]){
	       int n=-1234;
	       MPI_Unpack(recvbuf[i], recvsize[i], &recvpos[i],
			  &n, 1, MPI_INT, pc->getComm());
	       counts[i]=n;
	       totalParticles += n;
	       recvParticles += n;
	    } else {
	       counts[i] = 0;
	    }
	 }
      }

      ParticleVariableBase* newpos = pos->clone();
      ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, m, patch);
      newpos->gather(newsubset, subsets, posvars, recvParticles);

      particleIndex start = totalParticles - recvParticles;
      for(int i=0;i<(int)neighbors.size();i++){
	 if(sgargs.dest[i] != me && counts[i]){
	    newpos->unpackMPI(recvbuf[i], recvsize[i], &recvpos[i], pc,
			      start, counts[i]);
	    start += counts[i];
	 }
      }
      ASSERTEQ(start, totalParticles);

      new_dw->put(*newpos, reloc_new_posLabel);
      delete newpos;

      for(int v=0;v<(int)reloc_old_labels[m].size();v++){
	 vector<ParticleVariableBase*> gathervars;
	 ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);

	 gathervars.push_back(var);
	 for(int i=0;i<(int)sr.size();i++){
	    if(sr[i]->matls[m])
	       gathervars.push_back(sr[i]->matls[m]->vars[v+1]);
	 }
	 ParticleVariableBase* newvar = var->clone();
	 newvar->gather(newsubset, subsets, gathervars, recvParticles);
	 particleIndex start = totalParticles - recvParticles;
	 for(int i=0;i<(int)neighbors.size();i++){
	    if(sgargs.dest[i] != me && counts[i]){
	       newvar->unpackMPI(recvbuf[i], recvsize[i], &recvpos[i], pc,
				 start, counts[i]);
	       start += counts[i];
	    }
	 }
	 ASSERTEQ(start, totalParticles);
	 new_dw->put(*newvar, reloc_new_labels[m][v]);
	 delete newvar;
      }
      delete keepset;
   }
   for(int i=0;i<(int)sr.size();i++){
      for(int m=0;m<reloc_numMatls;m++){
	 if(sr[i]->matls[m]){
	    if(sr[i]->matls[m]->relocset->removeReference())
	       delete sr[i]->matls[m]->relocset;
	    delete sr[i]->matls[m];
	 }
      }
      delete sr[i];
   }
   for(int i=0;i<(int)neighbors.size();i++){
      // This may not be a valid assertion on some systems
      // ASSERTEQ(recvsize[i], recvpos[i]);
      if(sgargs.dest[i] != me && recvsize[i] != 0){
	 delete recvbuf[i];
      }
   }
}

LoadBalancer*
MPIScheduler::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

void
MPIScheduler::releaseLoadBalancer()
{
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
