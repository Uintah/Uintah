
#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/DetailedTask.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Containers/Array2.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <mpi.h>
#include <iomanip>
#include <map>
#include <set>
#include <Packages/Uintah/Core/Parallel/Vampir.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg("MPIScheduler", false);
static DebugStream timeout("MPIScheduler.timings", false);
#define RELOCATE_TAG            0x1000000

namespace Uintah {
struct SendRecord {
  void testsome(int me);
  void waitall(int me);
  vector<MPI_Request> ids;
  vector<MPI_Status> statii;
  vector<int> indices;
  void add(MPI_Request id) {
    ids.push_back(id);
  }
};
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

void SendRecord::testsome(int me)
{
  if(ids.size() == 0)
    return;
  statii.resize(ids.size());
  indices.resize(ids.size());
  dbg << me << " Calling send Testsome with " << ids.size() << " waiters\n";
  int donecount;
  MPI_Testsome((int)ids.size(), &ids[0], &donecount,
	       &indices[0], &statii[0]);
  dbg << me << " Done calling send Testsome with " << ids.size() 
      << " waiters and got " << donecount << " done\n";
  if(donecount == (int)ids.size() || donecount == MPI_UNDEFINED){
    ids.clear();
  }
}

void SendRecord::waitall(int me)
{
  if(ids.size() == 0)
    return;
  statii.resize(ids.size());
  dbg << me << " Calling recv waitall with " << ids.size() << " waiters\n";
  MPI_Waitall((int)ids.size(), &ids[0], &statii[0]);
  dbg << me << " Done calling recv waitall with " 
      << ids.size() << " waiters\n";
  ids.clear();
}

MPIScheduler::MPIScheduler(const ProcessorGroup* myworld, Output* oport)
   : SchedulerCommon(myworld, oport), log(myworld, oport)
{
  d_generation = 0;
  d_lasttime=Time::currentSeconds();
  reloc_matls = 0;
}


void
MPIScheduler::problemSetup(const ProblemSpecP& prob_spec)
{
   log.problemSetup(prob_spec);
}

MPIScheduler::~MPIScheduler()
{
  if(reloc_matls && reloc_matls->removeReference())
    delete reloc_matls;
}

void
MPIScheduler::compile(const ProcessorGroup* pg)
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

void
MPIScheduler::execute(const ProcessorGroup * pg )
{
  ASSERT(dt != 0);
  dbg << "MPIScheduler executing\n";
  VT_begin(VT_EXECUTE);

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
	dw->reduceMPI(comp->var, task->getMaterials(), d_myworld);
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
	  SendRecord recvs;
	  // Receive any of the foreign requires
	  for(DependencyBatch* batch = task->getRequires();
	      batch != 0; batch = batch->req_next){
	    // Prepare to receive a message
	    BufferInfo mpibuff;

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
	      mpibuff.get_type(buf, count, datatype);
	      int from = batch->fromTask->getAssignedResourceIndex();
	      MPI_Request requestid;
	      MPI_Irecv(buf, count, datatype, from, batch->messageTag,
			pg->getComm(), &requestid);
	      recvs.add(requestid);
	      totalrecvmpi+=Time::currentSeconds()-start;
	    }
	  }

	  double start = Time::currentSeconds();
	  recvs.waitall(me);
	  totalwaitmpi+=Time::currentSeconds()-start;
	}
	
	VT_end(VT_RECV_DEPENDENCIES);

	VT_begin(VT_PERFORM_TASK);
	dbg << me << " Starting task: ";
	printTask(dbg, task);
	dbg << '\n';
	double taskstart = Time::currentSeconds();
	task->doit(pg, dw[Task::OldDW], dw[Task::NewDW]);
	double sendstart = Time::currentSeconds();
	
	VT_end(VT_PERFORM_TASK);
	VT_begin(VT_SEND_COMPUTES);

	// Send data to dependendents
	for(DependencyBatch* batch = task->getComputes();
	    batch != 0; batch = batch->comp_next){
	  // Prepare to send a message
	  BufferInfo mpibuff;

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
	    mpibuff.get_type(buf, count, datatype);
	    dbg << "Sending message number " << batch->messageTag 
		<< " to " << to << "\n";
	    MPI_Request requestid;
	    MPI_Isend(buf, count, datatype, to, batch->messageTag,
		      pg->getComm(), &requestid);
	    sends.add(requestid);
	    totalsendmpi+=Time::currentSeconds()-start;
	  }
	}

	double start = Time::currentSeconds();
	sends.testsome(me);
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

  sends.waitall(me);
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

static PatchSet* createPerProcessorPatchset(const ProcessorGroup* world,
					    LoadBalancer* lb,
					    const LevelP& level)
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(world->size());
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int proc = lb->getPatchwiseProcessorAssignment(patch, world);
    ASSERTRANGE(proc, 0, world->size());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }
  return patches;
}

void
MPIScheduler::scheduleParticleRelocation(const LevelP& level,
					 const VarLabel* old_posLabel,
					 const vector<vector<const VarLabel*> >& old_labels,
					 const VarLabel* new_posLabel,
					 const vector<vector<const VarLabel*> >& new_labels,
					 const MaterialSet* matls)
{
  reloc_old_posLabel = old_posLabel;
  reloc_old_labels = old_labels;
  reloc_new_posLabel = new_posLabel;
  reloc_new_labels = new_labels;
  if(reloc_matls && reloc_matls->removeReference())
    delete reloc_matls;
  reloc_matls = matls;
  reloc_matls->addReference();
  ASSERTEQ(reloc_old_labels.size(), reloc_new_labels.size());
  int numMatls = (int)reloc_old_labels.size();
  ASSERTEQ(matls->size(), 1);
  ASSERTEQ(numMatls, matls->getSubset(0)->size());
  for (int m = 0; m< numMatls; m++)
    ASSERTEQ(reloc_old_labels[m].size(), reloc_new_labels[m].size());
  Task* t = scinew Task("MPIScheduler::relocParticles",
			this, &MPIScheduler::relocateParticles);
  t->usesMPI();
  t->requires( Task::NewDW, old_posLabel, Ghost::None);
  for(int m=0;m < numMatls;m++){
    MaterialSubset* thismatl = scinew MaterialSubset();
    thismatl->add(m);
    for(int i=0;i<(int)old_labels[m].size();i++)
      t->requires( Task::NewDW, old_labels[m][i], Ghost::None);

    t->computes( new_posLabel, thismatl);
    for(int i=0;i<(int)new_labels[m].size();i++)
      t->computes(new_labels[m][i], thismatl);
  }
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);

  const PatchSet* patches = createPerProcessorPatchset(d_myworld, lb, level);
  addTask(t, patches, matls);
}

namespace Uintah {
  struct MPIScatterRecord {
    const Patch* fromPatch;
    ParticleSubset* sendset;
    MPIScatterRecord(const Patch* fromPatch)
      : fromPatch(fromPatch), sendset(0)
    {
    }
  };
  struct MPIScatterProcessorRecord {
    typedef vector<const Patch*> patchestype;
    patchestype patches;
    void sortPatches();
  };
  struct MPIRecvBuffer {
    MPIRecvBuffer* next;
    char* databuf;
    int bufsize;
    int numParticles;
    MPIRecvBuffer(char* databuf, int bufsize, int numParticles)
      : next(0), databuf(databuf), bufsize(bufsize), numParticles(numParticles)
    {
    }
  };
  class MPIScatterRecords {
  public:
    typedef multimap<pair<const Patch*, int>, MPIScatterRecord*> maptype;
    maptype records;

    typedef map<int, MPIScatterProcessorRecord*> procmaptype;
    procmaptype procs;

    MPIScatterRecord* findRecord(const Patch* from, const Patch* to, int matl,
				 ParticleSubset* pset);
    MPIScatterRecord* findRecord(const Patch* from, const Patch* to, int matl);
    void addNeighbor(LoadBalancer* lb, const ProcessorGroup* pg,
		     const Patch* to);

    typedef map<pair<const Patch*, int>, MPIRecvBuffer*> recvmaptype;
    recvmaptype recvs;
    void saveRecv(const Patch* to, int matl,
		  char* databuf, int bufsize, int numParticles);
    MPIRecvBuffer* findRecv(const Patch* to, int matl);

    ~MPIScatterRecords();
  };
} // End namespace Uintah

void MPIScatterRecords::saveRecv(const Patch* to, int matl,
				 char* databuf, int datasize, int numParticles)
{
  recvmaptype::key_type key(to, matl);
  recvmaptype::iterator iter = recvs.find(key);
  MPIRecvBuffer* record = scinew MPIRecvBuffer(databuf, datasize, numParticles);
  if(iter == recvs.end()){
    recvs[key]=record;
  } else {
    record->next=iter->second;
    recvs[key]=record;
  }
}

MPIRecvBuffer* MPIScatterRecords::findRecv(const Patch* to, int matl)
{
  recvmaptype::iterator iter = recvs.find(make_pair(to, matl));
  if(iter == recvs.end())
    return 0;
  else
    return iter->second;
}

MPIScatterRecord* MPIScatterRecords::findRecord(const Patch* from,
						const Patch* to, int matl,
						ParticleSubset* pset)
{
  pair<maptype::iterator, maptype::iterator> pr = records.equal_range(make_pair(to, matl));
  for(;pr.first != pr.second;pr.first++){
    if(pr.first->second->fromPatch == from)
      break;
  }
  if(pr.first == pr.second){
    MPIScatterRecord* rec = scinew MPIScatterRecord(from);
    rec->sendset = scinew ParticleSubset(pset->getParticleSet(), false, -1, 0);
    records.insert(maptype::value_type(make_pair(to, matl), rec));
    return rec;
  } else {
    return pr.first->second;
  }
}

MPIScatterRecord* MPIScatterRecords::findRecord(const Patch* from,
						const Patch* to, int matl)
{
  pair<maptype::iterator, maptype::iterator> pr = records.equal_range(make_pair(to, matl));
  for(;pr.first != pr.second;pr.first++){
    if(pr.first->second->fromPatch == from)
      break;
  }
  if(pr.first == pr.second){
    return 0;
  } else {
    return pr.first->second;
  }
}

static bool ComparePatches(const Patch* p1, const Patch* p2)
{
  return p1->getID() < p2->getID();
}

void MPIScatterProcessorRecord::sortPatches()
{
  sort(patches.begin(), patches.end(), ComparePatches);
}

void MPIScatterRecords::addNeighbor(LoadBalancer* lb, const ProcessorGroup* pg,
				    const Patch* neighbor)
{
  int toProc = lb->getPatchwiseProcessorAssignment(neighbor, pg);
  procmaptype::iterator iter = procs.find(toProc);
  if(iter == procs.end()){
    MPIScatterProcessorRecord* pr = scinew MPIScatterProcessorRecord();
    procs[toProc]=pr;
    pr->patches.push_back(neighbor);
  } else {
    // This is linear, with the hope that the number of patches per
    // processor will not be huge.
    MPIScatterProcessorRecord* pr = iter->second;
    int i;
    for(i=0;i<(int)pr->patches.size();i++)
      if(pr->patches[i] == neighbor)
	break;
    if(i==(int)pr->patches.size())
      pr->patches.push_back(neighbor);
  }
}

MPIScatterRecords::~MPIScatterRecords()
{
  for(procmaptype::iterator iter = procs.begin(); iter != procs.end(); iter++)
    delete iter->second;
  for(maptype::iterator iter = records.begin(); iter != records.end(); iter++){
    delete iter->second->sendset;
    delete iter->second;
  }
  for(recvmaptype::iterator iter = recvs.begin(); iter != recvs.end(); iter++){
    MPIRecvBuffer* p = iter->second;
    while(p){
      MPIRecvBuffer* next = p->next;
      delete p;
      p=next;
    }
  }
}

void
MPIScheduler::relocateParticles(const ProcessorGroup* pg,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  int total_reloc=0;
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);

  typedef MPIScatterRecords::maptype maptype;
  typedef MPIScatterRecords::procmaptype procmaptype;
  typedef MPIScatterProcessorRecord::patchestype patchestype;

  // First pass: For each of the patches we own, look for particles
  // that left the patch.  Create a scatter record for each one.
  MPIScatterRecords scatter_records;
  int numMatls = (int)reloc_old_labels.size();
  Array2<ParticleSubset*> keepsets(patches->size(), numMatls);
  keepsets.initialize(0);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    // Find all of the neighbors, and add them to a set
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor=neighbors[i];
      scatter_records.addNeighbor(lb, pg, neighbor);
    }

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* relocset = scinew ParticleSubset(pset->getParticleSet(),
						       false, -1, 0);
      ParticleSubset* keepset = scinew ParticleSubset(pset->getParticleSet(),
						      false, -1, 0);

      // Look for particles that left the patch, and put them in relocset
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	if(patch->getBox().contains(px[idx])){
	  keepset->addParticle(idx);
	} else {
	  relocset->addParticle(idx);
	}
      }

      if(relocset->numParticles() == 0){
	delete keepset;
	keepset=pset;
      }
      keepset->addReference();
      keepsets(p, m)=keepset;

      if(relocset->numParticles() > 0){
	total_reloc+=relocset->numParticles();
	// Figure out exactly where they went...
	for(ParticleSubset::iterator iter = relocset->begin();
	    iter != relocset->end(); iter++){
	  particleIndex idx = *iter;
	  // This loop should change - linear searches are not good!
	  // However, since not very many particles leave the patches
	  // and there are a limited number of neighbors, perhaps it
	  // won't matter much
	  int i;
	  for(i=0;i<(int)neighbors.size();i++){
	    if(neighbors[i]->getBox().contains(px[idx])){
	      break;
	    }
	  }
	  if(i == (int)neighbors.size()){
	    // Make sure that the particle really left the world
	    if(level->containsPoint(px[idx]))
	      throw InternalError("Particle fell through the cracks!");
	  } else {
	    // Save this particle set for sending later
	    const Patch* toPatch=neighbors[i];
	    MPIScatterRecord* record = scatter_records.findRecord(patch,
								  toPatch,
								  matl, pset);
	    record->sendset->addParticle(idx);

	    // Optimization: see if other (consecutive) particles
	    // also went to this same patch
	    ParticleSubset::iterator iter2=iter;
	    iter2++;
	    for(;iter2 != relocset->end(); iter2++){
	      particleIndex idx2 = *iter2;
	      if(toPatch->getBox().contains(px[idx2])){
		iter++;
		record->sendset->addParticle(idx2);
	      } else {
		break;
	      }
	    }
	  }
	}
      }
      delete relocset;
    }
  }

  int me = pg->myrank();
  vector<char*> sendbuffers;
  vector<MPI_Request> sendrequests;
  for(procmaptype::iterator iter = scatter_records.procs.begin();
      iter != scatter_records.procs.end(); iter++){
    if(iter->first == me){
      // Local
      continue;
    }
    MPIScatterProcessorRecord* pr = iter->second;
    pr->sortPatches();

    // Go through once to calc the size of the message
    int psize;
    MPI_Pack_size(1, MPI_INT, pg->getComm(), &psize);
    int sendsize=psize; // One for the count of active patches
    int numactive=0;
    vector<int> datasizes;
    for(patchestype::iterator it = pr->patches.begin();
	it != pr->patches.end(); it++){
      const Patch* toPatch = *it;
      for(int matl=0;matl<numMatls;matl++){
	int numVars = (int)reloc_old_labels[matl].size();
	int numParticles=0;
	pair<maptype::iterator, maptype::iterator> pr;
	pr = scatter_records.records.equal_range(make_pair(toPatch, matl));
	for(;pr.first != pr.second; pr.first++){
	  numactive++;
	  int psize;
	  MPI_Pack_size(4, MPI_INT, pg->getComm(), &psize);
	  sendsize += psize; // Patch ID, matl #, # particles, datasize
	  int orig_sendsize=sendsize;
	  MPIScatterRecord* record = pr.first->second;
	  int np = record->sendset->numParticles();
	  numParticles += np;
	  ParticleSubset* pset = old_dw->getParticleSubset(matl, record->fromPatch);
	  ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, pset);
	  ParticleSubset* sendset=record->sendset;
	  posvar->packsizeMPI(&sendsize, pg, sendset);
	  for(int v=0;v<numVars;v++){
	    ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], pset);
	    var->packsizeMPI(&sendsize, pg, sendset);
	  }
	  int datasize=sendsize-orig_sendsize;
	  datasizes.push_back(datasize);
	}
      }
    }
    // Create the buffer for this message
    char* buf = scinew char[sendsize];
    int position=0;

    // And go through it again to pack the message
    int idx=0;
    MPI_Pack(&numactive, 1, MPI_INT, buf, sendsize, &position, pg->getComm());
    for(patchestype::iterator it = pr->patches.begin();
	it != pr->patches.end(); it++){
      const Patch* toPatch = *it;
      for(int matl=0;matl<numMatls;matl++){
	int numVars = (int)reloc_old_labels[matl].size();

	pair<maptype::iterator, maptype::iterator> pr;
	pr = scatter_records.records.equal_range(make_pair(toPatch, matl));
	for(;pr.first != pr.second; pr.first++){
	  int patchid = toPatch->getID();
	  MPI_Pack(&patchid, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  MPI_Pack(&matl, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  MPIScatterRecord* record = pr.first->second;
	  int totalParticles=record->sendset->numParticles();
	  MPI_Pack(&totalParticles, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());
	  int datasize = datasizes[idx];
	  ASSERT(datasize>0);
	  MPI_Pack(&datasize, 1, MPI_INT, buf, sendsize, &position,
		   pg->getComm());

	  int start = position;
	  ParticleSubset* pset = old_dw->getParticleSubset(matl, record->fromPatch);
	  ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, pset);
	  ParticleSubset* sendset=record->sendset;
	  posvar->packMPI(buf, sendsize, &position, pg, sendset);
	  for(int v=0;v<numVars;v++){
	    ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], pset);
	    var->packMPI(buf, sendsize, &position, pg, sendset);
	  }
	  int size=position-start;
	  if(size < datasize){
	    // MPI mis-esimated the size of the message.  For some
	    // reason, mpich does this all the time.  We must pad...
	    int diff=datasize-size;
	    char* junk = scinew char[diff];
	    MPI_Pack(junk, diff, MPI_CHAR, buf, sendsize, &position,
		     pg->getComm());
	    ASSERTEQ(position, start+datasize);
	    delete[] junk;
	  }
	  idx++;
	}
      }
    }
    ASSERT(position <= sendsize);
    // Send (isend) the message
    MPI_Request rid;
    int to=iter->first;
    MPI_Isend(buf, sendsize, MPI_PACKED, to, RELOCATE_TAG,
	      pg->getComm(), &rid);
    sendbuffers.push_back(buf);
    sendrequests.push_back(rid);
  }

  // Receive, and handle the local case too...
  // Foreach processor, post a receive
  vector<char*> recvbuffers(scatter_records.procs.size());

  // I wish that there was an Iprobe_some call, so that we could do
  // this more dynamically...
  int idx=0;
  for(procmaptype::iterator iter = scatter_records.procs.begin();
      iter != scatter_records.procs.end(); iter++, idx++){
    if(iter->first == me){
      // Local - put a placeholder here for the buffer and request
      recvbuffers[idx]=0;
      continue;
    }
    MPI_Status status;
    MPI_Probe(iter->first, RELOCATE_TAG, pg->getComm(), &status);
    int size;
    MPI_Get_count(&status, MPI_PACKED, &size);
    char* buf = scinew char[size];
    recvbuffers[idx]=buf;
    MPI_Recv(recvbuffers[idx], size, MPI_PACKED, iter->first,
	     RELOCATE_TAG, pg->getComm(), &status);

    // Partially unpack
    int position=0;
    int numrecords;
    MPI_Unpack(buf, size, &position, &numrecords, 1, MPI_INT,
	       pg->getComm());
    for(int i=0;i<numrecords;i++){
      int patchid;
      MPI_Unpack(buf, size, &position, &patchid, 1, MPI_INT,
		 pg->getComm());
      const Patch* toPatch = Patch::getByID(patchid);
      ASSERT(toPatch != 0);
      int matl;
      MPI_Unpack(buf, size, &position, &matl, 1, MPI_INT,
		 pg->getComm());
      ASSERTRANGE(matl, 0, numMatls);
      int numParticles;
      MPI_Unpack(buf, size, &position, &numParticles, 1, MPI_INT,
		 pg->getComm());
      int datasize;
      MPI_Unpack(buf, size, &position, &datasize, 1, MPI_INT,
		 pg->getComm());
      char* databuf=buf+position;
      ASSERTEQ(lb->getPatchwiseProcessorAssignment(toPatch, pg), me);
      scatter_records.saveRecv(toPatch, matl,
			       databuf, datasize, numParticles);
      position+=datasize;
    }
  }

  // No go through each of our patches, and do the merge.  Also handle
  // the local case
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      int numVars = (int)reloc_old_labels[matl].size();
      vector<ParticleSubset*> orig_psets;
      vector<ParticleSubset*> subsets;
      ParticleSubset* keepset = keepsets(p, m);
      ASSERT(keepset != 0);
      orig_psets.push_back(old_dw->getParticleSubset(matl, patch));
      subsets.push_back(keepset);
      for(int i=0;i<(int)neighbors.size();i++){
	const Patch* fromPatch=neighbors[i];
	int from = lb->getPatchwiseProcessorAssignment(fromPatch, pg);
	if(from == me){
	  MPIScatterRecord* record = scatter_records.findRecord(fromPatch,
								patch, matl);
	  if(record){
	    ParticleSubset* pset = old_dw->getParticleSubset(matl, record->fromPatch);
	    orig_psets.push_back(pset);
	    subsets.push_back(record->sendset);
	  }
	}
      }
      MPIRecvBuffer* recvs = scatter_records.findRecv(patch, matl);
      ParticleSubset* orig_pset = old_dw->getParticleSubset(matl, patch);
      if(recvs == 0 && subsets.size() == 1 && keepset == orig_pset){
	// carry forward old data
	new_dw->saveParticleSubset(matl, patch, orig_pset);
	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	new_dw->put(*posvar, reloc_new_posLabel);
	for(int v=0;v<numVars;v++){
	  ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[matl][v], orig_pset);
	  new_dw->put(*var, reloc_new_labels[matl][v]);
	}
      } else {
	int totalParticles=0;
	for(int i=0;i<(int)subsets.size();i++)
	  totalParticles+=subsets[i]->numParticles();
	int numRemote=0;
	for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
	  numRemote+=buf->numParticles;
	}
	totalParticles+=numRemote;

	ParticleVariableBase* posvar = new_dw->getParticleVariable(reloc_old_posLabel, orig_pset);
	ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, matl, patch);

	// Merge local portion
	vector<ParticleVariableBase*> invars(subsets.size());
	for(int i=0;i<(int)subsets.size();i++)
	  invars[i]=new_dw->getParticleVariable(reloc_old_posLabel,
						orig_psets[i]);
	ParticleVariableBase* newpos = posvar->clone();
	newpos->gather(newsubset, subsets, invars, numRemote);

	vector<ParticleVariableBase*> vars(numVars);
	for(int v=0;v<numVars;v++){
	  const VarLabel* label = reloc_old_labels[matl][v];
	  ParticleVariableBase* var = new_dw->getParticleVariable(label, orig_pset);
	  for(int i=0;i<(int)subsets.size();i++)
	    invars[i]=new_dw->getParticleVariable(label, orig_psets[i]);
	  ParticleVariableBase* newvar = var->clone();
	  newvar->gather(newsubset, subsets, invars, numRemote);
	  vars[v]=newvar;
	}
	// Unpack MPI portion
	particleIndex idx = totalParticles-numRemote;
	for(MPIRecvBuffer* buf=recvs;buf!=0;buf=buf->next){
	  int position=0;
	  ParticleSubset* unpackset = scinew ParticleSubset(newsubset->getParticleSet(),
							 false, matl, patch);
	  for(int p=0;p<buf->numParticles;p++,idx++)
	    unpackset->addParticle(idx);
	  newpos->unpackMPI(buf->databuf, buf->bufsize, &position,
			    pg, unpackset);
	  for(int v=0;v<numVars;v++)
	    vars[v]->unpackMPI(buf->databuf, buf->bufsize, &position,
			       pg, unpackset);
	  ASSERT(position <= buf->bufsize);
	  delete unpackset;
	}
	ASSERTEQ(idx, totalParticles);
	// Put the data back in the data warehouse
	new_dw->put(*newpos, reloc_new_posLabel);
	delete newpos;
	for(int v=0;v<numVars;v++){
	  new_dw->put(*vars[v], reloc_new_labels[matl][v]);
	  delete vars[v];
	}
      }
      if(keepset->removeReference())
	delete keepset;
    }
  }

  // Communicate the number of particles to processor zero, and
  // print them out
  int alltotal;
  MPI_Reduce(&total_reloc, &alltotal, 1, MPI_INT, MPI_SUM, 0,
	     pg->getComm());
  if(pg->myrank() == 0){
    if(total_reloc != 0)
      cerr << "Particles crossing patch boundaries: " << total_reloc << '\n';
  }

  // Wait to make sure that all of the sends completed
  int numsends = (int)sendrequests.size();
  vector<MPI_Status> statii(numsends);
  MPI_Waitall(numsends, &sendrequests[0], &statii[0]);

  // delete the buffers
  for(int i=0;i<(int)sendbuffers.size();i++)
    delete[] sendbuffers[i];
  for(int i=0;i<(int)recvbuffers.size();i++)
    if(recvbuffers[i])
      delete[] recvbuffers[i];
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
