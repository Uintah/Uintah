/*
 * General code for managing simplification
 *
 * Peter-Pike Sloan
 */

#include "GenSimpObj.h"

#include <iostream.h>
#include <Malloc/Allocator.h>

GenSimpObj::GenSimpObj():
keepRecords_(0),error_queue(0),ErrFunc(0),qPercent_(0),qFlags_(0),
qBadWeight_(1),keep_time(0),dump_size(0),chunk_proc(10),numProc(1),
qfill_init(0),FlushQ(0),WakeMain(0)
{

}

void GenSimpObj::BuildEdgeList()
{
  // build the edge array from the hash table...

  edgelst.setsize(edges.size());
  
  FastHashTableIter<AugEdge> augiter(&edges);
  int index=0;
  
  for(augiter.first();augiter.ok();++augiter) {
    edgelst[index] = augiter.get_key();
    edgelst[index]->id = index;
    index++;
  }	
}

class PriorityQueueFiller : public Task {
public:
  GenSimpObj *owner;
  virtual int body(int);
  PriorityQueueFiller(GenSimpObj *);
};

PriorityQueueFiller::PriorityQueueFiller(GenSimpObj *own)
:Task("Asynchronus PQueue Filler"),owner(own)
{

}

void GenSimpObj::DoPQueueStuff()
{
  FlushQ.down(); // block on this semaphore...

  edgeQlock.lock();  // lock down the queue

  while(!edgeQ.is_empty()) {
    workQ.add(edgeQ.pop());
  }
  
  edgeQlock.unlock(); // release the lock

  if (!workQ.size())  // just poping off a extra one...
    return;

  // dump this workQ into the priority queue

  int sentinel_hit=0; // check for sentinel in the queue

  for(int i=0;i<workQ.size();i++) {
    if (workQ[i]) {
      for(int j=0;j<workQ[i]->size();j++) {
	int cedge = (*workQ[i])[j];
	error_queue->replace(cedge+1,edgelst[cedge]->weight);
      }
      workQ[i]->resize(0); // zero it out - ready for next user...
    } else {
      sentinel_hit=1; // found it!
    }
  }

  freeQlock.lock();

  for(i=0;i<workQ.size();i++) {
    if (workQ[i]) {
      freeQ.append(workQ[i]);
    }
  }
  
  freeQlock.unlock();

  workQ.resize(0); // reset this

  if (sentinel_hit) { // wake up the main thread...
    WakeMain.up();
  }
}

int PriorityQueueFiller::body(int )
{
  while(1) {
    owner->DoPQueueStuff();  // just snarf stuff...
  }
}

// C routine to aid in filling the priority queue

static void do_parallel_q_fill(void *obj, int proc)
{
  GenSimpObj *sobj = (GenSimpObj*) obj;
  sobj->parallel_q_fill(proc);
}
static void do_parallel_q_refill(void *obj, int proc)
{
  GenSimpObj *sobj = (GenSimpObj*) obj;
  sobj->parallel_q_refill(proc);
}

// just serial code - for debugging...

void GenSimpObj::SFillQ(int useOld)
{
  if (!useOld) {
    int start=0,end=edgelst.size();
    int ngood=0,nbad=0,nhosed=0;

    for(int cedge=start; cedge < end; cedge++) {
      if (mpieces[0]->SetEdge(cedge)) {
	Point p;
	double err;
	double sv;

	if (ErrFunc->MinError(p,sv,err,mpieces[0])) {	
	  edgelst[cedge]->p = p;
	  edgelst[cedge]->v = sv;
	  edgelst[cedge]->weight = err;

	  error_queue->replace(cedge+1,edgelst[cedge]->weight);
	  ++ngood;
	} else {
	  ++nbad;
	}
      } else {
	++nhosed;
      }
    }

    cerr << "Did initial fill: " << ngood << " " << nbad;
    cerr << " " << nhosed << endl;

  } else {
    for(int i=0; i < oEdges_.size(); i++) {
      int cedge = oEdges_[i];

      if (mpieces[0]->SetEdge(cedge)) {
	Point p;
	double err;
	double sv;
	if (ErrFunc->MinError(p,sv,err,mpieces[0])) {	
	  edgelst[cedge]->p = p;
	  edgelst[cedge]->v = sv;
	  edgelst[cedge]->weight = err;

	  error_queue->replace(cedge+1,edgelst[cedge]->weight);
	}
      }

    }
  }
}

void GenSimpObj::parallel_q_fill(int proc)
{
  int start,end;

  // gobble as much as you can...
  while(1) {
    
    chunkLock.lock();
    if (curChunk == chunkPos.size()) {
      chunkLock.unlock();
      return;  // this thread is finished...
    }
  
    start = chunkPos[curChunk++];
    end = chunkPos[curChunk];
    
    if (curChunk == chunkPos.size()-1) {
      curChunk++;
    }
    chunkLock.unlock();
    
    Array1<int> *build_list=0;
    
    for(int cedge=start; cedge < end; cedge++) {
      if (mpieces[proc]->SetEdge(cedge)) {
	Point p;
	double err;
	double sv;
	int j;
	if (ErrFunc->TrySimp(p,sv,err,mpieces[proc],j)) {
	  edgelst[cedge]->p = p;
	  edgelst[cedge]->v = sv;
	  edgelst[cedge]->weight = err;

//	  cerr << "Doing a edge: " << p << " - " << err << endl;

	  if (!build_list) { // get a new one...
	    freeQlock.lock();
	    if (freeQ.is_empty()) {
	      freeQlock.unlock();
	      build_list = new Array1<int>;
	    } else {
	      build_list = freeQ.pop(); // just grab one...
	      freeQlock.unlock();
	    }
	  }

	  build_list->add(cedge);

	  if (build_list->size() > dump_size) {
	    edgeQlock.lock();
	    edgeQ.append(build_list);
	    edgeQlock.unlock();

	    build_list=0; // get a new one next time

	    FlushQ.up();  // Release the Hounds!
	  }
	}
      }
    }

    if ((build_list) && build_list->size()) {

      edgeQlock.lock();
      edgeQ.append(build_list);
      edgeQlock.unlock();

      FlushQ.up();  // Release the Hounds!
    }
  }
}


void GenSimpObj::parallel_q_refill(int proc)
{
  int end;

  // gobble as much as you can...
  while(1) {
    
    chunkLock.lock();
    if (curChunk == oEdges_.size()) {
      chunkLock.unlock();
      return;  // this thread is finished...
    }
  
    end = oEdges_[curChunk++];

    chunkLock.unlock();
    
    Array1<int> *build_list=0;
    
    int cedge = end;
    if (mpieces[proc]->SetEdge(cedge)) {
      Point p;
      double err;
      double sv;
      int j;
      if (ErrFunc->TrySimp(p,sv,err,mpieces[proc],j)) {
	edgelst[cedge]->p = p;
	edgelst[cedge]->v = sv;
	edgelst[cedge]->weight = err;

//	cerr << "ReDoing a edge: " << p << " - " << err << endl;

	if (!build_list) { // get a new one...
	  freeQlock.lock();
	  if (freeQ.is_empty()) {
	    freeQlock.unlock();
	    build_list = new Array1<int>;
	  } else {
	    build_list = freeQ.pop(); // just grab one...
	    freeQlock.unlock();
	  }
	}
	
	build_list->add(cedge);
	
	if (build_list->size() > dump_size) {
	  edgeQlock.lock();
	  edgeQ.append(build_list);
	  edgeQlock.unlock();
	  
	  build_list=0; // get a new one next time
	  
	  FlushQ.up();  // Release the Hounds!
	}
      }
    }

    if ((build_list) && build_list->size()) {
      
      edgeQlock.lock();
      edgeQ.append(build_list);
      edgeQlock.unlock();
      
      FlushQ.up();  // Release the Hounds!
    }
  }
}

// fills the priority queue

void GenSimpObj::FillQ(int useOld)
{
 
  if (!error_queue) {
    error_queue = scinew PQueue(edgelst.size());
  }

  if (numProc > 1) {
    curChunk=0;
    chunkPos.resize(0);

    int edgesize = edgelst.size();
    if (useOld) { 
      edgesize = oEdges_.size();
      if (!edgesize)
	return;
    }

    int interval =  ((1.0/chunk_proc)*edgesize)/numProc;

    if (!interval)
      interval = 1;

    int curpos=0;
    int curedge=0;
    
    if (!useOld) {

      while(curedge < edgesize) {
	chunkPos.add(curedge);
	curedge += interval;
	curpos++;
      }
      if ((curedge-interval) != (edgesize-1)) { // scale back...
	chunkPos.add(edgesize-1);
	curpos++;
      }

      --curpos; // so it works...
    } else {

      // just work directly out of the oEdges_ array...

    }

    if (!qfill_init) {
      qfill_init = 1;
      PriorityQueueFiller *qfill = scinew PriorityQueueFiller(this);
      qfill->activate(1);
    }	

    if (!useOld)
      Task::multiprocess(numProc, do_parallel_q_fill,this);
    else
      Task::multiprocess(numProc, do_parallel_q_refill,this);
    

    // this return when the chunkPos list has been done...

    edgeQlock.lock();
    edgeQ.append(0);  // stick in the sentinel...
    edgeQlock.unlock();

    FlushQ.up(); // tell it to grab this piece...
   
    WakeMain.down(); // wake up when the Q filler is complete...
  } else { //just do serial code
    // just do serial code...
    SFillQ(useOld);
  }	
  // this means everything has finished...

//  cerr << "Finished Filling a PQ!\n";
}


int GenSimpObj::PopQE(int nelem)
{
  
  cerr << " Trying pop....\n";

  int tdone=0;
  while(tdone < nelem) {
    int edge = error_queue->remove()-1;

    if (edge == -1) {
      cerr << "Woah - -1 returned?\n";
      return 0;
    }

    if (mpieces[0]->SetEdge(edge)) {
      int ndone=0;
      ndone += mpieces[0]->TryFlush(1);
      tdone += ndone;
      if (!ndone) {
	cerr << "Woah - flush bailed\n";
      } else { // ok make sure the mesh is still ok...
	// have the error functional do a post-collapse...

	ErrFunc->PostCollapseStuff(mpieces[0]);

	// and re-Q the neccesary guys...
	FillQ(1); // do them all...
	oEdges_.resize(0); // clear them out...
      }
    } else {
      cerr << "Invalid guy in the Q...\n";
    }
  }
  return tdone;
}

int GenSimpObj::PopQ(int npop)
{
  cerr << " Trying pop....\n";
  int tdone=0;
  for(int i=0;i<npop;i++) {
    int edge = error_queue->remove()-1;

    if (edge == -1) {
      cerr << "Woah - -1 returned?\n";
      return 0;
    }

    if (mpieces[0]->SetEdge(edge)) {
      // cerr << "Did SetEdge\n";
      int ndone=0;
      ndone += mpieces[0]->TryFlush(1);
      tdone += ndone;
      if (!ndone) {
	cerr << "Woah - flush bailed\n";
      } else { // ok make sure the mesh is still ok...
	ErrFunc->PostCollapseStuff(mpieces[0]);

	// and re-Q the neccesary guys...
	FillQ(1); // do them all...
	oEdges_.resize(0); // clear them out...
      }
    } else {
      cerr << "Invalid guy in the Q...\n";
    }
    
    // here is where you see if you need to "encode" edges...
    // just do them all for now...

    

  }
  return tdone;
}

int GenSimpObj::PopQ(double percent)
{
  // this is really simple...

  return PopQE(percent*error_queue->size());
}

/* 
 * this code builds an array to renumber all of the nodes...
 *
 * Every edge collapse adds a single node...
 *
 * So the base mesh has norig - ncollapse nodes in it...
 * elements stay the same...
 */

void GenSimpObj::recomputeIndex(int norig, int nelem)
{
  vRemap_.resize(norig + vHiearchy_.size());

  vRemap_.initialize(0);

  eRemap_.resize(nelem); // always the same...

  eRemap_.initialize(0); // non allocated yet...

  // now remove all of the vertices from the collapses...

  for(int i=0;i<vHiearchy_.size();i++) {
    vRemap_[vHiearchy_[i].s] = -1;
    vRemap_[vHiearchy_[i].u] = -1;
    for(int j=0;j<vHiearchy_[i].ei.size();j++) {
      eRemap_[vHiearchy_[i].ei[j]] = -1; // clear out the elements...
    }
  }

  // the vertices you have left are all part of the
  // base mesh...

  // now all of the vertices that are not -1
  // are unmodified originals...

  int vcntr=0;
  for(i=0;i<vRemap_.size();i++) {
    if (vRemap_[i] != -1) {
      vRemap_[i] = vcntr++;
    }
  }

  int ecntr=0;

  for(i=0; i<eRemap_.size();i++) {
    if (eRemap_[i] != -1)
      eRemap_[i] = ecntr++;
  }

  // now go through the records in reverse order
  // building up the new verts as you go...

  for(i=vHiearchy_.size()-1; i >=0; i--) {
    vRemap_[vHiearchy_[i].s] = vcntr++;
    vRemap_[vHiearchy_[i].u] = vcntr++;

    // also do the elements...
    for(int j=0;j<vHiearchy_[i].ei.size();j++) {
      eRemap_[vHiearchy_[i].ei[j]] = ecntr++;
    }
  }
}
   


/*
 * Code for generalized error functionals
 */

GenSimpFunct::GenSimpFunct(GenSimpObj* sobj)
:owner(sobj)
{

}

int GenSimpFunct::TrySimp(Point& p, double& v, double& err, 
			  GenMeshEdgePiece *m,
			  int& which)
{
  Point pmin;
  double vmin;
  double emin;
  int got_one=0;

  Point A = m->pA;
  Point B = m->pB;
  
  double Av = m->vA;
  double Bv = m->vB;
  
  p = A;
  v = Av;
  
  if (Error(p,v,err,m)) { // it passed the test
    if (!got_one) { // just set this stuff...
      pmin = p;
      vmin = v;
      emin = err;
      got_one = 1;
      which=0;
    } else {
      if (err < emin) {
	pmin = p;
	vmin = v;
	emin = err;
	which=0;
      }
    }
  }
  
  p = B;
  v = Bv;
  if (Error(p,v,err,m)) { // it passed the test
    if (!got_one) { // just set this stuff...
      pmin = p;
      vmin = v;
      emin = err;
      got_one = 1;
      which=1;
    } else {
      if (err < emin) {
	pmin = p;
	vmin = v;
	emin = err;
	which=1;
      }
    }
  }
  
  p = ((A.vector()+B.vector())*0.5).point();
  v = (Av + Bv)*0.5;
  if (Error(p,v,err,m)) { // it passed the test
    if (!got_one) { // just set this stuff...
      pmin = p;
      vmin = v;
      emin = err;
      got_one = 1;
      which=2;
    } else {
      if (err < emin) {
	pmin = p;
	vmin = v;
	emin = err;
	which=2;
      }
    }
  }

  if (!got_one) // none of the points were valid...
    return 0;
  
  p = pmin;
  v = vmin;
  err = emin;

  return 1;
}

GenMeshEdgePiece::GenMeshEdgePiece()
{
  edge = -1; 
  owner = 0;
}

void GenMeshEdgePiece::Init(GenSimpObj *o)
{
  owner = o;
}


int GenMeshEdgePiece::TryZapEdge(int va, int vb)
{
  AugEdge test(va,vb),*res=0;

  if (owner->edges.lookup(&test,res)) {
    if (!owner->error_queue->nuke(res->id+1)) {
      cerr << "Edge is hosed in the PQ...\n";
      return 0;
    }

    owner->edgelst[res->id] = 0; // free spot now...
    owner->edges.remove(res);
    return 1;
  }
  return 0;
}


int GenMeshEdgePiece::UnFlagEdge(int va, int vb)
{
  AugEdge test(va,vb),*res=0;

  if (owner->edges.lookup(&test,res)) {
    res->flags |= AugEdge::pulled_from_queue;
    if (!owner->error_queue->nuke(res->id+1)) {
      cerr << "Edge is hosed in PQ, unflag\n";
      return 0;
    }
    owner->oEdges_.add(res->id); // needs to be recomputed...
    return 1;
  }
  return 0;
}


int GenMeshEdgePiece::TryReplaceEdge(int va, int vb, int vc)
{
  AugEdge test(va,vb),*res=0;

  if (owner->edges.lookup(&test,res)) {
    int slot = res->id;

    owner->edges.remove(res); // remove it...

    AugEdge *nedge = new AugEdge(va,vc);
    if (owner->edges.cond_insert(nedge)) {
      nedge->id = slot;
      owner->edgelst[slot] = nedge;
      owner->oEdges_.add(nedge->id);  // need to recompute this guy...
      return 1;
    } else {
      delete nedge; // it is already in there
      //cerr << "Conditional insert failed - shouldn't happen!\n";
      return 0;
    }
  }
  return 0; // already replaced...
}
