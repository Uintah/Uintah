/*
 * Generic Code for simplification
 *
 * Peter-Pike Sloan
 *
 */

#ifndef _GENSIMPBJ_H_
#define _GENSIMPBJ_H_ 1

// this is the generic class...

/*
 * Keep in mind that we want to create merge trees - so we
 * need to always increase the number of vertices as we do
 * collapses!
 */

#include <Multitask/Task.h>
#include <Multitask/ITC.h>

#include <Classlib/PQueue.h>
#include <Classlib/Queue.h>
#include <Classlib/Array1.h>


#include <Classlib/FastHashTable.h>
#include <Modules/Simplify/AugEdge.h>

class GenSimpFunct;
class GenMeshEdgePiece;
class mVertex;

class GenMeshEdgePiece;

class GenSimpObj {
public:

  friend class GenMeshEdgePiece;

  GenSimpObj(); // default constructor...

  // if useOld is not zero, looks in oEdges_ instead...
  void FillQ(int useOld=0); // fills up the priority queue

  void SFillQ(int useOld=0); // serial code - for debugging...

  void ReQ();   // tries to enq all potential edges
  
  int  PopQ(int npop=1); // tries to pop requested number of times

  int  PopQE(int ne);    // collapses Q, tries to remove a certain # of elems

  int  PopQ(double percent); // tries to remove percent of elements

  // how empty can the Q get before you refill it?

  void QPercent(double pc) { qPercent_ = pc; };

  // You shouldn't have to call below, will be called when the Q
  // reaches empty enough level
  // edges are rq'd based on flag settings and "bad" weights

  int ReFillQ(); // returns number of elements that were rq'd

  // this is to get it ready for parallel stuff...

  void SetNumProc(int np);

  // this is to control granularity...

  void SetDumpSize(int d) { dump_size = d; };

  // this controls granularity of thread stuff..

  void SetChunkSize(int v) { chunk_proc = v; };

  // this controls other stuff...

  void SetKeepTime(int v) { keep_time = v;};

  // below is used by multi-tasking stuff...

  void DoEdgeChunk(int proc);
  
  // this function validates the structure

  void CheckMesh();

  inline void IncNodeCount(int nd); // how many times collapsed
  inline int  NodeInQ(int nd);      // if the node is in the Q
  inline void enQNode(int nd);      // flags node as in the Q
  inline void NodeQ(int nd, int flag); // sets/unsets node

  // edges are the same, independant of dimension...

  FastHashTable<AugEdge> edges;
  Array1<AugEdge*>       edgelst;

  // this function builds edgelst from hash table...
  // only call this if you actualy create the hash table...

  void BuildEdgeList();

  // these are functions to be used by threads...

  void DoPQueueStuff(); // single thread loads pqueue

  void parallel_q_fill(int proc); // fills pqueue
  void parallel_q_refill(int proc); // fills pqueue

  // this recomputes the indeces for vertices
  // and any other geometry
  void recomputeIndex(int norig, int nelem);

  int keepRecords_; // 1 if you should be keeping hiearchy...

  Array1< mVertex > vHiearchy_;

  Array1< int >     vRemap_; // used by above...
  Array1< int >     eRemap_; // remaps elements - 

  // the remap arrays are used for merge tree's...

  // this array is used to figure out which edges
  // need to be recomputed...

  Array1< int >     oEdges_;

  // below needs to be set!
  GenSimpFunct *ErrFunc; // current error functional

  PQueue *error_queue; // actual error queue

protected:

  Array1< GenMeshEdgePiece* >  mpieces; // thread local state

  // node flags can contain a bunch of junk...

  Array1<unsigned int>     ndFlags;

  // 0 percent below means you always fill up after you pop
  double qPercent_;   // how far you can drain the queue before filling up
  
  unsigned int    qFlags_;     // flags for the queue, see above

  int             qBadWeight_; // how many refill passes before you try again
  int             keep_time;   // wether or not to have threads keep time...
  int             dump_size;   // how many edges to do at a time...
  int             chunk_proc;  // this *numProc = desired chunk interval size
  int             numProc;   // number to use

  int             qfill_init; // weather filler thread has been created

  Mutex chunkLock,edgeQlock,freeQlock;

  Array1<int>        chunkPos; // posistion of chunks
  int                curChunk; // working interval is [curChunk,curChunk+1]

  Queue< Array1<int>* >   edgeQ,freeQ; // edges to be enqueued/freed

  Array1< Array1<int>* >  workQ; // ones that are currently getting enqueued
 
  Semaphore   FlushQ,WakeMain;   // semaphores for enqueuing...
}; 


// this is a localized piece of a mesh - it doesn't
// flush back to the original until TryFlush is called

// this is allocated once per processor...

class GenMeshEdgePiece {
public:
  GenMeshEdgePiece();

  virtual void Init(GenSimpObj*); // simple initializations

  virtual int SetEdge(int eid, int top_check=1)=0; // sets active edge

  // flush has to update the mesh, and keep a record
  // of the collapses so you can build a merge tree...
  virtual int TryFlush(int do_checks=1)=0;         // tries to flush

  // below checks if mesh is valid collapsing to given point
  virtual int ValidPoint(Point& p)=0; 

  int edge; // active edge
  int A,B;  // nodes for above edge...

  Point pA,pB; // actual posistions for above edge...
  double vA,vB; // scalar values for above edge...

  GenSimpObj *owner; // simplify obj that owns this piece...

  // these are functions for replacing edges and stuff...

  int TryZapEdge(int va, int vb);
  int UnFlagEdge(int va, int vb);
  int TryReplaceEdge(int va, int vb, int vc); // ab replaced with ac...
};

// this is the representation for a generic merge
// tree - you simply have split and collapse records...

class mVertex {
public:
  Point pt; // this is the point in 3 space
  // int parent; // -1 if in M0, otherwise vsplit that created this
  
  // mVertex *s; // other vertex is s+1 - 0 if highest level

  int s,u;   // vertices this can split into
  int nv;    // new value...

  Array1< int > ei; // element indeces that have been removed

  // everything else depends on what type of
  // mesh you are dealing with...

  // keep that in a seperate array...
};

class GenSimpFunct {
public:
  GenSimpFunct(GenSimpObj*);
  
  // below is virtual function that returns if collapse
  // is possible and returns the error that would be incurred
  // first double is the scalar value to evaluate with

  virtual int Error(Point&,double&,double&,GenMeshEdgePiece*)=0; 

  // below is more aggresive, it tries to optimize the node
  // posistion better - might be expensive though, depending
  // on implimantation

  virtual int MinError(Point&, double&, double&, GenMeshEdgePiece*)=0;

  // below is called when a event is flushed into the system

  virtual void PostCollapseStuff(GenMeshEdgePiece*)=0;

  // below is the "stock" method, 2 pts and mid point are tested

  int TrySimp(Point&, double&, double&, GenMeshEdgePiece*,int&);

  GenSimpObj *owner;
};


#endif
