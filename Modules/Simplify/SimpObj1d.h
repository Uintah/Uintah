/*
 * Simplification for 1D streamlines
 * Peter-Pike Sloan
 */

#ifndef _SIMPOBJ1D_H_
#define _SIMPOBJ1D_H_ 1

#include <Modules/Simplify/GenSimpObj.h>
#include <Modules/Simplify/Quadric.h>

struct Seg1d {
  int A,B; // every segment just connects 2 points

  int Ac,Bc; // other segs these are attached to...

};

class SimpObj1d : public GenSimpObj {
public:
  SimpObj1d(); // default constructor...
  Array1< Point > pts_;  // everything indexes into this...
  Array1< Seg1d > segs_; // line segments...
  
  Array1< Quadric3d > quadrics_; // only initialized if needed...

  void Init(Array1<Point> *pts,
	    Array1<float> *sv=0); // points, maybe scalar values...

  // if planeEnds != 1, don't cap curve with planes at the end...
  void ComputeQuadrics(int planeEnds=1);

  int startSize_; // initial size - in pts...

protected:

  
};

// this data augments the ecol/split records...

class mVertex1d {
public:

  int Ai,Bi; // "edges" have that have to be present for this...
  
  int nE;    // new edge created by this split
};

// piece of the mesh - kind of simple for polylines...

class MeshEdgePiece1d:public GenMeshEdgePiece {
public:
  MeshEdgePiece1d();

  virtual void Init(SimpObj1d*); // simple initializations
  
  virtual int SetEdge(int eid, int top_check=1); // sets active edge
  virtual int TryFlush(int do_checks=1);         // tries to flush
  
  // below checks if mesh is valid collapsing to given point
  virtual int ValidPoint(Point& p);   

  Quadric3d curQ;

};


class QuadricError1d : public GenSimpFunct {
public:
  QuadricError1d(SimpObj1d*);

  // below is virtual function that returns if collapse
  // is possible and returns the error that would be incurred
  // first double is the scalar value to evaluate with

  virtual int Error(Point&,double&,double&,GenMeshEdgePiece*); 

  // below is more aggresive, it tries to optimize the node
  // posistion better - might be expensive though, depending
  // on implimantation - Quadrics in this case - also

  virtual int MinError(Point&, double&, double&, GenMeshEdgePiece*);

  // below is called when a event is flushed into the system

  virtual void PostCollapseStuff(GenMeshEdgePiece*);  

  // the error functional needs to know what the quadrics are
  // for the given mesh...

  Array1< Quadric3d > *quadrics_;
};

// below computes the minimum distance between the tested point
// and the original poly line

class ProjectError1d : public GenSimpFunct {
public:
  ProjectError1d(SimpObj1d*);

  // below is virtual function that returns if collapse
  // is possible and returns the error that would be incurred
  // first double is the scalar value to evaluate with

  virtual int Error(Point&,double&,double&,GenMeshEdgePiece*); 

  // below is more aggresive, it tries to optimize the node
  // posistion better - might be expensive though, depending
  // on implimantation - Quadrics in this case - also

  virtual int MinError(Point&, double&, double&, GenMeshEdgePiece*);

  // below is called when a event is flushed into the system

  virtual void PostCollapseStuff(GenMeshEdgePiece*);  
};


#endif
