/*
 * Simplification for non-manifold 2d surfaces
 * Peter-Pike Sloan
 */

#ifndef _SIMPOBJNM2D_H_
#define _SIMPOBJNM2D_H_

#include <Modules/Simplify/SimpObj2d.h>

// there are really 2 flavors of NM points
// ones that are connected to 2 NM edges - along a seam
// and ones that are connected to 1 or more than 2, junctions
// between curves.

struct AugNMPoint {
  int pid; // real point that this corresponds too...

  Array1< int > conPts; // connected points -> NM edges
  
  // a face on every "connected" surface that has this point
  Array1< int > conFcs; 
};

class SimpObjNM2d : public SimpObj2d {
public:
  void Init(TriSurface *ts); // inits from t-surf

  void DumpSurface(TriSurface *);

  SimpObjNM2d();

  Array1< int > isPtNM_; // -1 or reference into nmPoint_

  Array1< AugNMPoint > nmPoint_;
protected:

  int AddEdge(int v0, int v1, int f); // returns 1 if NM

};

const int BOTH_MANIFOLD=1;
const int A_NONMANIFOLD=2;
const int B_NONMANIFOLD=4;

class MeshEdgePieceNM2d : public GenMeshEdgePiece {
public:
  MeshEdgePieceNM2d();

  virtual void Init(SimpObj2d*); // simple initializations
  
  virtual int SetEdge(int eid, int top_check=1); // sets active edge
  virtual int TryFlush(int do_checks=1);         // tries to flush
  
  // below checks if mesh is valid collapsing to given point
  virtual int ValidPoint(Point& p);   
  
  Array1< MeshEdgePiece2d > pieces; // sub parts of the mesh

  Quadric3d cumQ; // cumlative quadric...

  // if both points are manifold, this is a plain old
  // edge collapse

  // if 1 point is nm, the face(s) are removed from 1 sheet
  // and the new node is a NM node, all of the original NM
  // nodes have to be changed to reflect this

  // if both nodes are NM:
  // there must be a NM edge between them
  // they must have only 1 other NM edge 
  // they can not share that vertex (or they close a loop)

  int flags;
};

#endif
