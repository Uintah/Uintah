/*
 * Simplification for 2d manifold surfaces
 * Peter-Pike Sloan
 */

#ifndef _SIMPOBJ2D_H_
#define _SIMPOBJ2D_H_ 1

#include <Modules/Simplify/GenSimpObj.h>
#include <Modules/Simplify/Mesh2d.h>

#include <Modules/Simplify/Quadric.h>
#include <Classlib/BitArray1.h>

struct mergeInfo;

class SimpObj2d : public GenSimpObj {
public:
  Mesh2d              mesh2d;   // surface that is being optimized
  
  Array1< Quadric3d > quadrics_; // only initialized if needed...

  void Init(TriSurface* ts); // inits from tri-surface

  void ComputeQuadrics();    // does this if neccesary...

  void DumpSurface(TriSurface *);

  SimpObj2d();

  Array1< mergeInfo > mergeInfo_;
//  BitArray1           *faceFlags_; // cleared if deleted
protected:
  int origNodes; // starting number of nodes
  int origElems; // starting number of elems
};

/*
    ____ ____ 
    \a0/B\a1/ 
     \/ | \/  
     /\0|1/\  
    /b0\A/b1\ 
   +---- ----+

   0 and 1 are the indeces of the faces in the mVertex struct,
   there should only be 2...
   
*/	    
	    
struct mergeInfo {
  int fA[2],fB[2]; // 2nd value only filled in if it isn't a boundary...
};

class Element2dAB;

class MeshEdgePiece2d : public GenMeshEdgePiece {
public:
  MeshEdgePiece2d();

  virtual void Init(SimpObj2d*); // simple initializations
  
  virtual int SetEdge(int eid, int top_check=1); // sets active edge
  virtual int TryFlush(int do_checks=1);         // tries to flush
  
  // below checks if mesh is valid collapsing to given point
  virtual int ValidPoint(Point& p);   

  // you need the array of normals to check...

  Array1< Element2dAB > elems; // elements to be checked by error functional...

  int toDie[2]; // faces to be removed from the mesh...

  Quadric3d curQ; // current quadric - if appropriate...

};


class Element2dAB {
public:
  int eid; // element on the surface

  void Init(Mesh2d *msh,int node, int elem);

//  double Cx,Cy,Cz; // precomputed coefs
  Vector Cv;

  Vector BA;
  Vector No;                // old normal for this face - normalized

  Point pA,pB; // old points...

  double IsOk(Point& p);
};

class QuadricError2d : public GenSimpFunct {
public:
  QuadricError2d(SimpObj2d*);

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

#endif
