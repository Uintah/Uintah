/*
 * Simplification for 3d volumes
 * Peter-Pike Sloan
 */

#ifndef _SIMPOBJ3D_H_
#define _SIMPOBJ3D_H_ 1

#include <Modules/Simplify/GenSimpObj.h>

#include <Datatypes/Mesh.h>

#include <Modules/Simplify/Quadric.h>
#include <Classlib/BitArray1.h>

struct mergeInfo;
struct mergeInfoV;

class SimpObj3d : public GenSimpObj {
public:
  Mesh                mesh3d;   // surface that is being optimized
  
  Array1< Quadric3d > quadrics_;  // only initialized if needed...
 
  Array1< Quadric4d > quadrics4_; // also scalar value...

  void Init(Mesh* ts); // inits from tri-surface

  void ComputeQuadrics();    // does this if neccesary...

  SimpObj3d();

  Array1< mergeInfoV > mergeInfo_;

  // the mesh data structure has the neccesary
  // stuff (node face in particular...)

protected:
  int origNodes; // starting number of nodes
  int origElems; // starting number of elems
};

/*

  This structure encodes the elements opposite of the 2 vertices
  that define this edge.  This array is the same size as the
  mVertex array, and is "lockstep" with it.


       /A 
      /	|\
     D 	| \    
      \	|  \   
       \B---C  
	       
   For this element (AB is the edge to be collapsed), the faces
   (really elements on the other side of a face) that are stored
   in this record would be other side of DBC in eA, and other side
   of DAC in eB.
   
   This is stored for every element that is collapsed with this
   edge.
   
*/	    
	    
struct mergeInfoV {
  Array1<int> eA;
  Array1<int> eB;
};

class Element3dAB;
class Element3dC;

class MeshEdgePiece3d : public GenMeshEdgePiece {
public:
  MeshEdgePiece3d();

  virtual void Init(SimpObj3d*); // simple initializations
  
  virtual int SetEdge(int eid, int top_check=1); // sets active edge
  virtual int TryFlush(int do_checks=1);         // tries to flush
  
  // below checks if mesh is valid collapsing to given point
  virtual int ValidPoint(Point& p);   

protected:
  
  int TryAdd(Node *nd,               // node to add
	     int me, int other,      // indeces for edge
	     Array1< Element3dAB >& ring, // A or B ring
	     int do_Cring,           // wether to build C
	     int top_check);         // 1 for topology check

  // you need the array of normals to check...

  Array1< Element3dAB > Aring; // has A but not B
  Array1< Element3dAB > Bring; // has B but not A

  Array1< Element3dC >  Cring; // ring that has both

  Quadric3d curQ; // current quadric - if appropriate...

  // per-proc state for generation info

  Array1< unsigned int > elemGen;
  Array1< unsigned int > nodeGen;

  int curGen;

  FastHashTable<Face>    bdryFaces; // to check

  // this is per-proc data that would be allocated to
  // often otherwise...

  Array1<int> scratch1; // scratch array 
  Array1<int> scratch2; // ditto - use as reference when needed...

  Mesh *mesh;
};


class Element3dAB {
public:
  int eid; // element on the surface

  int nd;  // index [0,3] that needs to be fixed - is either A or B

  void Init(Mesh *msh,int node, int elem);

  double C,X,Y,Z; // precomputed portions of volume equation

  double S;       // scalar value - if neccesary

  double IsOk(Point& p) { return (C+X*p.x() + Y*p.y() + Z*p.z()); };
};

class Element3dC {
public:
  int eid;

  int Ai,Bi;  // indeces in [0,3] for A and B

  int Oppi[2]; // other indeces for tet in [0,3]

  void Init(int nodeA, int nodeB, int elem, int opp0, int opp1);
};

class QuadricError3d : public GenSimpFunct {
public:
  QuadricError3d(SimpObj3d*);

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
