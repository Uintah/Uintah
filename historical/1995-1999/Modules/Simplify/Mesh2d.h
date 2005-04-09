
/*
 * Simple surface class with more topology than
 * the general TriSurface stuff.
 *
 * Peter-Pike Sloan
 */

#ifndef _MESH_2D_H_
#define _MESH_2D_H_ 1

#include <Geometry/Point.h>
#include <Datatypes/TriSurface.h>

const int ELEM_IS_VALID=1;
const int NODE_IS_VALID=1;

// you can keep extra bits around so that you know how
// to remap the vertex indices with the neighboring face
// there are 6 possible mappings - 

/*



 D---------A
  \       / \
   \ TB  /   \
    \   /  TA \	
     \ /       \ 
      C---------B
  	   
      
      For triangle TA: (A-0,B-1,C-2)
      For triangle TB: (C-0,D-1,A-2)

      So for vertex B (index 1)

      2 -> 0
      0 -> 2

      index 1 is opposite vertex across this face

      To follow a ring (A), you start on a face (TA) and pick
      a vertex (not yourself) to follow the ring with (B - 1)

      You then follow the face neighbors opposite this vertex (enter
      into TB).  You need to know where C (other vertex not A) is
      mapped to, and where the "opposite" (D) index is (for the next
      step.)  You repeat this until you reneter the starting face or
      hit a boundary (in which case you need to follow the ring from C).

*/

const int MAP1D_P1_0_P2_1 = 1; // vert+1 -> 0, vert +2 -> 1, 2 is opp
const int MAP1D_P1_0_P2_2 = 2; // vert+1 -> 0, vert +2 -> 2, 1 is opp

const int MAP1D_P1_1_P2_0 = 3; // vert+1 -> 1, vert +2 -> 0, 2 is opp
const int MAP1D_P1_1_P2_2 = 4; // vert+1 -> 1, vert +2 -> 2, 0 is opp

const int MAP1D_P1_2_P2_0 = 5; // vert+1 -> 2, vert +2 -> 0, 1 is opp
const int MAP1D_P1_2_P2_1 = 6; // vert+1 -> 2, vert +2 -> 1, 0 is opp

const int MAP_MASK = (1+2+4); // 3 bits

// bit 1 is for valid/invalid

struct MeshTop2d {
  int v[3];  // vertices
  int f[3];  // face neighbors

  int flags;     // 0 means invalid face...

  inline int getFlags(int i) { // get flags for index i
    return ((flags >> (i*3 + 1))&MAP_MASK);
  };

  inline void setFlags(int i, int val) {
    int shft = (i*3 + 1);
    flags = (flags&(~(MAP_MASK<<shft))) | (val<<shft);
  };

};

struct Mesh2d {
  Array1<Vector> vnormals; // normals - only there if computed...
  Array1<Vector> fnormals; // face normals...

  Array1<Point> points; // data for this object...
  Array1< MeshTop2d > topology;

  Array1< int >     nodeFace; // a face that contains given node...

  Array1< Array1<int> > nodeFaces; // face ring for each node...

  Array1<int>   pt_flags;      // flags for a node - validity...
  
  void GetFaceRing(int nid, Array1<int>&); //gets face ring...

  // below uses the bitflags - might be faster...
  void GetFaceRing2(int nid, Array1<int>&);

  void CreateFaceNeighbors(); // creates face neighbors from vids

  void Init(TriSurface* ts, int DoConnect=1); // inits from tri-surface

  void Dump(TriSurface* ts);

  void Validate(); // tries to validate the mesh...

  void ComputeNormals(int); // computes vertex normals from face normals...
};

#endif
