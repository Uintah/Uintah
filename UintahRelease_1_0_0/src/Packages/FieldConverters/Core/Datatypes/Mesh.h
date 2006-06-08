/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Mesh.h: Unstructured meshes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_FieldConverters_Mesh_h
#define SCI_FieldConverters_Mesh_h 1

#include <FieldConverters/share/share.h>

#include <Core/Datatypes/Datatype.h>

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Handle.h>

#include <FieldConverters/Core/Datatypes/Surface.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <stdlib.h> // For size_t

namespace SCIRun {
class GeomGroup;
}

namespace FieldConverters {

using namespace SCIRun;

struct Node;

typedef Handle<Node> NodeHandle;

#define STORE_ELEMENT_BASIS

class RPoint;
class BigRational;
class OMesh;

struct Element {
  int faces[4];
  int n[4];
  int generation; // For insert_delaunay
  int cond; // index to the conductivities array for the cond
  // tensor of this element

#ifdef STORE_ELEMENT_BASIS
  Vector g[4];
  double a[4];
  double vol;
#endif

  void compute_basis();

  OMesh* mesh;
  Element(OMesh*, int, int, int, int);
  Element(const Element&, OMesh* mesh);
  void* operator new(size_t);
  void operator delete(void*, size_t);
  inline int face(int);

  double volume();
  int orient();
  void get_sphere2(Point& cen, double& rad2, double& err);
  void get_sphere2(RPoint& cen, BigRational& rad2);
  Point centroid();
};

struct DirichletBC {
  SurfaceHandle fromsurf;
  double value;
  DirichletBC(const SurfaceHandle&, double);
};

struct PotentialDifferenceBC {
  int diffNode;
  double diffVal;
  PotentialDifferenceBC(int, double);
};

struct Node : public Persistent {
  //struct Node {
  Point p;
  int ref_cnt;
  Node(const Point&);
  Array1<int> elems;

  DirichletBC* bc;
  int fluxBC;
  PotentialDifferenceBC* pdBC;

  Node(const Node&);
  virtual ~Node();
  virtual void io(Piostream&);
  virtual Node* clone();
  static PersistentTypeID type_id;
  void* operator new(size_t);
  void operator delete(void*, size_t);
};

struct NodeVersion1 {
  Point p;
};

struct ElementVersion1 {
  int n0;
  int n1;
  int n2;
  int n3;
};

struct Face {
  Face* next;
  int hash;
  int face_idx;
  int n[3];
  Face(int, int, int);
  inline int operator==(const Face& f) const {
    return n[0]==f.n[0] && n[1]==f.n[1] && n[2]==f.n[2];
  }

  void* operator new(size_t);
  void operator delete(void*, size_t);
};

struct Edge{
  int n[2];
  Edge();
  Edge(int, int);
  int hash(int hash_size) const;
  int operator==(const Edge&) const;
  int operator<(const Edge&) const;
};

struct MeshGrid {
  Point min, max;
  int nx, ny, nz;
  Array3<Array1<int> > elems;
  int locate(OMesh* mesh, const Point& p, double epsilon);
};

struct Octree{
  Point mid;
  Array1<int> elems;
  Octree* child[8];
  Octree();
  ~Octree();

  int locate(OMesh* mesh, const Point& p, double epsilon);
};

class OMesh;
typedef LockingHandle<OMesh> OMeshHandle;

class SCICORESHARE OMesh : public Datatype {
  MeshGrid grid;
  Octree* octree;
public:
  int bld_grid;
  Array1<int> ids;
  Array1<NodeHandle> nodes;
  Array1<Element*> elems;
  Array1<Array1<double> > cond_tensors;
  int have_all_neighbors;
  void compute_face_neighbors();
  OMesh();
  OMesh(const OMesh&);
  OMesh(int nnodes, int nelems);
  virtual OMesh* clone();
  virtual ~OMesh();

  int current_generation;

  void detach_nodes();
  void get_elem_nbrhd(int eidx, Array1<int>& nbrs, int dupsOk=1);
  void get_node_nbrhd(int nidx, Array1<int>& nbrs, int dupsOk=1);
  void compute_neighbors();
  int locate(const Point&, int&, double epsilon1=1.e-6, double epsilon2=1.e-6);
  int locate(const RPoint&, int&);
  int locate2(const Point&, int&, double epsilon1=1.e-6);
  int inside(const Point& p, Element* elem);
  void get_interp(Element* elem, const Point& p, double& s1,
		  double& s2, double& s3, double& s4);
  double get_grad(Element* elem, const Point& p, Vector& g1,
		  Vector& g2, Vector& g3, Vector& g4);
  void get_bounds(Point& min, Point& max);
  int unify(Element*, const Array1<int>&, const Array1<int>&,
	    const Array1<int>&);

  int insert_delaunay( int node );
  int insert_delaunay( const Point& );
  void remove_delaunay(int node, int fill);
  void pack_nodes();
  void pack_elems();
  void pack_all();
  int face_idx(int, int);
  void add_node_neighbors(int node, Array1<int>& idx, int apBC=1);
  void new_element(Element* ne, HashTable<Face, int> *new_faces);
  void remove_all_elements();
  void get_boundary_nodes(Array1<int> &pts);

  void get_boundary_lines(Array1<Point>& lines);

  void draw_element(int, GeomGroup*);
  void draw_element(Element* e, GeomGroup*);

  bool vertex_in_tetra(const Point& v, const Point& p0, const Point& p1,
		       const Point& p2, const Point& p3);
  bool tetra_edge_in_box(const Point& min, const Point& max,
			 const Point& orig, const Vector& dir);
  bool overlaps(Element* e, const Point& min, const Point& max);

  void make_octree(int level, Octree*& octree, const Point& min,
		   const Point& max, const Array1<int>& elems);

  void make_grid(int nx, int ny, int nz, const Point &min, const Point &max,
		 double eps);

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

inline int Element::face(int i)
{
  if(faces[i] == -2){
    int i1=n[(i+1)%4];
    int i2=n[(i+2)%4];
    int i3=n[(i+3)%4];
    Node* n1=mesh->nodes[i1].get_rep();
    Node* n2=mesh->nodes[i2].get_rep();
    Node* n3=mesh->nodes[i3].get_rep();
    // Compute it...
    faces[i]=mesh->unify(this, n1->elems, n2->elems, n3->elems);
  }
  return faces[i];
}
}

namespace SCIRun {

void Pio(Piostream&, FieldConverters::Element*&);
void Pio(Piostream& stream, FieldConverters::NodeVersion1& node);
void Pio(Piostream& stream, FieldConverters::ElementVersion1& node);

} // End namespace SCIRun

#endif
