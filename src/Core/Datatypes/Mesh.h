
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

#ifndef SCI_project_Mesh_h
#define SCI_project_Mesh_h 1

#include <Core/share/share.h>

#include <Core/Datatypes/Datatype.h>

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Handle.h>

#include <Core/Datatypes/Surface.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <stdlib.h> // For size_t
#include <vector>

namespace SCIRun {

class GeomGroup;
//struct Node;
//typedef Handle<Node> NodeHandle;

//#define STORE_ELEMENT_BASIS

class RPoint;
class BigRational;
class Mesh;


struct Element {
  int faces[4];
  int n[4];
  //int generation; // For insert_delaunay
  int cond; // index to the conductivities array for the cond
  // tensor of this element

#ifdef STORE_ELEMENT_BASIS
  Vector g[4];
  double a[4];
  double vol;
#endif

  void compute_basis();

  Mesh* mesh;

  Element(Mesh*, int, int, int, int);
  Element(const Element &e, Mesh *mesh);
  void *operator new(size_t);
  void operator delete(void*, size_t);
  inline int face(int);

  double volume();
  int orient();
  void get_sphere2(Point &cen, double &rad2, double &err);
  void get_sphere2(RPoint &cen, BigRational &rad2);
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


struct Node {
  Point p;

  Array1<int> elems;
  
  DirichletBC* bc;
  int fluxBC;
  PotentialDifferenceBC* pdBC;

  Node();
  Node(const Point &p);
  Node(const Node &n);
  ~Node();

  void *operator new(size_t);
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
  Face *next;
  int hash;
  int face_idx;
  int n[3];

  Face(int, int, int);
  inline int operator==(const Face& f) const {
    return n[0]==f.n[0] && n[1]==f.n[1] && n[2]==f.n[2];
  }

  void *operator new(size_t);
  void operator delete(void*, size_t);
};


struct Edge {
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
  int locate(Mesh* mesh, const Point& p, double epsilon);
};

#if 0
struct MeshOctree {
  Point mid;
  Array1<int> elems;
  MeshOctree* child[8];
  MeshOctree();
  ~MeshOctree();

  int locate(Mesh* mesh, const Point& p, double epsilon);
};
#endif

class Mesh;
typedef LockingHandle<Mesh> MeshHandle;

class SCICORESHARE Mesh : public Datatype {
  friend class MeshGrid;
//  friend class MeshOctree;
  friend class Element;

protected:
  MeshGrid grid;
//  MeshOctree *octree;
  int bld_grid;
  //Array1<NodeHandle> nodes;
  //Array1<Element*> elems;
  //Array1<Array1<double> > cond_tensors;

//  std::vector<int> *delaunay_generation;
//  int current_generation;

public:
  Array1<Node> nodes;
  Array1<Element *> elems;
  Array1<Array1<double> > cond_tensors;
  int have_all_neighbors;

  const Node &node(int i) const { return nodes[i]; }
  const Point &point(int i) const { return nodes[i].p; }
  Element *element(int i) const { return elems[i]; }
  int nodesize() const { return nodes.size(); }
  int elemsize() const { return elems.size(); }

  // Constructors and destructors.
  Mesh();
  Mesh(const Mesh &copy);
  Mesh(int nnodes, int nelems);
  virtual Mesh *clone();
  virtual ~Mesh();

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  bool locate(int *strt, const Point &p, double eps1=1.e-6, double eps2=1.e-6);
  bool locate2(int *strt, const Point &p, double epsilon1=1.e-6);


#if 0
  bool mike_locate(int *loc, const Point &p, double eps=1.0e-6)
  {
    return locate(p, *loc, eps);
  }


  const static int ELESIZE = 1;


  int el[ELESIZE];
  if (mike_locate(el, some_point))
  {
    for (int i=0; i< mike_edgecount(el); i++)
    {
      int edge0[ELESIZE], edge1[ELESIZE];
      mike_edgeindex(edge0, edge1, el, i);
      printf("edge %n goes from %d to %d\n", i, *edge0, *edge1)
    }
  }

  int mike_nodecount(int *loc) { return 4; }
  void mike_nodeindex(int *ni, int *loc, int n) { *ni = *loc + n; }
  
  int mike_edgecount(int *loc) { return 6; }
  void mike_edgeindex(int *e0, int *e2, int *loc, int n)
  {
    switch (n)
    {
    case 0:
      *e0 = *loc+0; *e1 = *loc+1; break;
    case 1:
      *e0 = *loc+0; *e1 = *loc+2; break;
    case 2:
      *e0 = *loc+0; *e1 = *loc+3; break;
    case 3:
      *e0 = *loc+1; *e1 = *loc+2; break;
    case 4:
      *e0 = *loc+1; *e1 = *loc+3; break;
    case 5:
      *e0 = *loc+2; *e1 = *loc+3; break;
    }
  }

  int mike_facecount(int *loc) { return 4; }
  void mike_faceindex(int *v0, int *v1, int *v2, int *loc, int n)
  {
    switch (n)
    {
    case 0:
      *v0 = *loc+0; *v1 = *loc+1; *v2 = *loc+2; break;
    case 0:
      *v0 = *loc+0; *v1 = *loc+3; *v2 = *loc+1; break;
    case 0:
      *v0 = *loc+0; *v1 = *loc+2; *v2 = *loc+3; break;
    case 0:
      *v0 = *loc+1; *v1 = *loc+2; *v2 = *loc+3; break;
    }
  }
#endif


  void get_interp(Element* elem, const Point& p, double& s1,
		  double& s2, double& s3, double& s4);

  void get_bounds(Point& min, Point& max);

  void get_boundary_lines(Array1<Point>& lines);
  void get_elem_nbrhd(int eidx, Array1<int>& nbrs, int dupsOk=1);
  void get_node_nbrhd(int nidx, Array1<int>& nbrs, int dupsOk=1);

  double get_grad(Element* elem, const Point& p, Vector& g1,
  		  Vector& g2, Vector& g3, Vector& g4);

  void detach_nodes();
  void compute_neighbors();
  void compute_face_neighbors();

//  bool insert_delaunay( int node );
//  bool insert_delaunay( const Point& );
//  void remove_delaunay(int node, int fill);
  void pack_nodes();
  void pack_elems();
  void pack_all();

  void remove_all_elements();

  void add_node_neighbors(int node, Array1<int>& idx, int apBC=1);
  int inside(const Point& p, Element* elem);

private:
  int unify(Element*, const Array1<int>&, const Array1<int>&,
	    const Array1<int>&);

  int face_idx(int, int);
  void new_element(Element* ne, HashTable<Face, int> *new_faces);
  void get_boundary_nodes(Array1<int> &pts);

  void draw_element(int, GeomGroup*);
  void draw_element(Element* e, GeomGroup*);

  bool vertex_in_tetra(const Point& v, const Point& p0, const Point& p1,
		       const Point& p2, const Point& p3);
  bool tetra_edge_in_box(const Point& min, const Point& max,
			 const Point& orig, const Vector& dir);
  bool overlaps(Element* e, const Point& min, const Point& max);

//  void make_octree(int level, MeshOctree*& octree, const Point& min,
//		   const Point& max, const Array1<int>& elems);

  void make_grid(int nx, int ny, int nz, const Point &min, const Point &max,
		 double eps);
};


inline int Element::face(int i)
{
  if (faces[i] == -2) {
    int i1=n[(i+1)%4];
    int i2=n[(i+2)%4];
    int i3=n[(i+3)%4];
    const Node &n1=mesh->node(i1);
    const Node &n2=mesh->node(i2);
    const Node &n3=mesh->node(i3);
    // Compute it...
    faces[i]=mesh->unify(this, n1.elems, n2.elems, n3.elems);
  }
  return faces[i];
}


void Pio(Piostream&, Element*&);
void Pio(Piostream& stream, NodeVersion1& node);
void Pio(Piostream& stream, ElementVersion1& node);

} // End namespace SCIRun

#endif
