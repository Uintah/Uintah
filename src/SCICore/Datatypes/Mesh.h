
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

#include <SCICore/share/share.h>

#include <SCICore/Datatypes/Datatype.h>

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Handle.h>

#include <SCICore/Datatypes/Surface.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

#include <stdlib.h> // For size_t

namespace SCICore {

namespace GeomSpace {
  class GeomGroup;
}

namespace Datatypes {

using Containers::Array3;
using Containers::LockingHandle;
using Containers::Handle;
using Geometry::Vector;
using Geometry::Point;
using GeomSpace::GeomGroup;
using Containers::HashTable;

struct Node;
typedef Handle<Node> NodeHandle;

#define STORE_ELEMENT_BASIS

class RPoint;
class BigRational;
class Mesh;

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

    Mesh* mesh;
    Element(Mesh*, int, int, int, int);
    Element(const Element&, Mesh* mesh);
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
    int locate(Mesh* mesh, const Point& p, double epsilon);
};

struct Octree{
  Point mid;
  Array1<int> elems;
  Octree* child[8];
  Octree();
  ~Octree();

  int locate(Mesh* mesh, const Point& p, double epsilon);
};

class Mesh;
typedef LockingHandle<Mesh> MeshHandle;

class SCICORESHARE Mesh : public Datatype {
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
    Mesh();
    Mesh(const Mesh&);
    Mesh(int nnodes, int nelems);
    virtual Mesh* clone();
    virtual ~Mesh();

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

void Pio(Piostream&, Element*&);
void Pio(Piostream& stream, NodeVersion1& node);
void Pio(Piostream& stream, ElementVersion1& node);
} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.7.2.4  2000/11/01 23:03:01  mcole
// Fix for previous merge from trunk
//
// Revision 1.7.2.2  2000/10/26 17:30:46  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.8  2000/07/12 15:45:09  dmw
// Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
//
// Revision 1.7  2000/03/11 00:41:29  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.6  2000/03/04 00:18:29  dmw
// added new Mesh BC and fixed sparserowmatrix bug
//
// Revision 1.5  2000/02/02 22:07:11  dmw
// Handle - added detach and Pio
// TrivialAllocator - fixed mis-allignment problem in alloc()
// Mesh - changed Nodes from LockingHandle to Handle so we won't run out
// 	of locks for semaphores when building big meshes
// Surface - just had to change the forward declaration of node
//
// Revision 1.4  1999/09/05 05:32:27  dmw
// updated and added Modules from old tree to new
//
// Revision 1.3  1999/08/25 03:48:35  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:48  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:23  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:48  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:41  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:28  dav
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:38  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
