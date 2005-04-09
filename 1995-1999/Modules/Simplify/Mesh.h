
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

#include <Datatypes/Datatype.h>

#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Datatypes/GeometryPort.h>

struct OldNode;
// don't use this thing anymore...
typedef LockingHandle<OldNode> NodeHandle;

#include <Datatypes/Surface.h>
#include <Geometry/Point.h>

#include <stdlib.h> // For size_t

// I probably don't need this anyways - so I might as well skip it...
//#define STORE_ELEMENT_BASIS

class GeomGroup;
class Mesh;
class RPoint;
class BigRational;

struct Element {
    int faces[4];
    int n[4];
    int generation; // For insert_delaunay - and simplification
    int cond; // index to the conductivities array for the cond
              // tensor of this element

#ifdef STORE_ELEMENT_BASIS
    Vector g[4];
    double a[4];
    double vol;
#endif

#ifdef STORE_THE_MESH
    void compute_basis();

    Mesh* mesh;
#else
    void compute_basis(Mesh* mesh);
#endif

    Element(Mesh*, int, int, int, int);
    Element(const Element&, Mesh* mesh);
    void* operator new(size_t);
    void operator delete(void*, size_t);
    inline int face(int);

#ifdef STORE_THE_MESH
    double volume();
    int orient();
    void get_sphere2(Point& cen, double& rad2, double& err);
    void get_sphere2(RPoint& cen, BigRational& rad2);
    Point centroid();
#else
    double volume(Mesh* mesh);
    int orient(Mesh* mesh);
    void get_sphere2(Point& cen, double& rad2, double& err, Mesh* mesh);
    void get_sphere2(RPoint& cen, BigRational& rad2, Mesh* mesh);
    Point centroid(Mesh* mesh);
#endif

    inline int ValidFace(int i, Element *opp, int mid);  // 1 if opp is valid...

};
void Pio(Piostream&, Element*&);

struct DirichletBC {
    SurfaceHandle fromsurf;
    double value;
    DirichletBC(const SurfaceHandle&, double);
};

// get rid of Datatype inheritance
// should make things more efficient...

struct OldNode : public Datatype {
    Point p;
    OldNode(const Point&);
    Array1<int> elems;

    DirichletBC* bc;
    int fluxBC;

    OldNode(const OldNode&);
    virtual ~OldNode();
    virtual void io(Piostream&);
    virtual OldNode* clone();
    static PersistentTypeID type_id;
    void* operator new(size_t);
    void operator delete(void*, size_t);
};

// new type...
struct Node {
    Point p;
    Node(const Point&);
#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY
    Array1<int> elems;
#else
    int elem; // a element that contains this node

    // below uses queus - really slow, low mem, threadsafe
    void GetElems(Array1<int>& build, Mesh*, int me);

    // below is not thread safe - uses generation field in elems
    void GetElemsNT(Array1<int>& build, Mesh*, int me, 
		    int check,// set to -1 or to elem to start search
		    int from, // set to -1 to start search...
		    int gen);
    
    // below is same as above, but uses an external gen array
    void GetElemsNT(Array1<int>& build, Mesh*, int me, 
		    int check,// set to -1 or to elem to start search
		    int from, // set to -1 to start search...
		    int gen,
		    Array1<unsigned int> &genA);
    
    // builds an array of all of the node in the 1 neighborhood
    // of this node - not including source node!
    // below is also not thread safe...
    void GetNodesNT(Array1<int>& build, Mesh*, int me,
		    int check, // set to -1 to start
		    int from,  // ditto
		    int gen,
		    Array1<unsigned int> &NgenA); // node gen array
    
    // below is same as above, but threadsafe if arrays are different
    void GetNodesNT(Array1<int>& build, Mesh*, int me,
		    int check, // set to -1 to start
		    int from,  // ditto
		    int gen,
		    Array1<unsigned int> &NgenA,  // node gen array
		    Array1<unsigned int> &genA);

    // builds arrays of all of the boundary edges, above
    // can give you "bristle" from source - these are
    // part of the 2 edge neighborhood
    // below is not thread safe...
    void GetEdgesNT(Array1<int> &builds, Array1<int> &buildd,
		    Mesh *, int me, 
		    int check, // -1 for start
		    int gen);
    // thread safe version of above
    void GetEdgesNT(Array1<int> &builds, Array1<int> &buildd,
		    Mesh *, int me, 
		    int check, // -1 for start
		    int gen,
		    Array1< unsigned int> &genA);


    // the rest of the 2 edge neighborhood would need to be generted
    // by hash tables are a extremely complex algorithm - hugues
    // said he doesn't compute those anyways...

#endif

    DirichletBC* bc;

    Node(const Node&);
    ~Node();
    void* operator new(size_t);
    void operator delete(void*, size_t);
};

struct NodeVersion1 {
    Point p;
};
void Pio(Piostream& stream, NodeVersion1& node);

void Pio(Piostream& stream, Node*& node);

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
};

class Mesh;
typedef LockingHandle<Mesh> MeshHandle;

class Mesh : public Datatype {
public:
    Array1<int> ids;
//    Array1<NodeHandle> nodes;
    Array1<Node*> nodes; // changed from handle...
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

    void detach_nodes();   // kind of a bogus function now
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

    int insert_delaunay(int node, GeometryOPort* ogeom=0);
    int insert_delaunay(const Point&, GeometryOPort* ogeom=0);
    void remove_delaunay(int node, int fill);
    void pack_nodes();
    void pack_elems();
    void pack_all();
    int face_idx(int, int);
    void add_node_neighbors(int node, Array1<int>& idx, int apBC=1);
    void new_element(Element* ne, HashTable<Face, int> *new_faces);
    void remove_all_elements();
    void get_boundary_lines(Array1<Point>& lines);

    void draw_element(int, GeomGroup*);
    void draw_element(Element* e, GeomGroup*);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

// you must have computed this stuff...

inline int Element::face(int i)
{
#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY
    if(faces[i] == -2){
	int i1=n[(i+1)%4];
        int i2=n[(i+2)%4];
	int i3=n[(i+3)%4];
#if 0
	Node* n1=mesh->nodes[i1].get_rep();
	Node* n2=mesh->nodes[i2].get_rep();
	Node* n3=mesh->nodes[i3].get_rep();
#else
	Node* n1=mesh->nodes[i1];
	Node* n2=mesh->nodes[i2];
	Node* n3=mesh->nodes[i3];
#endif
	// Compute it...
	faces[i]=mesh->unify(this, n1->elems, n2->elems, n3->elems);
    }
#endif
    return faces[i];
}

inline int Element::ValidFace(int i, Element *opp, int mid)
{
  int nd_check = n[i];
  int nmatch=0;

  for(int index=0;index<4;index++) {
    if (opp->n[index] == n[(i+1)%4]) nmatch++;
    if (opp->n[index] == n[(i+2)%4]) nmatch++;
    if (opp->n[index] == n[(i+3)%4]) nmatch++;
    if (opp->n[index] == nd_check) {
      if (opp->faces[index] != mid)
	return 0; // pointing to wrong element!
    }
  }

  return (nmatch == 3);
}

#endif
