
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
#include <Geometry/Point.h>

#include <stdlib.h> // For size_t

class Mesh;

struct Element {
    int faces[4];
    int n[4];
    int cond; // index to the conductivities array for the cond
              // tensor of this element
    Mesh* mesh;
    Element(Mesh*, int, int, int, int);
    Element(const Element&, Mesh* mesh);
    void* operator new(size_t);
    void operator delete(void*, size_t);
    int face(int);

    double volume();
    int orient();
    void get_sphere(Point& cen, double& rad);
    void get_sphere2(Point& cen, double& rad2);
};

void Pio(Piostream&, Element*&);

struct Node;
typedef LockingHandle<Node> NodeHandle;

struct Node : public Datatype {
    Point p;
    Node(const Point&);
    Array1<int> elems;

    int ndof;
    enum NodeType {
	VSource,
	ISource,
	Interior,
    };
    NodeType nodetype;
    double value;
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
void Pio(Piostream& stream, NodeVersion1& node);

struct Face {
    int n[3];
    Face(int, int, int);
    int hash(int hash_size) const;
    int operator==(const Face&) const;
};

class Mesh;
typedef LockingHandle<Mesh> MeshHandle;

class Mesh : public Datatype {
public:
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

    void compute_neighbors();
    int locate(const Point&, int&);
    int inside(const Point& p, Element* elem);
    void get_interp(Element* elem, const Point& p, double& s1,
		    double& s2, double& s3, double& s4);
    double get_grad(Element* elem, const Point& p, Vector& g1,
		  Vector& g2, Vector& g3, Vector& g4);
    void get_bounds(Point& min, Point& max);
    int unify(Element*, const Array1<int>&, const Array1<int>&,
	      const Array1<int>&);

    int insert_delaunay(int node);
    int insert_delaunay(const Point& p);
    void remove_delaunay(int node, int fill);
    void pack_nodes();
    void pack_elems();
    int face_idx(int, int);
    void add_node_neighbors(int node, Array1<int>& idx);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
