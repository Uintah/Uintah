
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
    int n1;
    int n2;
    int n3;
    int n4;
    int cond; // index to the conductivities array for the cond
              // tensor of this element
    Mesh* mesh;
    Element(Mesh*, int, int, int, int);
    void* operator new(size_t);
    void operator delete(void*, size_t);
    int face(int);
};

void Pio(Piostream&, Element*&);

struct Node {
    Point p;
    Node(const Point&);
    Array1<int> elems;
    void* operator new(size_t);
    void operator delete(void*, size_t);
};

void Pio(Piostream&, Node*&);

class Mesh;
typedef LockingHandle<Mesh> MeshHandle;

class Mesh : public Datatype {
public:
    Array1<Node*> nodes;
    Array1<Element*> elems;
    Array1<Array1<double> > cond_tensors;
    Mesh();
    Mesh(int nnodes, int nelems);
    virtual Mesh* clone();
    virtual ~Mesh();

    void compute_neighbors();
    int locate(const Point&, int&);
    int inside(const Point& p, Element* elem);
    void get_interp(Element* elem, const Point& p, double& s1,
		    double& s2, double& s3, double& s4);
    void get_grad(Element* elem, const Point& p, Vector& g1,
		  Vector& g2, Vector& g3, Vector& g4);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
