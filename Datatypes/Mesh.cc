
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

#include <Datatypes/Mesh.h>

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/Timer.h>
#include <Classlib/TrivialAllocator.h>
#include <iostream.h>

static TrivialAllocator Element_alloc(sizeof(Element));
static TrivialAllocator Node_alloc(sizeof(Node));

static Persistent* make_Mesh()
{
    return new Mesh;
}

PersistentTypeID Mesh::type_id("Mesh", "Datatype", make_Mesh);

Mesh::Mesh()
{
}

Mesh::Mesh(int nnodes, int nelems)
: nodes(nnodes), elems(nelems)
{
}

Mesh::~Mesh()
{
    for(int i=0;i<nodes.size();i++)
	delete nodes[i];
    for(i=0;i<elems.size();i++)
	delete elems[i];
}

Mesh* Mesh::clone()
{
    NOT_FINISHED("Mesh::clone()");
    return 0;
}

#define MESH_VERSION 1

void Mesh::io(Piostream& stream)
{
    int version=stream.begin_class("Mesh", MESH_VERSION);
    CPUTimer t1;
    t1.start();
    Pio(stream, nodes);
    Pio(stream, elems);
    stream.end_class();
    t1.stop();
    cerr << "Read time: " << t1.time() << endl;
    if(stream.reading()){
	for(int i=0;i<elems.size();i++)
	    elems[i]->mesh=this;
	compute_neighbors();
    }
}

void Pio(Piostream& stream, Element*& data)
{
    if(stream.reading())
	data=new Element(0, 0,0,0,0);
    stream.begin_cheap_delim();
    Pio(stream, data->n1);
    Pio(stream, data->n2);
    Pio(stream, data->n3);
    Pio(stream, data->n4);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, Node*& data)
{
    if(stream.reading())
	data=new Node(Point(0,0,0));
    stream.begin_cheap_delim();
    Pio(stream, data->p);
    stream.end_cheap_delim();
}

Element::Element(Mesh* mesh, int n1, int n2, int n3, int n4)
: mesh(mesh), n1(n1), n2(n2), n3(n3), n4(n4)
{
}

Node::Node(const Point& p)
: p(p), elems(0, 4)
{
}

static int unify(int not, const Array1<int>& n1, const Array1<int>& n2, const Array1<int>& n3)
{
    int s1=n1.size();
    int s2=n2.size();
    int s3=n3.size();
    int i1=0;
    int i2=0;
    int i3=0;
    while(i1<s1 && i2<s2 && i3<s3){
	int d1=n1[i1];
	int d2=n2[i2];
	int d3=n3[i3];
	if(d1==d2){
	    if(d2==d3){
		if(d1 != not){
		    // Found it...
		    return d1;
		} else {
		    i1++;
		    i2++;
		    i3++;
		}
	    } else if(d3<d1){
		i3++;
	    } else {
		i1++;
		i2++;
	    }
	} else if(d1<d2){
	    if(d1<d3){
		i1++;
	    } else {
		i3++;
	    }
	} else {
	    if(d2<d3){
		i2++;
	    } else {
		i3++;
	    }
	}
    }
    return -1;
}

int Element::face(int i)
{
    if(faces[i] == -2){
	int* idx=&n1;
	int i1=idx[(i+1)%4];
        int i2=idx[(i+2)%4];
	int i3=idx[(i+3)%4];
	Node* n1=mesh->nodes[i1];
	Node* n2=mesh->nodes[i2];
	Node* n3=mesh->nodes[i3];
	// Compute it...
	unify(i, n1->elems, n2->elems, n3->elems);
    }
    return faces[i];
}

void Mesh::compute_neighbors()
{
    // Clear old neighbors...
    for(int i=0;i<nodes.size();i++)
	nodes[i]->elems.remove_all();
    // Compute element info for nodes
    cerr << "Computing element info...\n";
    CPUTimer t1;
    t1.start();
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	Node* n1=nodes[elem->n1];
	Node* n2=nodes[elem->n2];
	Node* n3=nodes[elem->n3];
	Node* n4=nodes[elem->n4];
	n1->elems.add(i);
	n2->elems.add(i);
	n3->elems.add(i);
	n4->elems.add(i);
    }
    t1.stop();
    cerr << "Element info time: " << t1.time() << endl;
    t1.clear();
    t1.start();
    cerr << "Computing face neighbors...\n";
    // Compute face neighbors
    for(i=0;i<elems.size();i++){
	elems[i]->faces[0]=-2;
	elems[i]->faces[1]=-2;
	elems[i]->faces[2]=-2;
	elems[i]->faces[3]=-2;
    }
#if 0
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	Node* n1=nodes[elem->n1];
	Node* n2=nodes[elem->n2];
	Node* n3=nodes[elem->n3];
	Node* n4=nodes[elem->n4];
	elem->faces[0]=unify(i, n2->elems, n3->elems, n4->elems);
	elem->faces[1]=unify(i, n1->elems, n3->elems, n4->elems);
	elem->faces[2]=unify(i, n1->elems, n2->elems, n4->elems);
	elem->faces[3]=unify(i, n1->elems, n2->elems, n3->elems);
    }
#endif
    t1.stop();
    cerr << "Face neighbors time: " << t1.time() << endl;
    cerr << "compute_neighbors done\n";
}

// Barycentric coordinate computation from 
// Computer-Aided Analysis and Design of
// Electromagnetic Devices
// S. Ranajeeven & H. Hoole

int Mesh::inside(const Point& p, Element* elem)
{
    Point p1(nodes[elem->n1]->p);
    Point p2(nodes[elem->n2]->p);
    Point p3(nodes[elem->n3]->p);
    Point p4(nodes[elem->n4]->p);
    double x1=p1.x();
    double y1=p1.y();
    double z1=p1.z();
    double x2=p2.x();
    double y2=p2.y();
    double z2=p2.z();
    double x3=p3.x();
    double y3=p3.y();
    double z3=p3.z();
    double x4=p4.x();
    double y4=p4.y();
    double z4=p4.z();
    double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
    double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
    double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
    double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
    double iV6=1./(a1+a2+a3+a4);

    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
    double s1=iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());
    if(s1<0)
	return 0;

    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    double s2=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
    if(s2<0)
	return 0;

    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    double s3=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
    if(s3<0)
	return 0;

    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    double s4=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
    if(s4<0)
	return 0;
    return 1;
}

void Mesh::get_interp(Element* elem, const Point& p,
		      double& s1, double& s2, double& s3, double& s4)
{
    Point p1(nodes[elem->n1]->p);
    Point p2(nodes[elem->n2]->p);
    Point p3(nodes[elem->n3]->p);
    Point p4(nodes[elem->n4]->p);
    double x1=p1.x();
    double y1=p1.y();
    double z1=p1.z();
    double x2=p2.x();
    double y2=p2.y();
    double z2=p2.z();
    double x3=p3.x();
    double y3=p3.y();
    double z3=p3.z();
    double x4=p4.x();
    double y4=p4.y();
    double z4=p4.z();
    double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
    double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
    double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
    double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
    double iV6=1./(a1+a2+a3+a4);

    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
    s1=iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());
    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    s2=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    s3=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    s4=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
}

void Mesh::get_grad(Element* elem, const Point&,
		    Vector& g1, Vector& g2, Vector& g3, Vector& g4)
{
    Point p1(nodes[elem->n1]->p);
    Point p2(nodes[elem->n2]->p);
    Point p3(nodes[elem->n3]->p);
    Point p4(nodes[elem->n4]->p);
    double x1=p1.x();
    double y1=p1.y();
    double z1=p1.z();
    double x2=p2.x();
    double y2=p2.y();
    double z2=p2.z();
    double x3=p3.x();
    double y3=p3.y();
    double z3=p3.z();
    double x4=p4.x();
    double y4=p4.y();
    double z4=p4.z();
    double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
    double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
    double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
    double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
    double iV6=1./(a1+a2+a3+a4);

    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
    g1=Vector(b1*iV6, c1*iV6, d1*iV6);
    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    g2=Vector(b2*iV6, c2*iV6, d2*iV6);
    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    g3=Vector(b3*iV6, c3*iV6, d3*iV6);
    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    g4=Vector(b4*iV6, c4*iV6, d4*iV6);
}

int Mesh::locate(const Point& p, int& ix)
{
    for(int i=0;i<elems.size();i++){
	if(inside(p, elems[i])){
	    ix=i;
	    return 1;
	}
    }
    return 0;
}

void* Element::operator new(size_t)
{
    return Element_alloc.alloc();
}

void Element::operator delete(void* rp, size_t)
{
    Element_alloc.free(rp);
}

void* Node::operator new(size_t)
{
    return Node_alloc.alloc();
}

void Node::operator delete(void* rp, size_t)
{
    Node_alloc.free(rp);
}
