
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

#include <Mesh.h>
#include <Classlib/String.h>
#include <iostream.h>

static Persistent* make_Mesh()
{
    return new Mesh;
}
PersistentTypeID Mesh::typeid("Mesh", "Datatype", make_Mesh);

Mesh::Mesh()
{
}

Mesh::Mesh(int nnodes, int nelems)
: nodes(nnodes), elems(nelems)
{
}

Mesh::~Mesh()
{
}

#define MESH_VERSION 1

void Mesh::io(Piostream& stream)
{
    int version=stream.begin_class("Mesh", MESH_VERSION);
    Pio(stream, nodes);
    Pio(stream, elems);
    stream.end_class();
    if(stream.reading())
	compute_neighbors();
}

void Pio(Piostream& stream, Element*& data)
{
    if(stream.reading())
	data=new Element(0,0,0,0);
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

Element::Element(int n1, int n2, int n3, int n4)
: n1(n1), n2(n2), n3(n3), n4(n4)
{
}

Node::Node(const Point& p)
: p(p), elems(0, 4)
{
}

static int unify(int not, const Array1<int>& n1, const Array1<int>& n2, const Array1<int>& n3)
{
    for(int i=0;i<n1.size();i++){
	int idx=n1[i];
	if(idx != not){
	    for(int j=0;j<n2.size();j++){
		if(n2[j] == idx){
		    // Look in n3
		    for(int k=0;k<n3.size();k++){
			if(n3[k] == idx)
			    return idx;
		    }
		}
	    }
	}
    }
}

void Mesh::compute_neighbors()
{
    // Clear old neighbors...
    for(int i=0;i<nodes.size();i++)
	nodes[i]->elems.remove_all();
    // Compute element info for nodes
    cerr << "Computing element info...\n";
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	Node* n1=nodes[elem->n1];
	Node* n2=nodes[elem->n2];
	Node* n3=nodes[elem->n3];
	Node* n4=nodes[elem->n4];
     	int haveit=0;
	for(int ii=0;ii<n1->elems.size();ii++){
	    if(n1->elems[ii] == i){
		haveit=1;
		break;
	    }
	}
	if(!haveit)
	    n1->elems.add(i);
	haveit=0;
	for(ii=0;ii<n2->elems.size();ii++){
	    if(n2->elems[ii] == i){
		haveit=1;
		break;
	    }
	}
	if(!haveit)
	    n2->elems.add(i);
	haveit=0;
	for(ii=0;ii<n3->elems.size();ii++){
	    if(n3->elems[ii] == i){
		haveit=1;
		break;
	    }
	}
	if(!haveit)
	    n3->elems.add(i);
	haveit=0;
	for(ii=0;ii<n4->elems.size();ii++){
	    if(n4->elems[ii] == i){
		haveit=1;
		break;
	    }
	}
	if(!haveit)
	    n4->elems.add(i);
    }
    cerr << "Computing face neighbors...\n";
    // Compute face neighbors
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	Node* n1=nodes[elem->n1];
	Node* n2=nodes[elem->n2];
	Node* n3=nodes[elem->n3];
	Node* n4=nodes[elem->n4];
	elem->face[0]=unify(i, n2->elems, n3->elems, n4->elems);
	elem->face[1]=unify(i, n1->elems, n3->elems, n4->elems);
	elem->face[2]=unify(i, n1->elems, n2->elems, n4->elems);
	elem->face[3]=unify(i, n1->elems, n2->elems, n3->elems);
    }
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
