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

#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/ColumnMatrix.h>
#include <Malloc/Allocator.h>
#include <Math/Mat.h>
#include <iostream.h>
#include <fstream.h>

static TrivialAllocator Element_alloc(sizeof(Element));
static TrivialAllocator Node_alloc(sizeof(Node));

static Persistent* make_Mesh()
{
    return scinew Mesh;
}

static Persistent* make_Node()
{
    return new Node(Point(0,0,0));
}

PersistentTypeID Mesh::type_id("Mesh", "Datatype", make_Mesh);
PersistentTypeID Node::type_id("Node", "Datatype", make_Node);

Mesh::Mesh()
: have_all_neighbors(0)
{
    cond_tensors.grow(1);
    cond_tensors[0].grow(6);
    cond_tensors[0][0]=1;
    cond_tensors[0][1]=0;
    cond_tensors[0][2]=0;
    cond_tensors[0][3]=1;
    cond_tensors[0][4]=0;
    cond_tensors[0][5]=1;
}

Mesh::Mesh(int nnodes, int nelems)
: nodes(nnodes), elems(nelems), have_all_neighbors(0)
{
    cond_tensors.grow(1);
    cond_tensors[0].grow(6);
    cond_tensors[0][0]=1;
    cond_tensors[0][1]=0;
    cond_tensors[0][2]=0;
    cond_tensors[0][3]=1;
    cond_tensors[0][4]=0;
    cond_tensors[0][5]=1;
}

Mesh::Mesh(const Mesh& copy)
: nodes(copy.nodes), elems(copy.elems.size()),
  cond_tensors(copy.cond_tensors), have_all_neighbors(copy.have_all_neighbors)
{
    int nelems=elems.size();
    for(int i=0;i<nelems;i++){
	Element* e=new Element(*copy.elems[i], this);
	elems[i]=e;
    }
}

Mesh::~Mesh()
{
    for(int i=0;i<elems.size();i++)
	if(elems[i])
	    delete elems[i];
}

Mesh* Mesh::clone()
{
    return scinew Mesh(*this);
}

#define MESH_VERSION 2

void Pio(Piostream& stream, NodeVersion1& node)
{
    stream.begin_cheap_delim();
    Pio(stream, node.p);
    stream.end_cheap_delim();
}

void Mesh::io(Piostream& stream)
{
    int version=stream.begin_class("Mesh", MESH_VERSION);
    if(version == 1){
	Array1<NodeVersion1> tmpnodes;
	Pio(stream, tmpnodes);
	nodes.resize(tmpnodes.size());
	for(int i=0;i<tmpnodes.size();i++)
	    nodes[i]=new Node(tmpnodes[i].p);
    } else {
	Pio(stream, nodes);
    }
    Pio(stream, elems);
    stream.end_class();
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
    Pio(stream, data->n[0]);
    Pio(stream, data->n[1]);
    Pio(stream, data->n[2]);
    Pio(stream, data->n[3]);
    stream.end_cheap_delim();
}

#define NODE_VERSION 2

void Node::io(Piostream& stream)
{
    /* int version= */ stream.begin_class("Node", NODE_VERSION);
    Pio(stream, p);
    stream.end_class();
}

Node::~Node()
{
}

Element::Element(Mesh* mesh, int n1, int n2, int n3, int n4)
: cond(0), mesh(mesh)
{
    n[0]=n1; n[1]=n2; n[2]=n3; n[3]=n4;
    faces[0]=faces[1]=faces[2]=faces[3]=-2;
}

Element::Element(const Element& copy, Mesh* mesh)
: cond(copy.cond), mesh(mesh)
{
    faces[0]=copy.faces[0];
    faces[1]=copy.faces[1];
    faces[2]=copy.faces[2];
    faces[3]=copy.faces[3];
    n[0]=copy.n[0];
    n[1]=copy.n[1];
    n[2]=copy.n[2];
    n[3]=copy.n[3];
}

Node::Node(const Point& p)
: p(p), elems(0, 4), ndof(1), nodetype(Interior)
{
}

Node::Node(const Node& copy)
: p(copy.p), elems(copy.elems), ndof(copy.ndof), nodetype(copy.nodetype),
  value(copy.value)
{
}

Node* Node::clone()
{
    return new Node(*this);
}

int Mesh::unify(Element* not,
		const Array1<int>& n1, const Array1<int>& n2,
		const Array1<int>& n3)
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
		if(elems[d1] != not){
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

void Element::get_sphere2(Point& cen, double& rad2)
{
    Point p0(mesh->nodes[n[0]]->p);
    Point p1(mesh->nodes[n[1]]->p);
    Point p2(mesh->nodes[n[2]]->p);
    Point p3(mesh->nodes[n[3]]->p);
    Vector v1(p1-p0);
    Vector v2(p2-p0);
    Vector v3(p3-p0);
    double c0=(p0-Point(0,0,0)).length2();
    double c1=(p1-Point(0,0,0)).length2();
    double c2=(p2-Point(0,0,0)).length2();
    double c3=(p3-Point(0,0,0)).length2();
    double mat[3][3];
    mat[0][0]=v1.x();
    mat[0][1]=v1.y();
    mat[0][2]=v1.z();
    mat[1][0]=v2.x();
    mat[1][1]=v2.y();
    mat[1][2]=v2.z();
    mat[2][0]=v3.x();
    mat[2][1]=v3.y();
    mat[2][2]=v3.z();
    double rhs[3];
    rhs[0]=(c1-c0)*0.5;
    rhs[1]=(c2-c0)*0.5;
    rhs[2]=(c3-c0)*0.5;
    matsolve3by3(mat, rhs);
    cen=Point(rhs[0], rhs[1], rhs[2]);
    rad2=(p0-cen).length2();
}

void Mesh::detach_nodes()
{
    for(int i=0;i<nodes.size();i++)
	if(nodes[i].get_rep())
	    nodes[i].detach();
}

void Mesh::compute_neighbors()
{
    // Clear old neighbors...
    for(int i=0;i<nodes.size();i++)
	if(nodes[i].get_rep())
	    nodes[i]->elems.remove_all();
    // Compute element info for nodes
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	if(elem){
	    if(nodes[elem->n[0]].get_rep())
		nodes[elem->n[0]]->elems.add(i);
	    if(nodes[elem->n[1]].get_rep())
		nodes[elem->n[1]]->elems.add(i);
	    if(nodes[elem->n[2]].get_rep())
		nodes[elem->n[2]]->elems.add(i);
	    if(nodes[elem->n[3]].get_rep())
		nodes[elem->n[3]]->elems.add(i);
	}
    }
    // Reset face neighbors
    for(i=0;i<elems.size();i++){
	if(elems[i]){
	    elems[i]->faces[0]=-2;
	    elems[i]->faces[1]=-2;
	    elems[i]->faces[2]=-2;
	    elems[i]->faces[3]=-2;
	}
    }
}

// Barycentric coordinate computation from 
// Computer-Aided Analysis and Design of
// Electromagnetic Devices
// S. Ranajeeven & H. Hoole

int Mesh::inside(const Point& p, Element* elem)
{
    Point p1(nodes[elem->n[0]]->p);
    Point p2(nodes[elem->n[1]]->p);
    Point p3(nodes[elem->n[2]]->p);
    Point p4(nodes[elem->n[3]]->p);
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
    if(s1<-1.e-6)
	return 0;

    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    double s2=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
    if(s2<-1.e-6)
	return 0;

    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    double s3=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
    if(s3<-1.e-6)
	return 0;

    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    double s4=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
    if(s4<-1.e-6)
	return 0;
    return 1;
}

void Mesh::get_interp(Element* elem, const Point& p,
		      double& s1, double& s2, double& s3, double& s4)
{
    Point p1(nodes[elem->n[0]]->p);
    Point p2(nodes[elem->n[1]]->p);
    Point p3(nodes[elem->n[2]]->p);
    Point p4(nodes[elem->n[3]]->p);
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

double Mesh::get_grad(Element* elem, const Point&,
		    Vector& g1, Vector& g2, Vector& g3, Vector& g4)
{
    Point p1(nodes[elem->n[0]]->p);
    Point p2(nodes[elem->n[1]]->p);
    Point p3(nodes[elem->n[2]]->p);
    Point p4(nodes[elem->n[3]]->p);
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

    double vol=(1./iV6)/6.0;
    return(vol);
}

void print_element(Element* e, Mesh* mesh)
{
    cerr << "Element is composed of nodes: " << e->n[0] << ", " << e->n[1] << ", " << e->n[2] << ", " << e->n[3] << endl;
    for(int i=0;i<4;i++){
	int nn=e->n[i];
	NodeHandle& n=mesh->nodes[nn];
	cerr << nn << ": " << n->p << endl;
    }
}

void dump_mesh(Mesh* mesh)
{
    ofstream out("mesh.dump");
    out << "Nodes:" << endl;
    for(int i=0;i<mesh->nodes.size();i++){
	Node* n=mesh->nodes[i].get_rep();
	if(!n)continue;
	out << i << ": " << n->p;
	for(int ii=0;ii<n->elems.size();ii++){
	    out << n->elems[ii] << " ";
	}
	out << endl;
    }
    out << "Elements:" << endl;
    for(i=0;i<mesh->elems.size();i++){
	Element* e=mesh->elems[i];
	if(!e)continue;
	out << i << ": " << e->n[0] << " " << e->n[1] << " " << e->n[2] << e->n[3] << "(" << e->faces[0] << " " << e->faces[1] << " " << e->faces[2] << " " << e->faces[3] << ")" << endl;
    }
}

int Mesh::locate(const Point& p, int& ix)
{
    int i=0;
    while(!elems[i])i++;
#if 0
    Array1<int> beenhere(elems.size());
    for(int ii=0;ii<beenhere.size();ii++)
	beenhere[ii]=0;
#endif
    while(1){
	Element* elem=elems[i];
#if 0
	if(beenhere[i]){
	    if(beenhere[i]>1){
		dump_mesh(this);
		ASSERT(0);
	    }
	    cerr << "We have already been to element: " << i << endl;
	    cerr << "Neighbors are: " << endl;
	    cerr << elem->face(0) << endl;
	    cerr << elem->face(1) << endl;
	    cerr << elem->face(2) << endl;
	    cerr << elem->face(3) << endl;
	    print_element(elems[i], this);
	    cerr << endl;
	}
	beenhere[i]++;
#endif
	Point p1(nodes[elem->n[0]]->p);
	Point p2(nodes[elem->n[1]]->p);
	Point p3(nodes[elem->n[2]]->p);
	Point p4(nodes[elem->n[3]]->p);
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
	if(s1<-1.e-6){
	    i=elem->face(0);
	    if(i==-1)
		return 0;
	    continue;
	}

	double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
	double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
	double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
	double s2=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
	if(s2<-1.e-6){
	    i=elem->face(1);
	    if(i==-1)
		return 0;
	    continue;
	}

	double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
	double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
	double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
	double s3=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
	if(s3<-1.e-6){
	    i=elem->face(2);
	    if(i==-1)
		return 0;
	    continue;
	}

	double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
	double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
	double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
	double s4=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
	if(s4<-1.e-6){
	    i=elem->face(3);
	    if(i==-1)
		return 0;
	    continue;
	}
	ix=i;
	return 1;
    }
//    return 0;
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

int Element::orient()
{
    double sgn=volume();
    if(sgn< 0.0){
	// Switch two of the edges so that the volume is positive
	int tmp=n[0];
	n[0]=n[1];
	n[1]=tmp;
	sgn=-sgn;
    }
    if(sgn < 1.e-6){
	return 0; // Degenerate...
    } else {
	return 1;
    }
}

double Element::volume()
{
    Point p1(mesh->nodes[n[0]]->p);
    Point p2(mesh->nodes[n[1]]->p);
    Point p3(mesh->nodes[n[2]]->p);
    Point p4(mesh->nodes[n[3]]->p);
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
    return (a1+a2+a3+a4)/6.;
}

void Mesh::get_bounds(Point& min, Point& max)
{
    min=nodes[0]->p;
    max=nodes[0]->p;
    for(int i=1;i<nodes.size();i++){
	min=Min(min, nodes[i]->p);
	max=Max(max, nodes[i]->p);
    }
}    

Face::Face(int n0, int n1, int n2)
{
    n[0]=n0;
    n[1]=n1;
    n[2]=n2;
    if(n[0] < n[1]){
	int tmp=n[0]; n[0]=n[1]; n[1]=tmp;
    }
    if(n[0] < n[2]){
	int tmp=n[0]; n[0]=n[2]; n[2]=tmp;
    }
    if(n[1] < n[2]){
	int tmp=n[1]; n[1]=n[2]; n[2]=tmp;
    }
}

int Face::hash(int hash_size) const
{
    return (((n[0]*7+5)^(n[1]*5+3)^(n[2]*3+1))^(3*hash_size+1))%hash_size;
}

int Face::operator==(const Face& f) const
{
    return n[0]==f.n[0] && n[1]==f.n[1] && n[2]==f.n[2];
}

Edge::Edge(int n0, int n1)
{
    if (n0 < n1)
    {
	n[0] = n0;
	n[1] = n1;
    }
    else
    {
	n[0] = n1;
	n[1] = n0;
    }
}

int Edge::hash(int hash_size) const
{
    return (((n[0]*7+5)^(n[1]*5+3))^(3*hash_size+1))%hash_size;
}

int Edge::operator==(const Edge& e) const
{
    return n[0]==e.n[0] && n[1]==e.n[1];
}

int Mesh::face_idx(int p, int f)
{
    Element* e=elems[p];
    int n=e->faces[f];
    if(n==-1)
	return -1;
    Element* ne=elems[n];
    for(int i=0;i<4;i++){
	if(ne->faces[i]==p){
	    return (n<<2)|i;
	}
    }
    cerr << "face_idx confused!\n";
    return 0;
}

#if 0
int face_idx2(Mesh* mesh, const Array1<int>& to_remove, int nr, int p, int f)
{
    Element* e=mesh->elems[p];
    int n=e->faces[f];
    if(n==-1)
	return -1;
    Element* ne=mesh->elems[n];
    for(int i=0;i<4;i++){
	if(ne->faces[i]==p){
	    for(int ii=0;ii<=nr;ii++){
		ASSERT(n!=to_remove[ii]);
	    }
	    return (n<<2)|i;
	}
    }
    cerr << "face_idx confused!\n";
    return 0;
}
#endif

int Mesh::insert_delaunay(const Point& p)
{
    int idx=nodes.size();
    nodes.add(new Node(p));
    return insert_delaunay(idx);
}

int Mesh::insert_delaunay(int node)
{
    if(!have_all_neighbors)
	compute_face_neighbors();
    Point p(nodes[node]->p);
//    cerr << "Adding node: " << node << " at point " << p << endl;
    // Find which element this node is in
    int in_element;
    if(!locate(p, in_element)){
	return 0;
    }

    Array1<int> to_remove;
    to_remove.add(in_element);

#if 0
    Element* ee=elems[in_element];
    for(int ii=0;ii<4;ii++){
	double dist2=(p-nodes[ee->n[ii]]->p).length2();
	if(dist2 < 1.e-2){
	   // We are going to call this a duplicate point.  Junk it.
	   delete nodes[node];
	   nodes[node]=0;
	   cerr << "Killing node: " << node << endl;
	   return 0;
       }
    }
#endif


    // Find it's neighbors...
    // We might be able to fix this loop to make it
    // O(N) instead of O(n^2) - use a Queue
    Array1<int> done;
    done.add(in_element);
    HashTable<Face, int> face_table;
    int i=0;
#ifdef CHECKER
    Array1<int> bound(elems.size());
    for(int iii=0;iii<elems.size();iii++)
	bound[iii]=0;
#endif
    while(i<to_remove.size()){
	// See if the neighbor should also be removed...
	int tr=to_remove[i];
	Element* e=elems[tr];
#ifdef PRINTOUT
	cerr << "Looking at " << tr << endl;
	cerr << "Tetra: " << e->n[0] << ", " << e->n[1] << ", " << e->n[2] << ", " << e->n[3] << endl;
#endif
	// Add these faces to the list of exposed faces...
	Face f1(e->n[1], e->n[2], e->n[3]);
	Face f2(e->n[2], e->n[3], e->n[0]);
	Face f3(e->n[3], e->n[0], e->n[1]);
	Face f4(e->n[0], e->n[1], e->n[2]);
	
	// If the face is in the list, remove it.
	// Otherwise, add it.
	int dummy;
#ifdef PRINTOUT
	cerr << "Removing element: " << tr << endl;
	if(face_table.lookup(f1, dummy))
	    cerr << "1a. removing face: " << f1.n[0] << ", " << f1.n[1] << ", " << f1.n[2] << endl;
	else
	    cerr << "1a. inserting face: " << f1.n[0] << ", " << f1.n[1] << ", " << f1.n[2] << endl;
	if(face_table.lookup(f2, dummy))
	    cerr << "2a. removing face: " << f2.n[0] << ", " << f2.n[1] << ", " << f2.n[2] << endl;
	else
	    cerr << "2a. inserting face: " << f2.n[0] << ", " << f2.n[1] << ", " << f2.n[2] << endl;
	if(face_table.lookup(f3, dummy))
	    cerr << "3a. removing face: " << f3.n[0] << ", " << f3.n[1] << ", " << f3.n[2] << endl;
	else
	    cerr << "3a. inserting face: " << f3.n[0] << ", " << f3.n[1] << ", " << f3.n[2] << endl;
	if(face_table.lookup(f4, dummy))
	    cerr << "4a. removing face: " << f4.n[0] << ", " << f4.n[1] << ", " << f4.n[2] << endl;
	else
	    cerr << "4a. inserting face: " << f4.n[0] << ", " << f4.n[1] << ", " << f4.n[2] << endl;
#endif


	if(face_table.lookup(f1, dummy))
	    face_table.remove(f1);
	else
	    face_table.insert(f1, face_idx(tr, 0));

	if(face_table.lookup(f2, dummy))
	    face_table.remove(f2);
	else
	    face_table.insert(f2, face_idx(tr, 1));

	if(face_table.lookup(f3, dummy))
	    face_table.remove(f3);
	else
	    face_table.insert(f3, face_idx(tr, 2));

	if(face_table.lookup(f4, dummy))
	    face_table.remove(f4);
	else
	    face_table.insert(f4, face_idx(tr, 3));

#ifdef PRINTOUT
	cerr << "Neighbors of " << tr << ":" << endl;
#endif
	for(int j=0;j<4;j++){
	    int skip=0;
	    int neighbor=e->face(j);
	    for(int ii=0;ii<done.size();ii++){
		if(neighbor==done[ii]){
#ifdef PRINTOUT
		    cerr << "Already done: " << neighbor << endl;
#endif
		    skip=1;
		    break;
		}
	    }
	    if(neighbor==-1 || neighbor==-2)
		skip=1;
	    if(!skip){
		// Process this neighbor
		if(!skip){
		    // See if this element is deleted by this point
		    Element* ne=elems[neighbor];
		    Point cen;
		    double rad2;
		    ne->get_sphere2(cen, rad2);
		    rad2*=0.999999;
		    double ndist2=(p-cen).length2();
		    if(ndist2 < rad2){
			// This one must go...
#ifdef PRINTOUT
			cerr << "Adding element: " << neighbor << endl;
#endif
			to_remove.add(neighbor);
		    } else {
#ifdef CHECKER
			bound[neighbor]++;
			if(bound[neighbor] > 1){
			    cerr << "Doing special hack!\n";
			    to_remove.add(neighbor);
			}
			cerr << "Element is OK: " << neighbor << endl;
#endif
		    }
		}
		done.add(neighbor);
	    }
	}
	i++;
    }
    // Remove the to_remove elements...
    for(i=0;i<to_remove.size();i++){
	int idx=to_remove[i];
	delete elems[idx];
	elems[idx]=0;
    }

    // Add the new elements from the faces...
    HashTableIter<Face, int> fiter(&face_table);
    
    // Make a copy of the face table.  We use the faces in there
    // To compute the new neighborhood information
    HashTable<Face, int> new_faces(face_table);
    for(fiter.first();fiter.ok();++fiter){
	Face f(fiter.get_key());
#ifdef PRINTOUT
	cerr << endl << endl;
	cerr << "New node is " << node << endl;
	cerr << "Processing face: " << f.n[0] << ", " << f.n[1] << ", " << f.n[2] << endl;
	cerr << "New element: " << elems.size() << endl;
#endif
	Element* ne=new Element(this, node, f.n[0], f.n[1], f.n[2]);
	
	// If the new element is not degenerate, add it to the mix...
	if(ne->orient()){
	    int nen=elems.size();
	    
	    // The face neighbor is in the Face data item
	    Face f1(ne->n[1], ne->n[2], ne->n[3]);
	    Face f2(ne->n[2], ne->n[3], ne->n[0]);
	    Face f3(ne->n[3], ne->n[0], ne->n[1]);
	    Face f4(ne->n[0], ne->n[1], ne->n[2]);
	    int ef;

#ifdef PRINTOUT
	    int dummy;
	if(new_faces.lookup(f1, dummy))
	    cerr << "1b. removing face: " << f1.n[0] << ", " << f1.n[1] << ", " << f1.n[2] << endl;
	else
	    cerr << "1b. inserting face: " << f1.n[0] << ", " << f1.n[1] << ", " << f1.n[2] << endl;
	if(new_faces.lookup(f2, dummy))
	    cerr << "2b. removing face: " << f2.n[0] << ", " << f2.n[1] << ", " << f2.n[2] << endl;
	else
	    cerr << "2b. inserting face: " << f2.n[0] << ", " << f2.n[1] << ", " << f2.n[2] << endl;
	if(new_faces.lookup(f3, dummy))
	    cerr << "3b. removing face: " << f3.n[0] << ", " << f3.n[1] << ", " << f3.n[2] << endl;
	else
	    cerr << "3b. inserting face: " << f3.n[0] << ", " << f3.n[1] << ", " << f3.n[2] << endl;
	if(new_faces.lookup(f4, dummy))
	    cerr << "4b. removing face: " << f4.n[0] << ", " << f4.n[1] << ", " << f4.n[2] << endl;
	else
	    cerr << "4b. inserting face: " << f4.n[0] << ", " << f4.n[1] << ", " << f4.n[2] << endl;
#endif
	    if(new_faces.lookup(f1, ef)){
		// We have this face...
		if(ef==-1){
		    ne->faces[0]=-1; // Boundary
		} else {
		    int which_face=ef%4;
		    int which_elem=ef/4;
		    ne->faces[0]=which_elem;
#ifdef PRINTOUT
		    cerr << "which_elem=" << which_elem << endl;
		    cerr << "which_face=" << which_face << endl;
#endif
		    elems[which_elem]->faces[which_face]=nen;
#ifdef CHECKER
		    Element* ee=elems[which_elem];
		    Face ff(ee->n[(which_face+1)%4], ee->n[(which_face+2)%4], ee->n[(which_face+3)%4]);
		    ASSERT(ff==f1);
#endif
		}
		new_faces.remove(f1);
	    } else {
		new_faces.insert(f1, (nen<<2)|0);
		ne->faces[0]=-3;
	    }
	    if(new_faces.lookup(f2, ef)){
		// We have this face...
		if(ef==-1){
		    ne->faces[1]=-1; // Boundary;
		} else {
		    int which_face=ef%4;
		    int which_elem=ef/4;
		    ne->faces[1]=which_elem;
		    elems[which_elem]->faces[which_face]=nen;
#ifdef CHECKER
		    Element* ee=elems[which_elem];
		    Face ff(ee->n[(which_face+1)%4], ee->n[(which_face+2)%4], ee->n[(which_face+3)%4]);
		    ASSERT(ff==f2);
#endif
		}
		new_faces.remove(f2);
	    } else {
		new_faces.insert(f2, (nen<<2)|1);
		ne->faces[1]=-3;
	    }
	    if(new_faces.lookup(f3, ef)){
		// We have this face...
		if(ef==-1){
		    ne->faces[2]=-1; // Boundary
		} else {
		    int which_face=ef%4;
		    int which_elem=ef/4;
		    ne->faces[2]=which_elem;
		    elems[which_elem]->faces[which_face]=nen;
#ifdef CHECKER
		    Element* ee=elems[which_elem];
		    Face ff(ee->n[(which_face+1)%4], ee->n[(which_face+2)%4], ee->n[(which_face+3)%4]);
		    ASSERT(ff==f3);
#endif
		}
		new_faces.remove(f3);
	    } else {
		new_faces.insert(f3, (nen<<2)|2);
		ne->faces[2]=-3;
	    }
	    if(new_faces.lookup(f4, ef)){
		// We have this face...
		if(ef==-1){
		    ne->faces[3]=-1;
		} else {
		    int which_face=ef%4;
		    int which_elem=ef/4;
		    ne->faces[3]=which_elem;
		    elems[which_elem]->faces[which_face]=nen;
#ifdef CHECKER
		    Element* ee=elems[which_elem];
		    Face ff(ee->n[(which_face+1)%4], ee->n[(which_face+2)%4], ee->n[(which_face+3)%4]);
		    ASSERT(ff==f4);
#endif
		}
		new_faces.remove(f4);
	    } else {
		new_faces.insert(f4, (nen<<2)|3);
		ne->faces[3]=-3;
	    }
	    elems.add(ne);
	} else {
	    cerr << "Degenerate element (node=" << node << ")\n";
	    return 0;
	}
    }
    // Check the consistency of the mesh....
#ifdef CHECKER
    HashTable<Face, int> facetab;
    for(i=0;i<elems.size();i++){
	Element* e=elems[i];
	if(e){
	    Face f1(e->n[1], e->n[2], e->n[3]);
	    Face f2(e->n[2], e->n[3], e->n[0]);
	    Face f3(e->n[3], e->n[0], e->n[1]);
	    Face f4(e->n[0], e->n[1], e->n[2]);
	    int n=0;
	    int dummy;
	    if(facetab.lookup(f1, dummy)){
		n=dummy;
		facetab.remove(f1);
	    }
	    n++;
	    ASSERT(n<=2);
	    n=0;
	    facetab.insert(f1, n);
	    if(facetab.lookup(f2, dummy)){
		n=dummy;
		facetab.remove(f2);
	    }
	    n++;
	    ASSERT(n<=2);
	    n=0;
	    facetab.insert(f2, n);
	    if(facetab.lookup(f3, dummy)){
		n=dummy;
		facetab.remove(f3);
	    }
	    n++;
	    ASSERT(n<=2);
	    n=0;
	    facetab.insert(f3, n);
	    if(facetab.lookup(f4, dummy)){
		n=dummy;
		facetab.remove(f4);
	    }
	    n++;
	    ASSERT(n<=2);
	    facetab.insert(f4, n);
	}
    }
#endif
    return 1;
}

void Mesh::pack_elems()
{
    // Pack the elements...
    int nelems=elems.size();
    int idx=0;
    Array1<int> map(nelems);
    for(int i=0;i<nelems;i++){
	Element* e=elems[i];
	if(e){
	    map[i]=idx;
	    elems[idx++]=e;
	} else {
	    map[i]=-1234;
	}
    }
    elems.resize(idx);
    int nnodes=nodes.size();
    for(i=0;i<nnodes;i++){
	NodeHandle&  n=nodes[i];
	if(n.get_rep()){
	    int ne=n->elems.size();
	    for(int j=0;j<ne;j++){
		int elem=n->elems[j];
		int new_elem=map[elem];
		if(new_elem == -1234)
		    cerr << "Warning: pointing to old element: " << elem << endl;
		n->elems[j]=new_elem;
	    }
	}
    }
    for(i=0;i<elems.size();i++){
	Element* e=elems[i];
	for(int j=0;j<4;j++){
	    int face=e->faces[j];
	    if(face>=0){
		if(map[face] == -1234)
		    cerr << "Warning: pointing to old element: " << e->faces[j] << endl;
		e->faces[j]=map[face];
	    }
	}
    }
}

void Mesh::pack_nodes()
{
    // Pack the elements...
    int nnodes=nodes.size();
    int idx=0;
    Array1<int> map(nnodes);
    for(int i=0;i<nnodes;i++){
	NodeHandle& n=nodes[i];
	if(n.get_rep()){
	    map[i]=idx;
	    nodes[idx++]=n;
	} else {
	    map[i]=-1234;
	}
    }
    nodes.resize(idx);
    int nelems=elems.size();
    for(i=0;i<nelems;i++){
	Element* e=elems[i];
	if(e){
	    for(int j=0;j<4;j++){
		if(map[e->n[j]]==-1234)
		    cerr << "Warning: pointing to old node: " << e->n[j] << endl;
		e->n[j]=map[e->n[j]];
	    }
	}
    }
}

void Mesh::pack_all()
{
    compute_neighbors();
    pack_nodes();
    pack_elems();
}

void Mesh::remove_delaunay(int node, int fill)
{
    if(!fill){
	NodeHandle& n=nodes[node];
	for(int i=0;i<n->elems.size();i++){
	    if(elems[n->elems[i]]){
		delete elems[n->elems[i]];
		elems[n->elems[i]]=0;
	    }
	}
	nodes[node]=0;
    } else {
	NOT_FINISHED("Mesh::remove_delaunay");
    }
}

void Mesh::compute_face_neighbors()
{
    // This could probably be done much faster...
    for(int i=0;i<elems.size();i++){
	if(elems[i]){
	    for(int j=0;j<4;j++){
		elems[i]->face(j);
	    }
	}
    }
    have_all_neighbors=1;
}

void Mesh::add_node_neighbors(int node, Array1<int>& idx)
{
    NodeHandle& n=nodes[node];
    int ne=n->elems.size();
    Array1<int> neighbor_nodes(4*ne+1);
    int nodesi=0;
    // Gather all of the nodes
    for(int i=0;i<ne;i++){
	int ei=n->elems[i];
	Element* e=elems[ei];
	for(int j=0;j<4;j++){
	    int n=e->n[j];
	    neighbor_nodes[nodesi++]=n;
	}
    }
    // Sort it...
    // We should get shot for this..
    for(i=0;i<nodesi-1;i++){
	for(int j=i+1;j<nodesi;j++){
	    if(neighbor_nodes[i] > neighbor_nodes[j]){
		int tmp=neighbor_nodes[i];
		neighbor_nodes[i]=neighbor_nodes[j];
		neighbor_nodes[j]=tmp;
	    }
	}
    }
    // Find the unique set...
    for(i=0;i<nodesi;i++){
	if(i==0 || neighbor_nodes[i] != neighbor_nodes[i-1])
	    idx.add(neighbor_nodes[i]);
    }
}
