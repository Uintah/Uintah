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

#include <Classlib/FastHashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/ColumnMatrix.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Sphere.h>
#include <Geom/Polyline.h>
#include <Geom/Tri.h>
#include <Malloc/Allocator.h>
#include <Math/Mat.h>
#include <iostream.h>
#include <fstream.h>

static TrivialAllocator Element_alloc(sizeof(Element));
static TrivialAllocator Node_alloc(sizeof(Node));
static TrivialAllocator OldNode_alloc(sizeof(OldNode));
static TrivialAllocator Face_alloc(sizeof(Face));
static TrivialAllocator DFace_alloc(sizeof(Face));

struct DFace {
    DFace* next;
    int hash;
    int face_idx;
    int n[3];
    inline DFace(int n0, int n1, int n2) {
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
	hash=(n[0]*7+5)^(n[1]*5+3)^(n[2]*3+1);
    }
    inline int operator==(const DFace& f) const {
	return n[0]==f.n[0] && n[1]==f.n[1] && n[2]==f.n[2];
    }

    inline void* operator new(size_t) {
	return DFace_alloc.alloc();
    }
    void operator delete(void* rp, size_t) {
	DFace_alloc.free(rp);
    }
};

static Persistent* make_Mesh()
{
    return scinew Mesh;
}

static Persistent* make_Node()
{
    return new OldNode(Point(0,0,0));
}

PersistentTypeID Mesh::type_id("Mesh", "Datatype", make_Mesh);
PersistentTypeID OldNode::type_id("Node", "Datatype", make_Node);

Mesh::Mesh()
: have_all_neighbors(0), current_generation(2)
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
: nodes(nnodes), elems(nelems), have_all_neighbors(0), current_generation(2)
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
  cond_tensors(copy.cond_tensors), have_all_neighbors(copy.have_all_neighbors),
  current_generation(2)
{
    int nelems=elems.size();
    for(int i=0;i<nelems;i++){
	Element* e=new Element(*copy.elems[i], this);
	elems[i]=e;
    }
}

Mesh::~Mesh()
{
    remove_all_elements();
}

void Mesh::remove_all_elements()
{
    for(int i=0;i<elems.size();i++)
	if(elems[i])
	    delete elems[i];
    elems.remove_all();
}

Mesh* Mesh::clone()
{
    return scinew Mesh(*this);
}

#define MESH_VERSION 2
#define MESH_VERSION 3
// new version...

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
    } else if (version == 2) {
      Array1<NodeHandle> tmpnodes;
      Pio(stream,tmpnodes);
      nodes.resize(tmpnodes.size());
      for(int i=0;i<tmpnodes.size();i++) {
	nodes[i] = new Node(tmpnodes[i]->p);
	if (tmpnodes[i]->bc)
	  nodes[i]->bc = new DirichletBC(tmpnodes[i]->bc->fromsurf,
					 tmpnodes[i]->bc->value);
      }
    } else { // mesh version is at least 3...
      Pio(stream, nodes);
    }
    Pio(stream, elems);
    stream.end_class();
    if(stream.reading()){
	for(int i=0;i<elems.size();i++){
#ifdef STORE_THE_MESH
	    elems[i]->mesh=this;
	    elems[i]->orient();
	    elems[i]->compute_basis();
#else
	    elems[i]->orient(this);
	    elems[i]->compute_basis(this);	    
#endif
	}
	compute_neighbors();
    }
}

void Pio(Piostream& stream, Element*& data)
{
    if(stream.reading())
	data=new Element(0,0,0,0,0);
    stream.begin_cheap_delim();
    Pio(stream, data->n[0]);
    Pio(stream, data->n[1]);
    Pio(stream, data->n[2]);
    Pio(stream, data->n[3]);
    stream.end_cheap_delim();
}

#define NODE_VERSION 3

void OldNode::io(Piostream& stream)
{
  int version=stream.begin_class("Node", NODE_VERSION);
  Pio(stream, p);
  if(version >= 3){
    int flag;
    if(!stream.reading()){
      flag=bc?1:0;
    }
    Pio(stream, flag);
    if(stream.reading() && flag)
      bc=new DirichletBC(0,0);
    if(flag){
      Pio(stream, bc->fromsurf);
      Pio(stream, bc->value);
    }
  }
  stream.end_class();
}

// this might have to change...

void Pio(Piostream& stream, Node*& data)
{
  if (stream.reading())
    data = new Node(Point(0,0,0));
  stream.begin_cheap_delim();  
  Pio(stream, data->p);
  // version is greater than 3, otherwise you are above...
  int flag;
  if(!stream.reading()){
    flag=data->bc?1:0;
  }
  Pio(stream, flag);
  if(stream.reading() && flag)
    data->bc=new DirichletBC(0,0);
  if(flag){
    Pio(stream, data->bc->fromsurf);
    Pio(stream, data->bc->value);
  }
  stream.end_cheap_delim();
}

Node::~Node()
{
    if(bc)
	delete bc;
}

OldNode::~OldNode()
{
    if(bc)
	delete bc;
}

#ifdef STORE_THE_MESH

Element::Element(Mesh* mesh, int n1, int n2, int n3, int n4)
: generation(0), cond(0), mesh(mesh)
{
    n[0]=n1; n[1]=n2; n[2]=n3; n[3]=n4;
    faces[0]=faces[1]=faces[2]=faces[3]=-2;

#ifdef STORE_ELEMENT_BASIS
    if(mesh)
	compute_basis();
    else
	vol=-9999;
#endif
}

#else

Element::Element(Mesh* mesh, int n1, int n2, int n3, int n4)
: generation(0), cond(0)
{
    n[0]=n1; n[1]=n2; n[2]=n3; n[3]=n4;
    faces[0]=faces[1]=faces[2]=faces[3]=-2;

#ifdef STORE_ELEMENT_BASIS
    if(mesh)
	compute_basis(mesh);
    else
	vol=-9999;
#endif
}
#endif

void Element::compute_basis(
#ifndef STORE_THE_MESH
Mesh *mesh
#endif
)
{
#ifdef STORE_ELEMENT_BASIS
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
    double iV6=1./(a1+a2+a3+a4);

    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
    g[0]=Vector(b1*iV6, c1*iV6, d1*iV6);
    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    g[1]=Vector(b2*iV6, c2*iV6, d2*iV6);
    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    g[2]=Vector(b3*iV6, c3*iV6, d3*iV6);
    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    g[3]=Vector(b4*iV6, c4*iV6, d4*iV6);
    a[0]=a1*iV6;
    a[1]=a2*iV6;
    a[2]=a3*iV6;
    a[3]=a4*iV6;

    vol=(1./iV6)/6.0;
#endif
}

Element::Element(const Element& copy, Mesh* mesh)
: generation(0), cond(copy.cond)
#ifdef STORE_THE_MESH
, mesh(mesh)
#endif
{
    faces[0]=copy.faces[0];
    faces[1]=copy.faces[1];
    faces[2]=copy.faces[2];
    faces[3]=copy.faces[3];
#ifdef STORE_ELEMENT_BASIS
    n[0]=copy.n[0];
    n[1]=copy.n[1];
    n[2]=copy.n[2];
    n[3]=copy.n[3];
    g[0]=copy.g[0];
    g[1]=copy.g[1];
    g[2]=copy.g[2];
    g[3]=copy.g[3];
    a[0]=copy.a[0];
    a[1]=copy.a[1];
    a[2]=copy.a[2];
    a[3]=copy.a[3];
    vol=copy.vol;
#endif
}

#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY

Node::Node(const Point& p)
: p(p), elems(0, 4), bc(0)
{
}

Node::Node(const Node& copy)
: p(copy.p), elems(copy.elems), bc(copy.bc?new DirichletBC(*copy.bc):0)
{
}

#else

Node::Node(const Point& p)
: p(p), elem(-1), bc(0)
{
}

Node::Node(const Node& copy)
: p(copy.p), elem(-1), bc(copy.bc?new DirichletBC(*copy.bc):0)
{
}

#endif

OldNode::OldNode(const Point& p)
: p(p), elems(0, 4), bc(0)
{
}

OldNode::OldNode(const OldNode& copy)
: p(copy.p), elems(copy.elems), bc(copy.bc?new DirichletBC(*copy.bc):0)
{
}

OldNode* OldNode::clone()
{
    return new OldNode(*this);
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

void Element::get_sphere2(Point& cen, double& rad2, double& err
#ifndef STORE_THE_MESH
,Mesh* mesh			 
#endif
)
{
    Point p0(mesh->nodes[n[0]]->p);
    Point p1(mesh->nodes[n[1]]->p);
    Point p2(mesh->nodes[n[2]]->p);
    Point p3(mesh->nodes[n[3]]->p);
    double mat[3][3];
    mat[0][0]=p1.x()-p0.x();
    mat[0][1]=p1.y()-p0.y();
    mat[0][2]=p1.z()-p0.z();
    mat[1][0]=p2.x()-p0.x();
    mat[1][1]=p2.y()-p0.y();
    mat[1][2]=p2.z()-p0.z();
    mat[2][0]=p3.x()-p0.x();
    mat[2][1]=p3.y()-p0.y();
    mat[2][2]=p3.z()-p0.z();
    double rhs[3];
    double c0=p0.x()*p0.x()+p0.y()*p0.y()+p0.z()*p0.z();
    double c1=p1.x()*p1.x()+p1.y()*p1.y()+p1.z()*p1.z();
    double c2=p2.x()*p2.x()+p2.y()*p2.y()+p2.z()*p2.z();
    double c3=p3.x()*p3.x()+p3.y()*p3.y()+p3.z()*p3.z();
    rhs[0]=(c1-c0)*0.5;
    rhs[1]=(c2-c0)*0.5;
    rhs[2]=(c3-c0)*0.5;
    double rcond;
    matsolve3by3_cond(mat, rhs, &rcond);
    if(rcond < 1.e-7){
	cerr << "WARNING - degenerate element, rcond=" << rcond << endl;
    }
    cen=Point(rhs[0], rhs[1], rhs[2]);
    rad2=(p0-cen).length2();
    //err=Max(1.e-6, 1.e-6/rcond);
    err=1.e-4;

}

Point Element::centroid(
#ifndef STORE_THE_MESH
Mesh* mesh
#endif
)
{
    Point p0(mesh->nodes[n[0]]->p);
    Point p1(mesh->nodes[n[1]]->p);
    Point p2(mesh->nodes[n[2]]->p);
    Point p3(mesh->nodes[n[3]]->p);
    return AffineCombination(p0, .25, p1, .25, p2, .25, p3, .25);
}

void Mesh::detach_nodes()
{
  // no handles for now...
#if 0  
    for(int i=0;i<nodes.size();i++)
	if(nodes[i].get_rep())
	    nodes[i].detach();

#endif
}

#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY

void Mesh::compute_neighbors()
{
    // Clear old neighbors...
    int i;
    for(i=0;i<nodes.size();i++)
      if(nodes[i])
	    nodes[i]->elems.remove_all();
    // Compute element info for nodes
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	if(elem){
	    if(nodes[elem->n[0]])
		nodes[elem->n[0]]->elems.add(i);
	    if(nodes[elem->n[1]])
		nodes[elem->n[1]]->elems.add(i);
	    if(nodes[elem->n[2]])
		nodes[elem->n[2]]->elems.add(i);
	    if(nodes[elem->n[3]])
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

#else

void Mesh::compute_neighbors()
{
    // Clear old neighbors...
    // now this builds all of the face neighbors as well...
    //
    int i;
    for(i=0;i<nodes.size();i++)
      if(nodes[i])
	    nodes[i]->elem = -1;

    Array1< Array1< int > > nodeElems;  // to build this stuff...

    nodeElems.setsize(nodes.size()); // allocate these arrays

    // Compute element info for nodes
    for(i=0;i<elems.size();i++){
	Element* elem=elems[i];
	if(elem){
	    if(nodes[elem->n[0]]) {
	      if (nodes[elem->n[0]]->elem == -1)
		nodes[elem->n[0]]->elem = i;
	      nodeElems[elem->n[0]].add(i);
	    }
	    if(nodes[elem->n[1]]) {
	      if (nodes[elem->n[1]]->elem == -1)
		nodes[elem->n[1]]->elem = i;
	      nodeElems[elem->n[1]].add(i);
	    }
	    if(nodes[elem->n[2]]) {
	      if (nodes[elem->n[2]]->elem == -1)
		nodes[elem->n[2]]->elem = i;
	      nodeElems[elem->n[2]].add(i);
	    }
	    if(nodes[elem->n[3]]) {
	      if (nodes[elem->n[3]]->elem == -1)
		nodes[elem->n[3]]->elem = i;
	      nodeElems[elem->n[3]].add(i);
	    }
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
    
    // now build all of the face neighbors...

    for(i=0;i<elems.size();i++) {
      if (elems[i]){
	for(int j=0;j<4;j++) {
	  if (elems[i]->faces[j] == -2) { // build it...
	    	int i1=elems[i]->n[(j+1)%4];
		int i2=elems[i]->n[(j+2)%4];
		int i3=elems[i]->n[(j+3)%4];

		elems[i]->faces[j]=unify(elems[i],nodeElems[i1],
					 nodeElems[i2],nodeElems[i3]);

		// could make opposite face point to this one
		// don't do it for now...  NEED CODE
	      }
	}
      }
    }

    // now try and do one of the nodes...

    int wichnode = nodes.size()/2;
    Array1<int> tarray;
    nodes[wichnode]->GetElems(tarray,this,wichnode);

    cerr << "Expected: " << nodeElems[wichnode].size() << "\n";
    cerr << "Got: " << tarray.size() << "\n";

    int print_em=1;
    if (tarray.size() == nodeElems[wichnode].size()) {
      print_em=0;
      for(int i=0;i<nodeElems[wichnode].size();i++) {
	int findme = nodeElems[wichnode][i];
	for(int j=0;j<tarray.size();j++) {
	  if (tarray[j] == findme) {
	    findme = -1;
	    j = tarray.size()+1;
	  }
	}
	if (findme != -1) { // they don't match
	  print_em=1;
	}
      }
    }

    if (print_em) {
      for (int q=0;q<nodeElems[wichnode].size();q++) {
	cerr << nodeElems[wichnode][q] << "  ";
	if (q < tarray.size())
	  cerr << tarray[q];
	cerr << endl;
      }
      if (tarray.size() > nodeElems[wichnode].size()) {
	for(;q<tarray.size();q++)
	  cerr << "   " << tarray[q] << endl;
      }
    }
}

#endif
// Barycentric coordinate computation from 
// Computer-Aided Analysis and Design of
// Electromagnetic Devices
// S. Ranajeeven & H. Hoole

int Mesh::inside(const Point& p, Element* elem)
{
    cerr << "inside called...\n";
#ifndef STORE_ELEMENT_BASIS
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
#else
    double s0=elem->a[0]+Dot(elem->g[0], p);
    if(s0<-1.e-6)
	return 0;
    double s1=elem->a[1]+Dot(elem->g[1], p);
    if(s0<-1.e-6)
	return 0;
    double s2=elem->a[2]+Dot(elem->g[2], p);
    if(s0<-1.e-6)
	return 0;
    double s3=elem->a[3]+Dot(elem->g[3], p);
    if(s0<-1.e-6)
	return 0;
#endif

    return 1;
}

void Mesh::get_interp(Element* elem, const Point& p,
		      double& s0, double& s1, double& s2, double& s3)
{
#ifndef STORE_ELEMENT_BASIS
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
    s0=iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());
    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    s1=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    s2=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    s3=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
#else
    s0=elem->a[0]+Dot(elem->g[0], p);
    s1=elem->a[1]+Dot(elem->g[1], p);
    s2=elem->a[2]+Dot(elem->g[2], p);
    s3=elem->a[3]+Dot(elem->g[3], p);
#endif
}

double Mesh::get_grad(Element* elem, const Point&,
		      Vector& g0, Vector& g1, Vector& g2, Vector& g3)
{
#ifndef STORE_ELEMENT_BASIS
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
    g0=Vector(b1*iV6, c1*iV6, d1*iV6);
    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    g1=Vector(b2*iV6, c2*iV6, d2*iV6);
    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    g2=Vector(b3*iV6, c3*iV6, d3*iV6);
    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    g3=Vector(b4*iV6, c4*iV6, d4*iV6);

    double vol=(1./iV6)/6.0;
    return(vol);
#else
    g0=elem->g[0];
    g1=elem->g[1];
    g2=elem->g[2];
    g3=elem->g[3];
    return elem->vol;
#endif
}

void print_element(Element* e, Mesh* mesh)
{
    cerr << "Element is composed of nodes: " << e->n[0] << ", " << e->n[1] << ", " << e->n[2] << ", " << e->n[3] << endl;
    for(int i=0;i<4;i++){
	int nn=e->n[i];
	Node* n=mesh->nodes[nn];
	cerr << nn << ": " << n->p << endl;
    }
}

#if 0

void dump_mesh(Mesh* mesh)
{
    ofstream out("mesh.dump");
    out << "Nodes:" << endl;
    int i;
    for(i=0;i<mesh->nodes.size();i++){
	Node* n=mesh->nodes[i];
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

#endif

int Mesh::locate(const Point& p, int& ix, double epsilon1, double epsilon2)
{
    // Start with the initial element
    int i=ix;
    if(i<0)
	i=0;
    // Find the next valid element in the list
    while(i<elems.size() && !elems[i])i++;
    if(i>=elems.size()){
	// If we get to the end, start over...
	i=0;
	while(i<ix && i<elems.size() && !elems[i])i++;
    }
    int count=0;
    int nelems=elems.size();
    while(count++<nelems){
	Element* elem=elems[i];
#ifndef STORE_ELEMENT_BASIS
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
	double s0=iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());

	double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
	double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
	double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
	double s1=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());

	double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
	double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
	double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
	double s2=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());

	double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
	double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
	double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
	double s3=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
#else
	double s0=elem->a[0]+Dot(elem->g[0], p);
	double s1=elem->a[1]+Dot(elem->g[1], p);
	double s2=elem->a[2]+Dot(elem->g[2], p);
	double s3=elem->a[3]+Dot(elem->g[3], p);
#endif
	int f=0;
	double min=s0;
	if(s1<min){
	    min=s1;
	    f=1;
	}
	if(s2<min){
	    min=s2;
	    f=2;
	}
	if(s3<min){
	    min=s3;
	    f=3;
	}
	if(min<-epsilon1){
	    int ni=elem->face(f);
#if 0
	    if(i==-1){
		cerr << "Boundary, min=" << min << endl;
		min=s0;
		f=0;
		if(s1<min && elem->face(1)!=-1){
		    min=s1;
		    f=1;
		}
		if(s2<min && elem->face(2)!=-1){
		    min=s2;
		    f=2;
		}
		if(s3<min && elem->face(3)!=-1){
		    min=s3;
		    f=3;
		}
		if(min<-1.e-6){
		    i=elem->face(f);
	            if(i != -1)
		         continue;
	        }
		return 0;
	    }
#endif
	    if(ni==-1){
		ix=i;
#if 0
		if(min < -epsilon2){
		    cerr << "Boundary, min=" << min << endl;
		}
#endif
		return min<-epsilon2?0:1;
	    }
	    i=ni;
	    continue;
	}
	ix=i;
	return 1;
    }
    return 0;
}

int Mesh::locate2(const Point& p, int& ix, double epsilon1)
{
    // Exhaustive search
    int nelems=elems.size();
    for(int i=0;i<nelems;i++){
	Element* elem=elems[i];
	if(!elem)
	  continue;
#ifndef STORE_ELEMENT_BASIS
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
	double s0=iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());

	double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
	double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
	double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
	double s1=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());

	double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
	double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
	double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
	double s2=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());

	double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
	double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
	double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
	double s3=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
#else
	double s0=elem->a[0]+Dot(elem->g[0], p);
	double s1=elem->a[1]+Dot(elem->g[1], p);
	double s2=elem->a[2]+Dot(elem->g[2], p);
	double s3=elem->a[3]+Dot(elem->g[3], p);
#endif
	double min=s0;
	if(s1<min){
	    min=s1;
	}
	if(s2<min){
	    min=s2;
	}
	if(s3<min){
	    min=s3;
	}
	if(min>-epsilon1){
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


void* OldNode::operator new(size_t)
{
    return OldNode_alloc.alloc();
}

void OldNode::operator delete(void* rp, size_t)
{
    OldNode_alloc.free(rp);
}

#ifdef STORE_THE_MESH

int Element::orient()
{
    double sgn=volume();
    if(sgn< 0.0){
	// Switch two of the edges so that the volume is positive
	int tmp=n[0];
	n[0]=n[1];
	n[1]=tmp;
	tmp=faces[0];
	faces[0]=faces[1];
	faces[1]=tmp;
	compute_basis();
	sgn=-sgn;
    }
    if(sgn < 1.e-9){
//	return 0; // Degenerate...
	cerr << "Warning - small element, volume=" << sgn << endl;
    }
    return 1;
}

#else
int Element::orient(Mesh *mesh)
{
    double sgn=volume(mesh);
    if(sgn< 0.0){
	// Switch two of the edges so that the volume is positive
	int tmp=n[0];
	n[0]=n[1];
	n[1]=tmp;
	tmp=faces[0];
	faces[0]=faces[1];
	faces[1]=tmp;
	compute_basis(mesh);
	sgn=-sgn;
    }
    if(sgn < 1.e-9){
//	return 0; // Degenerate...
	cerr << "Warning - small element, volume=" << sgn << endl;
    }
    return 1;
}

#endif

double Element::volume(
#ifndef STORE_THE_MESH
Mesh *mesh
#endif
)
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

void* Face::operator new(size_t)
{
    return Face_alloc.alloc();
}

void Face::operator delete(void* rp, size_t)
{
    Face_alloc.free(rp);
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
    hash=(n[0]*7+5)^(n[1]*5+3)^(n[2]*3+1);
}

Edge::Edge()
{
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
    cerr << "p=" << p << endl;
    cerr << "f=" << f << endl;
    cerr << "faces: " << e->faces[0] << " " << e->faces[1] << " " << e->faces[2] << " " << e->faces[3] << endl;
    cerr << "nfaces: " << ne->faces[0] << " " << ne->faces[1] << " " << ne->faces[2] << " " << ne->faces[3] << endl;
    return 0;
}

int Mesh::insert_delaunay(const Point& p, GeometryOPort* ogeom)
{
    int idx=nodes.size();
    nodes.add(new Node(p));
    return insert_delaunay(idx, ogeom);
}

MaterialHandle ptmatl(scinew Material(Color(0,0,0), Color(1,1,0), Color(.6,.6,.6), 10));
MaterialHandle inmatl(ptmatl);
MaterialHandle remmatl(scinew Material(Color(0,0,0), Color(1,0,0), Color(.6, .6, .6), 10));
MaterialHandle facematl(scinew Material(Color(0,0,0), Color(1,0,1), Color(.6, .6, .6), 10));
MaterialHandle circummatl(scinew Material(Color(0,0,0), Color(0, 1, 1), Color(.6, .6, .6), 10));
MaterialHandle lamatl(scinew Material(Color(0,0,0), Color(0, 0, 1), Color(.6, .6, .6), 10));


void Mesh::draw_element(Element* e, GeomGroup* group)
{
    Point p1(nodes[e->n[0]]->p);
    Point p2(nodes[e->n[1]]->p);
    Point p3(nodes[e->n[2]]->p);
    Point p4(nodes[e->n[3]]->p);
    GeomPolyline* poly=new GeomPolyline;
    poly->add(p1);
    poly->add(p2);
    poly->add(p3);
    poly->add(p4);
    poly->add(p1);
    poly->add(p3);
    poly->add(p2);
    poly->add(p4);
    group->add(poly);
}

void Mesh::draw_element(int in_element, GeomGroup* group)
{
    Element* e=elems[in_element];
    draw_element(e, group);
}

int Mesh::insert_delaunay(int node, GeometryOPort*)
{
    if(!have_all_neighbors)
	compute_face_neighbors();
    Point p(nodes[node]->p);

    // Start the element search at the last added element...
    int in_element=elems.size()-1;
    while(!elems[in_element] && in_element>0)
	in_element--;
    if(!locate(p, in_element)){
      if(!locate2(p, in_element)){
        cerr << "Error locating point: " << p << endl;
	return 0;
      }
    }


    Array1<int> to_remove;
    to_remove.add(in_element);

    // Find it's neighbors...
    current_generation++;
    elems[in_element]->generation=current_generation;

    FastHashTable<DFace> face_table;
    int i=0;
    while(i<to_remove.size()){
	// See if the neighbor should also be removed...
	int tr=to_remove[i];
	Element* e=elems[tr];
	if(!e){
	    cerr << "Removing a zero element!!!!!!!!!!!!!\n\n\n\n";
	}

	for(int j=0;j<4;j++){
	    // Add these faces to the list of exposed faces...
	    DFace* f=new DFace(e->n[(j+1)%4], e->n[(j+2)%4], e->n[(j+3)%4]);
	
	    // If the face is in the list, remove it.
	    // Otherwise, add it.
	    DFace* dummy;
	    if(face_table.lookup(f, dummy)){
		face_table.remove(f);
		delete f;
	    } else {
		f->face_idx=face_idx(tr, j);
		face_table.insert(f);
	    }

	    int neighbor=e->faces[j];
	    if(neighbor != -1){
		Element* ne=elems[neighbor];
		if(!ne){
		    cerr << endl;
		    cerr << "neighbor=" << neighbor << endl;
		    cerr << "is a neighbor of " << tr << endl;
		    cerr << "node=" << node << endl;
		    cerr << "WHAT!!!!!!!!!!\n";
		}
		if(ne->generation != current_generation){
		    Point cen;
		    double rad2;
		    double err;
#ifdef STORE_THE_MESH
		    ne->get_sphere2(cen, rad2, err);
#else
		    ne->get_sphere2(cen, rad2, err,this);
#endif
		    double ndist2=(p-cen).length2();
		    if(ndist2 < rad2-err){
			// This one must go...
			to_remove.add(neighbor);
		    }
		}
		ne->generation=current_generation;
	    }
	}
	i++;
    }
    for(i=0;i<to_remove.size();i++){
	int tr=to_remove[i];
//	cerr << "Removing: " << tr << endl;
	delete elems[tr];
	elems[tr]=0;
    }

    int start_new=elems.size();

    // Add the new elements from the faces...
    FastHashTableIter<DFace> fiter(&face_table);
    
    // Make a copy of the face table.  We use the faces in there
    // To compute the new neighborhood information
    FastHashTable<DFace> new_faces(face_table);

    for(fiter.first();fiter.ok();++fiter){
	DFace* f=fiter.get_key();
	Element* ne=new Element(this, node, f->n[0], f->n[1], f->n[2]);
	
	// If the new element is not degenerate, add it to the mix...
#ifdef STORE_THE_MESH
	if(ne->orient()){
#else
	if(ne->orient(this)){
#endif
	    int nen=elems.size();
	    for(int j=0;j<4;j++){
		// Add these faces to the list of exposed faces...
		DFace* f=new DFace(ne->n[(j+1)%4], ne->n[(j+2)%4], ne->n[(j+3)%4]);

		DFace* ef;
		if(new_faces.lookup(f, ef)){
		    // We have this face...
		    if(ef->face_idx==-1){
			ne->faces[j]=-1; // Boundary
		    } else {
			int which_face=ef->face_idx%4;
			int which_elem=ef->face_idx>>2;
			ne->faces[j]=which_elem;
			elems[which_elem]->faces[which_face]=nen;
		    }
		    new_faces.remove(f);
		    delete f;
		} else {
		    f->face_idx=(nen<<2)|j;
		    new_faces.insert(f);
		    ne->faces[j]=-3;
		}
	    }
	    elems.add(ne);
	} else {
	    cerr << "Degenerate element (node=" << node << ")\n";
#ifdef STORE_THE_MESH
	    cerr << "Volume=" << ne->volume() << endl;
#else
	    cerr << "Volume=" << ne->volume(this) << endl;
#endif
	    return 0;
	}
    }

    
    // Go through and look for small elements - try to get rid of
    // them by swapping an edge
#if 0
    for(int degen=start_new;degen<elems.size();degen++){
	Element* e=elems[degen];
	if(e->volume() < 1.e-10){
	    cerr << "Crap, element " << degen << " is degenerate...\n";
	    // Hmmm...
	    int d1a=0;
	    int d1b=-999;
	    double d1l=0;
	    Point p0=nodes[e->n[0]]->p;
	    for(int i=1;i<4;i++){
		double l=(nodes[e->n[i]]->p-p0).length2();
		if(l>d1l){
		    d1l=l;
		    d1b=i;
		}
	    }
	    int d2a=1;
	    while(d2a == d1b)d2a++;
	    int d2b=1;
	    while(d2b == d1b || d2b == d2a)d2b++;
	    double d2l=(nodes[e->n[d2a]]->p - nodes[e->n[d2b]]->p).length2();
	    cerr << "d1l=" << d1l << ", d2l=" << d2l << endl;
	    int a, b;
	    int c, d;
	    if(d1l < d2l){
		a=d1a;
		b=d1b;
		c=d2a;
		d=d2b;
	    } else {
		a=d2a;
		b=d2b;
		c=d1a;
		d=d1b;
	    }

	    cerr << "degen=" << degen << endl;
	    cerr << "a,b,c,d=" << a << " " << b << " " << c << " " << d << endl;
	    cerr << "nodes of degen are: " << e->n[a] << " " << e->n[b] << " " << e->n[c] << " " << e->n[d] << endl;

	    int e1=e->faces[a];
	    if(e1 != -1){
		Element* ee1=elems[e1];
		int j1;
		for(j1=0;j1<4;j1++){
		    if(ee1->faces[j1] == degen)
			break;
		}
		cerr << "ee1=" << ee1->n[0] << " " << ee1->n[1] << " " << ee1->n[2] << " " << ee1->n[3] << endl;
		ASSERT(j1<4);
		int n1=ee1->n[j1];

		int e2=e->faces[b];
		if(e2 != -1){
		    Element* ee2=elems[e2];
		    cerr << "ee2=" << ee2->n[0] << " " << ee2->n[1] << " " << ee2->n[2] << " " << ee2->n[3] << endl;
		    int j2;
		    for(j2=0;j2<4;j2++){
			if(ee2->faces[j2] == degen)
			    break;
		    }
		    ASSERT(j2<4);
		    int n2=ee2->n[j2];

		    if(n1 == n2){
			ee1->n[(j1+1)%4]=a;
			ee1->n[(j1+2)%4]=b;
			ee1->n[(j1+3)%4]=c;
			ee2->n[(j2+1)%4]=a;
			ee2->n[(j2+2)%4]=b;
			ee2->n[(j2+3)%4]=d;

			// Fix up face neighbors...
			ee2->faces[j2]=e1;
			ee1->faces[j1]=e2;
			
			// Let the face code figure out the other ones
			ee1->faces[(j1+1)%4]=-2;
			ee1->faces[(j1+2)%4]=-2;
			ee1->faces[(j1+3)%4]=-2;
			ee2->faces[(j2+1)%4]=-2;
			ee2->faces[(j2+2)%4]=-2;
			ee2->faces[(j2+3)%4]=-2;
			int ii;
			for(ii=0;ii<4;ii++){
			    int nn=elems[degen]->n[ii];
			    Node* n=nodes[nn];
			    for(int j=0;j<n->elems.size();j++){
				if(n->elems[j] == degen){
				    n->elems.remove(j);
				    break;
				}
			    }
			}
			for(ii=0;ii<4;ii++){
			    ee1->face(ii);
			    ee2->face(ii);
			}
			ee1->orient();
			ee2->orient();
			cerr << "after, ee1=" << ee1->n[0] << " " << ee1->n[1] << " " << ee1->n[2] << " " << ee1->n[3] << endl;
			cerr << "after, ee2=" << ee2->n[0] << " " << ee2->n[1] << " " << ee2->n[2] << " " << ee2->n[3] << endl;
			delete elems[degen];
			elems[degen]=0;
		    } else {
			cerr << "Sorry, can't fix this one..." << endl;
		    }
		} else {
		    cerr << "Can't fix because it is on the edge, case 2\n";
		}
	    } else {
		cerr << "Can't fix because it is on the edge, case 1\n";
	    }
	}
    }
#endif

#if 0
    {
	int nelems=elems.size();
	for(int i=0;i<nelems;i++){
	    Element* e=elems[i];
	    if(e){
		if(e->volume() < 1.e-5){
		    cerr << "Degenerate... (" << i << ") volume=" << e->volume() << endl;
		}
		Point cen;
		double rad2;
		double err;
		e->get_sphere2(cen, rad2, err);
		for(int ii=0;ii<node;ii++){
		    Point p(nodes[ii]->p);
		    double ndist2=(p-cen).length2();
		    if(ndist2 < rad2-1.e-6){
			cerr << "Invalid tesselation!\n";
			cerr << "ndist2=" << ndist2 << endl;
			cerr << "rad2=" << rad2 << endl;
			cerr << "err=" << err << endl;
		    }
		}
	    }
	}
    }
    if(node > 13800){
	for(int i=0;i<elems.size();i++){
	    Element* e=elems[i];
	    if(e){
		for(int j=0;j<4;j++){
		    int face=e->faces[j];
		    if(face != -1){
			if(elems[face] == 0){
			    cerr << "FACE MESSED UP!!!!!!!!!\n";
			    cerr << "face=" << face << endl;
			    cerr << "element=" << i << endl;
			    cerr << "node=" << node << endl;
			    cerr << endl;
			} else {
			    Element* ne=elems[face];
			    for(int jj=0;jj>4;jj++){
				if(ne->faces[jj] == i)
				    break;
			    }
			    if(jj==4){
				cerr << "Inconsistent neighbor information!!!!!!!\n";
				cerr << "element=" << i << endl;
				cerr << "neighbor=" << face << endl;
				cerr << endl;
			    }
			}

		    }
		}
	    }
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
    int i;
    for(i=0;i<nelems;i++){
	Element* e=elems[i];
	if(e){
	    map[i]=idx;
	    elems[idx++]=e;
	} else {
	    map[i]=-1234;
	}
    }
    elems.setsize(idx);
    int nnodes=nodes.size();
    for(i=0;i<nnodes;i++){
	Node*  n=nodes[i];
	if(n){
#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY
	    int ne=n->elems.size();
	    for(int j=0;j<ne;j++){
		int elem=n->elems[j];
		int new_elem=map[elem];
		if(new_elem == -1234)
		    cerr << "Warning: pointing to old element: " << elem << endl;
		n->elems[j]=new_elem;
	    }
#else
	    if (n->elem != -1) {
	      if (map[n->elem] == -1234) {
		cerr << "Warning: pointing to old element: " << n->elem << endl;
	      }
	      n->elem = map[n->elem];
	    }
#endif	
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
    int i;
    for(i=0;i<nnodes;i++){
	Node* n=nodes[i];
	if(n){
	    map[i]=idx;
	    nodes[idx++]=n;
	} else {
	    map[i]=-1234;
	}
    }
    nodes.setsize(idx);
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
	Node* n=nodes[node];
#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY
	for(int i=0;i<n->elems.size();i++){
	    if(elems[n->elems[i]]){
		delete elems[n->elems[i]];
		elems[n->elems[i]]=0;
	    }
	}
#else
	Array1< int > elemsID;
	n->GetElems(elemsID, this,node);
	for(int i=0;i<elemsID.size();i++){
	    if(elems[elemsID[i]]){
		delete elems[elemsID[i]];
		elems[elemsID[i]]=0;
	    }
	}
#endif
	nodes[node]=0;
    } else {
	NOT_FINISHED("Mesh::remove_delaunay");
    }
}

void Mesh::compute_face_neighbors()
{
// compute all of this stuff earlier now...
#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY
    // This could probably be done much faster...
    for(int i=0;i<elems.size();i++){
	if(elems[i]){
	    for(int j=0;j<4;j++){
		elems[i]->face(j);
	    }
	}
    }
#endif
    have_all_neighbors=1;
}

static void heapify(int* data, int n, int i)
{
    int l=2*i+1;
    int r=l+1;
    int largest=i;
    if(l<n && data[l] > data[i])
	largest=l;
    if(r<n && data[r] > data[largest])
	largest=r;
    if(largest != i){
	int tmp=data[i];
	data[i]=data[largest];
	data[largest]=tmp;
	heapify(data, n, largest);
    }
}

#ifdef USE_OLD_HUGE_WASTEFUL_ELEM_ARRAY

void Mesh::add_node_neighbors(int node, Array1<int>& idx)
{
    Node* n=nodes[node];
    int ne=n->elems.size();
    Array1<int> neighbor_nodes(4*ne+1);
    int nodesi=0;
    // Gather all of the nodes
    int i;
    for(i=0;i<ne;i++){
	int ei=n->elems[i];
	Element* e=elems[ei];
	for(int j=0;j<4;j++){
	    int n=e->n[j];
	    if(!nodes[n]->bc || !apBC)
		neighbor_nodes[nodesi++]=n;
	}
    }
    // Sort it...
    // Build the heap...
    int* data=&neighbor_nodes[0];
    for(i=nodesi/2-1;i >= 0;i--){
	heapify(data, nodesi, i);
    }
    // Sort
    for(i=nodesi-1;i>0;i--){
	// Exchange 1 and i
	int tmp=data[i];
	data[i]=data[0];
	data[0]=tmp;
	heapify(data, i, 0);
    }


    // Find the unique set...
    for(i=0;i<nodesi;i++){
	if(i==0 || neighbor_nodes[i] != neighbor_nodes[i-1])
	    idx.add(neighbor_nodes[i]);
    }
}

#else

void Mesh::add_node_neighbors(int node, Array1<int>& idx, int apBC)
{
    Node* n=nodes[node];
    Array1<int> Nelems;
    n->GetElems(Nelems,this,node); // might want a GetNodes that does just this

    int ne=Nelems.size();
    Array1<int> neighbor_nodes(4*ne+1);
    int nodesi=0;
    // Gather all of the nodes
    int i;
    for(i=0;i<ne;i++){
	int ei=Nelems[i];
	Element* e=elems[ei];
	for(int j=0;j<4;j++){
	    int n=e->n[j];
	    if(!nodes[n]->bc)
		neighbor_nodes[nodesi++]=n;
	}
    }
    // Sort it...
    // Build the heap...
    int* data=&neighbor_nodes[0];
    for(i=nodesi/2-1;i >= 0;i--){
	heapify(data, nodesi, i);
    }
    // Sort
    for(i=nodesi-1;i>0;i--){
	// Exchange 1 and i
	int tmp=data[i];
	data[i]=data[0];
	data[0]=tmp;
	heapify(data, i, 0);
    }


    // Find the unique set...
    for(i=0;i<nodesi;i++){
	if(i==0 || neighbor_nodes[i] != neighbor_nodes[i-1])
	    idx.add(neighbor_nodes[i]);
    }
}

#endif

// this is a simple structure which has a element index and the
// element it was entered from...

struct ElementFace {
  int elem; // element to try -> -1 if this slot is free...
  int face; // face -> which is the element you came from...
};

// this function builds the element array by walking the connectivity
// structure, the basic idea is to start from a element that is
// connected, push it on the array and push the 3 elements opposite
// faces that are not this node on a Queue - if they are not already
// in the array and are also not already in the queue
//
// Then you just pop the queue and repeat the proccess until the queue
// is empty

inline void StickOnQ(Array1< ElementFace > &myQ, int node, int from,
		     Mesh *mesh, int &nQ)
{
  Element *elm = mesh->elems[from];

  for(int j=0;j<4;j++) {
    if ((elm->n[j] != node)         // can't be opposite this node
	&& (elm->faces[j] != -1)) { // we can try this one...
      int qid=-1;
      int target = elm->faces[j]; // element we are trying to fill
      for(int k=0;k<myQ.size();k++) {
	if (myQ[k].elem != -1) {
	  if (myQ[k].elem == target) { // already there
	    qid = k; // just write over what was already there...
	    nQ--; // since this is going to get pushed on again
	    k = myQ.size()+1;
	  }
	} else { // see if we need to use it as a empty slot
	  if (qid == -1)
	    qid = k;	
	}
      }

      if (qid == -1) {
	qid = myQ.size();
	myQ.grow(1); // bump it up
      }

      myQ[qid].elem = target;
      myQ[qid].face = from;
      nQ++;
    }
  }
}

inline void CStickOnQ(Array1<int> &inQ, // can't insert something that is there
		      Array1< ElementFace > &myQ, int node, int from,
		      int ffrom, //
		      Mesh *mesh, int &nQ)
{
  Element *elm = mesh->elems[from];

  for(int j=0;j<4;j++) {
    if ((elm->n[j] != node)         // can't be opposite this node
	&& (elm->faces[j] != ffrom) // don't bother with guy you came from
	&& (elm->faces[j] != -1)) { // we can try this one...
      int qid=-1;
      int target = elm->faces[j]; // element we are trying to fill

      // first make sure it isn't in the array handed in
      int can_go=1;
      for(int i=0;i<inQ.size();i++) {
	if (inQ[i] == target) {
	  can_go = 0;
	  break; // out of foor loop
	}
      }

      if (can_go) { // this guy isn't in the element array

	for(int k=0;k<myQ.size();k++) {
	  if (myQ[k].elem != -1) {
	    if (myQ[k].elem == target) { // already there
	      qid = k; // just write over what was already there...
	      nQ--; // since this is going to get pushed on again
	      k = myQ.size()+1;
	    }
	  } else { // see if we need to use it as a empty slot
	    if (qid == -1)
	      qid = k;	
	  }
	}

	if (qid == -1) {
	  qid = myQ.size();
	  myQ.grow(1); // bump it up
	}

	myQ[qid].elem = target;
	myQ[qid].face = from;
	nQ++;
      }
    }
  }
}

void Node::GetElems(Array1<int>& elems, Mesh* mesh, int me)
{
  elems.remove_all();

  Array1< ElementFace > myQ(10,5,-1);

  myQ.remove_all(); // make it look like it is empty...

  int nQ=0; // number on the Q...

  elems.add(elem); // this is the starting point...

  StickOnQ(myQ,me,elem,mesh,nQ); // sticks these on theQ
  
  while(nQ) { // while you can still do this...
    int test=-1;

    for(int i=0;i<myQ.size();i++) {
      if (myQ[i].elem != -1) { // got one!
	test = i;
	break; // out of the for loop
      }
    }

    if (test == -1) {
      cerr << "Error, nQ is set so something is wacked...\n";
    }

    ElementFace pop = myQ[test];

    myQ[test].elem = -1; // clear this guy out...
    nQ--; // decrement the number of elements on the Q

    elems.add(pop.elem); // stick this guy and enque neighbors

    // now conditionaly try and stick these guys on the Q...
    CStickOnQ(elems,myQ,me,pop.elem,pop.face,mesh,nQ);
  }
}

// this function is *not* threadsafe - use with caution!!!

void Node::GetElemsNT(Array1<int> &build, Mesh *mesh, int me, 
		      int check, int from,
		      int gen)
{
  if (check == -1) {
    check = elem;
  }
  Element *test = mesh->elems[check];

  if (!test) {
    cerr << check << " " << me << " Woah - check was bad!\n";
    return;
  }

  test->generation = gen; // set this one up...

  build.add(check); // add this guy...

  for(int i=0;i<4;i++) {
    if ((test->faces[i] != from) &&
	(test->n[i] != me) &&
	(test->faces[i] != -1)) { // canidate element...
      if (mesh->elems[test->faces[i]]->generation != gen) {
	GetElemsNT(build,mesh,me,test->faces[i],check,gen);
      }
    }
  }
}

// thread safe version of above...

void Node::GetElemsNT(Array1<int> &build, Mesh *mesh, int me, 
		      int check, int from,
		      int gen,
		      Array1< unsigned int > &elemG)
{
  if (check == -1) {
    check = elem;
  }
  Element *test = mesh->elems[check];
  if (!test) {
    cerr << check << " " << me << " Woah - check was bad!\n";
    return;
  }
  elemG[check] = gen; // set this one up...

  build.add(check); // add this guy...

  for(int i=0;i<4;i++) {
    if ((test->faces[i] != from) &&
	(test->n[i] != me) &&
	(test->faces[i] != -1)) { // canidate element...
      if (elemG[test->faces[i]] != gen) {
	GetElemsNT(build,mesh,me,test->faces[i],check,gen,elemG);
      }
    }
  }
}

// none thread safe version

void Node::GetNodesNT(Array1<int> &build, Mesh* mesh, int me,
		      int check, int from, int gen,
		      Array1<unsigned int> &NgenA) // nodes have g#
{
  if (check == -1) { // starting...
    Element *test = mesh->elems[elem];
    test->generation = gen;

    for(int i=0;i<4;i++) {
      if (test->n[i] != me) { // add this node
	build.add(test->n[i]);
	NgenA[test->n[i]] = gen;
	if ((test->faces[i] != -1) &&
	    (mesh->elems[test->faces[i]]->generation != gen)) { // recu
	  GetNodesNT(build,mesh,me,test->faces[i],elem,gen,NgenA);
	}
      }
    }
  } else { // normal recursion
    Element *test = mesh->elems[check];
    test->generation = gen;

    for(int i=0;i<4;i++) { // only add the node which leads into you
      if (test->faces[i] == from) {
	if (NgenA[test->n[i]] != gen)
	  build.add(test->n[i]); // stuck on the Q
      } else {
	if ((test->n[i] != me) &&
	    (test->faces[i] != -1) &&
	    (mesh->elems[test->faces[i]]->generation != gen)) { // recur
	  GetNodesNT(build,mesh,me,test->faces[i],check,gen,NgenA);
	}
      }
    }
  }
}

// thread safe version
// NgenA and genA must be per thread - along with build of course...

void Node::GetNodesNT(Array1<int> &build, Mesh* mesh, int me,
		      int check, int from, int gen,
		      Array1<unsigned int> &NgenA,
		      Array1<unsigned int> &genA) 
{
  if (check == -1) { // starting...
    Element *test = mesh->elems[elem];
    genA[elem] = gen;

    for(int i=0;i<4;i++) {
      if (test->n[i] != me) { // add this node
	build.add(test->n[i]);
	NgenA[test->n[i]] = gen;
	if ((test->faces[i] != -1) &&
	    (genA[test->faces[i]] != gen)) { // recu
	  GetNodesNT(build,mesh,me,test->faces[i],elem,gen,NgenA,genA);
	}
      }
    }
  } else { // normal recursion
    Element *test = mesh->elems[check];
    genA[check] = gen;

    for(int i=0;i<4;i++) { // only add the node which leads into you
      if (test->faces[i] == from) {
	if (NgenA[test->n[i]] != gen)
	  build.add(test->n[i]); // stuck on the Q
      } else {
	if ((test->n[i] != me) &&
	    (test->faces[i] != -1) &&
	    (genA[test->faces[i]] != gen)) { // recur
	  GetNodesNT(build,mesh,me,test->faces[i],check,gen,NgenA,genA);
	}
      }
    }
  }
}

// edges are much trickier - all of the edges defining the 1 element
// neighborhood can be created by using the above node list along
// with the boundary hull - these edges are special because they
// can only be shared by 2 elements, the functions below can get
// this outer hull of edges

void Node::GetEdgesNT(Array1<int> &builds, Array1<int> &buildd,
		      Mesh *mesh, int me, 
		      int check, // -1 for start
		      int gen)
{
  // define variable here so doesn't allocate twice??
  Element *test;
  int nds[3] = {-1,-1,-1};
  int num_got=0;
  if (check == -1) {
    check = elem;
    test = mesh->elems[elem];
    test->generation = gen;

    // ok find which node you are
    for(int j=0;j<4;j++) {
      if (test->n[j] != me) {
	nds[num_got++] = test->n[j]; // wierd recursion
	if ((test->faces[j] != -1) && 
	    (mesh->elems[test->faces[j]]->generation != gen)) {
	  GetEdgesNT(builds,buildd,mesh,me,test->faces[j],gen);
	}
      }
    }
    builds.add(nds[0]); // 3 edges for start 01,02,12
    builds.add(nds[0]); // even though they get added on in the end...
    builds.add(nds[1]);

    buildd.add(nds[1]);
    buildd.add(nds[2]);
    buildd.add(nds[2]);
  } else { // ok
    test = mesh->elems[check];
    test->generation = gen;

    // if the elem opposite a node hase been set,
    // the edge not containing it has been defined

    int opp_ok[3] = {0,0,0};
    
    
    for(int j=0;j<4;j++) {
      if (test->n[j] != me) {
	nds[num_got] = j; 
	if ((test->faces[j] != -1) && 
	    (mesh->elems[test->faces[j]]->generation != gen)) {
	  opp_ok[num_got] = 1; // you can set this edge and recurse
	}
	num_got++;
      }
    }
    
    for(j=0;j<3;j++) {
      if (opp_ok[j] || (test->faces[nds[j]] == -1)) {
	builds.add(test->n[nds[(j+1)%3]]);
	buildd.add(test->n[nds[(j+2)%3]]);
	if (test->faces[nds[j]] != -1)
	  GetEdgesNT(builds,buildd,mesh,me,test->faces[nds[j]],gen);
      }
    }
  }
}

// thread save version of above

void Node::GetEdgesNT(Array1<int> &builds, Array1<int> &buildd,
		      Mesh *mesh, int me, 
		      int check, // -1 for start
		      int gen,
		      Array1<unsigned int> &genA)
{
  // define variable here so doesn't allocate twice??
  Element *test;
  int nds[3] = {-1,-1,-1};
  int num_got=0;
  if (check == -1) {
    check = elem;
    test = mesh->elems[elem];
    genA[check] = gen;

    // ok find which node you are
    for(int j=0;j<4;j++) {
      if (test->n[j] != me) {
	nds[num_got++] = test->n[j]; // wierd recursion
	if ((test->faces[j] != -1) && 
	    (genA[test->faces[j]] != gen)) {
	  GetEdgesNT(builds,buildd,mesh,me,test->faces[j],gen,genA);
	}
      }
    }
    builds.add(nds[0]); // 3 edges for start 01,02,12
    builds.add(nds[0]); // even though they get added on in the end...
    builds.add(nds[1]);

    buildd.add(nds[1]);
    buildd.add(nds[2]);
    buildd.add(nds[2]);
  } else { // ok
    test = mesh->elems[check];
    genA[check] = gen;

    // if the elem opposite a node hase been set,
    // the edge not containing it has been defined

    int opp_ok[3] = {0,0,0};
    
    
    for(int j=0;j<4;j++) {
      if (test->n[j] != me) {
	nds[num_got] = j; 
	if ((test->faces[j] != -1) && 
	    (genA[test->faces[j]] != gen)) {
	  opp_ok[num_got] = 1; // you can set this edge and recurse
	}
	num_got++;
      }
    }
    
    for(j=0;j<3;j++) {
      if (opp_ok[j] || (test->faces[nds[j]] == -1)) {
	builds.add(test->n[nds[(j+1)%3]]);
	buildd.add(test->n[nds[(j+2)%3]]);
	if (test->faces[nds[j]] != -1)
	  GetEdgesNT(builds,buildd,mesh,me,test->faces[nds[j]],gen);
      }
    }
  }
}

DirichletBC::DirichletBC(const SurfaceHandle& fromsurf, double value)
: fromsurf(fromsurf), value(value)
{
}

void Mesh::get_boundary_lines(Array1<Point>& lines)
{
    NOT_FINISHED("Mesh::get_boundary_lines");
}

