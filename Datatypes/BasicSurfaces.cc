
/*
 *  BasicSurfaces.h: Cylinders and stuff
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/BasicSurfaces.h>
#include <Classlib/NotFinished.h>
#include <Geom/Cylinder.h>
#include <Geom/Group.h>
#include <Geom/Pt.h>
#include <Geom/Sphere.h>
#include <Geom/Triangles.h>
#include <Malloc/Allocator.h>
#include <Math/Trig.h>
#include <Math/TrigTable.h>
#include <TCL/TCL.h>
#include <stdio.h>
#include <strstream.h>

static Persistent* make_CylinderSurface()
{
    return scinew CylinderSurface(Point(0,0,0), Point(0,0,1),1,10,10,10);
}

PersistentTypeID CylinderSurface::type_id("CylinderSurface", "Surface",
					  make_CylinderSurface);

CylinderSurface::CylinderSurface(const Point& p1, const Point& p2,
				 double radius, int nu, int nv, int ndiscu)
: Surface(RepOther, 1),
  p1(p1), p2(p2), radius(radius), nu(nu), nv(nv), ndiscu(ndiscu)
{
    axis=p2-p1;
    rad2=radius*radius;
    if(axis.length2() > 1.e-6) {
	height=axis.normalize();
    } else {
	// Degenerate cylinder
	height=0;
	axis=Vector(0,0,1);
    }
    axis.find_orthogonal(u, v);
}

CylinderSurface::~CylinderSurface()
{
}

CylinderSurface::CylinderSurface(const CylinderSurface& copy)
: Surface(copy), p1(copy.p1), p2(copy.p2), radius(copy.radius),
  nu(copy.nu), nv(copy.nv), ndiscu(copy.ndiscu)
{
}

Surface* CylinderSurface::clone()
{
    return scinew CylinderSurface(*this);
}

int CylinderSurface::inside(const Point& p)
{
    double l=Dot(p-p1, axis);
    if(l<0)
	return 0;
    if(l>height)
	return 0;
    Point c(p1+axis*l);
    double dl2=(p-c).length2();
    if(dl2 > rad2)
	return 0;
    return 1;
}

void CylinderSurface::add_node(Array1<NodeHandle>& nodes,
			       char* id, const Point& p, double r, double rn,
			       double theta, double h, double hn)
{
    Node* node=new Node(p);
    if(boundary_type == DirichletExpression){
	char str [200];
	ostrstream s(str, 200);
	s << id << " " << p.x() << " " << p.y() << " " << p.z()
	  << " " << r << " " << rn << " " << theta << " " << h
	  << " " << hn << '\0';
	clString retval;
	int err=TCL::eval(str, retval);
	if(err){
	    cerr << "Error evaluating boundary value" << endl;
	    boundary_type = BdryNone;
	    return;
	}
	double value;
	if(!retval.get_double(value)){
	    cerr << "Bad result from boundary value" << endl;
	    boundary_type = BdryNone;
	    return;
	}
	node->bc=scinew DirichletBC(this, value);
    }
    nodes.add(node);
}

void CylinderSurface::set_surfnodes(const Array1<NodeHandle>& nodes) {
    NOT_FINISHED("CylinderSurface::set_surfnodes");
}

void CylinderSurface::get_surfnodes(Array1<NodeHandle>& nodes)
{
    char id[100];
    if(boundary_type == DirichletExpression){
	// Format this string - we will use it later...
	char proc_string[1000];
	sprintf(id, "CylinderSurface%p%p", this, &nodes);
	ostrstream proc(proc_string, 1000);
	proc << "proc " << id << " {x y z r rn theta h hn} { expr "
	     << boundary_expr << "}" << '\0';
	TCL::execute(proc_string);
    }
    add_node(nodes, id, p1, 0, 0, 0, 0, 0);
    if(boundary_type == BdryNone)
	return;
    SinCosTable tab(nv+1, 0, 2*Pi, radius);
    int i;
    for(i=1;i<ndiscu-1;i++){
	double r=double(i)/double(ndiscu-1);
	for(int j=0;j<nv;j++){
	    double theta=double(j)/double(nv)*2*Pi;
	    Point p(p1+(u*tab.sin(j)+v*tab.cos(j))*r);
	    add_node(nodes, id, p, r*radius, r, theta, 0, 0);
	    if(boundary_type == BdryNone)
		return;
	}
    }
    for(i=0;i<=nu;i++){
	double h=double(i)/double(nu);
	double hh=h*height;
	for(int j=0;j<nv;j++){
	    double theta=double(j)/double(nv)*2*Pi;
	    Point p(p1+u*tab.sin(j)+v*tab.cos(j)+axis*hh);
	    add_node(nodes, id, p, radius, 1, theta, hh, h);
	    if(boundary_type == BdryNone)
		return;
	}
    }
    for(i=ndiscu-2;i>=1;i--){
	double r=double(i)/double(ndiscu-1);
	for(int j=0;j<nv;j++){
	    double theta=double(j)/double(nv)*2*Pi;
	    Point p(p2+(u*tab.sin(j)+v*tab.cos(j))*r);
	    add_node(nodes, id, p, r*radius, r, theta, height, 1);
	    if(boundary_type == BdryNone)
		return;
	}
    }
    add_node(nodes, id, p2, 0, 0, 0, height, 1);
    if(boundary_type == BdryNone)
	return;
    TCL::execute(clString("rename ")+id+" \"\"");
}

#define CYLINDERSURFACE_VERSION 1

void CylinderSurface::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("CylinderSurface", CYLINDERSURFACE_VERSION);
    Surface::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, radius);
    Pio(stream, nu);
    Pio(stream, nv);
    Pio(stream, ndiscu);
    stream.end_class();
}

void CylinderSurface::construct_grid(int, int, int, const Point&, double)
{
    NOT_FINISHED("CylinderSurface::construct_grid");
}

void CylinderSurface::construct_grid()
{
    NOT_FINISHED("CylinderSurface::construct_grid");
}

GeomObj* CylinderSurface::get_obj(const ColorMapHandle& cmap)
{
    if(boundary_type == BdryNone)
	return scinew GeomCappedCylinder(p1, p2, radius);

    Array1<NodeHandle> nodes;
    get_surfnodes(nodes);

    // This is here twice, since get_surfnodes may reduce back to 
    // no BC's if there is an error...
    if(boundary_type == BdryNone)
	return scinew GeomCappedCylinder(p1, p2, radius);

    GeomGroup* group=scinew GeomGroup;
    GeomTrianglesPC* tris=scinew GeomTrianglesPC;
    group->add(tris);

    int s=1;
    int i;
    double v1=nodes[0]->bc->value;
    Color cc1(cmap->lookup(v1)->diffuse);
    Point& pp1(nodes[0]->p);
    int j;
    for(j=0;j<nv;j++){
	int i2=1+(j%nv);
	int i3=1+((j+1)%nv);
	double v2=nodes[i2]->bc->value;
	double v3=nodes[i3]->bc->value;
	Color cc2(cmap->lookup(v2)->diffuse);
	Color cc3(cmap->lookup(v3)->diffuse);
	Point& pp2(nodes[i2]->p);
	Point& pp3(nodes[i3]->p);
	tris->add(pp1, cc1, pp2, cc2, pp3, cc3);
    }
    for(i=0;i<2*(ndiscu-2)+nu;i++){
	for(int j=0;j<nv;j++){
	    int i1=s+(j%nv);
	    int i2=s+((j+1)%nv);
	    int i3=i1+nv;
	    int i4=i2+nv;
	    double v1=nodes[i1]->bc->value;
	    double v2=nodes[i2]->bc->value;
	    double v3=nodes[i3]->bc->value;
	    double v4=nodes[i4]->bc->value;
	    Color cc1(cmap->lookup(v1)->diffuse);
	    Color cc2(cmap->lookup(v2)->diffuse);
	    Color cc3(cmap->lookup(v3)->diffuse);
	    Color cc4(cmap->lookup(v4)->diffuse);
	    Point& pp1(nodes[i1]->p);
	    Point& pp2(nodes[i2]->p);
	    Point& pp3(nodes[i3]->p);
	    Point& pp4(nodes[i4]->p);
	    tris->add(pp1, cc1, pp2, cc2, pp3, cc3);
	    tris->add(pp2, cc2, pp3, cc3, pp4, cc4);
	}
	s+=nv;
    }
    int last=nodes.size()-1;
    double v3=nodes[last]->bc->value;
    Color cc3(cmap->lookup(v3)->diffuse);
    Point& pp3(nodes[last]->p);
    for(j=0;j<nv;j++){
	int i1=s+(j%nv);
	int i2=s+((j+1)%nv);
	double v1=nodes[i1]->bc->value;
	double v2=nodes[i2]->bc->value;
	Color cc1(cmap->lookup(v1)->diffuse);
	Color cc2(cmap->lookup(v2)->diffuse);
	Point& pp1(nodes[i1]->p);
	Point& pp2(nodes[i2]->p);
	tris->add(pp1, cc1, pp2, cc2, pp3, cc3);
    }
    return group;
}

static Persistent* make_SphereSurface()
{
    return scinew SphereSurface(Point(0,0,0),1,Vector(0,0,1),10,10);
}

PersistentTypeID SphereSurface::type_id("SphereSurface", "Surface",
					make_SphereSurface);

SphereSurface::SphereSurface(const Point& cen, double radius,
			     const Vector& p,
			     int nu, int nv)
: Surface(RepOther, 1),
  cen(cen), radius(radius), pole(p), nu(nu), nv(nv)
{
    rad2=radius*radius;
    if(pole.length2() > 1.e-6) {
	pole.normalize();
    } else {
	// Degenerate sphere
	pole=Vector(0,0,1);
    }
    pole.find_orthogonal(u, v);
}

SphereSurface::~SphereSurface()
{
}

SphereSurface::SphereSurface(const SphereSurface& copy)
: Surface(copy), cen(copy.cen), radius(copy.radius),
  pole(copy.pole),
  nu(copy.nu), nv(copy.nv)
{
}

Surface* SphereSurface::clone()
{
    return scinew SphereSurface(*this);
}

int SphereSurface::inside(const Point& p)
{
    double dl2=(p-cen).length2();
    if(dl2 > rad2)
	return 0;
    return 1;
}

void SphereSurface::add_node(Array1<NodeHandle>& nodes,
			     char* id, const Point& p, double r,
			     double theta, double phi)
{
    Node* node=new Node(p);
    if(boundary_type == DirichletExpression){
	char str [200];
	ostrstream s(str, 200);
	s << id << " " << p.x() << " " << p.y() << " " << p.z()
	  << " " << r << " " << theta << " " << phi << '\0';
	clString retval;
	int err=TCL::eval(str, retval);
	if(err){
	    cerr << "Error evaluating boundary value" << endl;
	    boundary_type = BdryNone;
	    return;
	}
	double value;
	if(!retval.get_double(value)){
	    cerr << "Bad result from boundary value" << endl;
	    boundary_type = BdryNone;
	    return;
	}
	node->bc=scinew DirichletBC(this, value);
    }
    nodes.add(node);
}

void SphereSurface::set_surfnodes(const Array1<NodeHandle>& nodes) {
    NOT_FINISHED("SphereSurface::set_surfnodes");
}

void SphereSurface::get_surfnodes(Array1<NodeHandle>& nodes)
{
    char id[100];
    if(boundary_type == DirichletExpression){
	// Format this string - we will use it later...
	char proc_string[1000];
	sprintf(id, "CylinderSurface%p%p", this, &nodes);
	ostrstream proc(proc_string, 1000);
	proc << "proc " << id << " {x y z r theta phi} { expr "
	     << boundary_expr << "}" << '\0';
	TCL::execute(proc_string);
    }
    add_node(nodes, id, cen-pole*radius, radius, 0, -Pi/2);
    if(boundary_type == BdryNone)
	return;
    SinCosTable phitab(nu, -Pi/2, Pi/2);
    SinCosTable tab(nv+1, 0, 2*Pi, radius);
    int i;
    for(i=1;i<nu-1;i++){
	double phi=(double(i)/double(nu)-0.5)*Pi;
	for(int j=0;j<nv;j++){
	    double theta=double(j)/double(nv)*2*Pi;
	    Point p(cen+(u*tab.sin(j)+v*tab.cos(j))*phitab.cos(i)+pole*phitab.sin(i)*radius);
	    add_node(nodes, id, p, radius, theta, phi);
	    if(boundary_type == BdryNone)
		return;
	}
    }
    add_node(nodes, id, cen+pole*radius, radius, 0, Pi/2);
    if(boundary_type == BdryNone)
	return;
    TCL::execute(clString("rename ")+id+" \"\"");
}

#define SPHERESURFACE_VERSION 1

void SphereSurface::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("SphereSurface", SPHERESURFACE_VERSION);
    Surface::io(stream);
    Pio(stream, cen);
    Pio(stream, radius);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

void SphereSurface::construct_grid(int, int, int, const Point&, double)
{
    NOT_FINISHED("SphereSurface::construct_grid");
}

void SphereSurface::construct_grid()
{
    NOT_FINISHED("SphereSurface::construct_grid");
}

GeomObj* SphereSurface::get_obj(const ColorMapHandle& cmap)
{
    if(boundary_type == BdryNone)
	return scinew GeomSphere(cen, radius);

    Array1<NodeHandle> nodes;
    get_surfnodes(nodes);

    // This is here twice, since get_surfnodes may reduce back to 
    // no BC's if there is an error...
    if(boundary_type == BdryNone)
	return scinew GeomSphere(cen, radius);

    GeomGroup* group=scinew GeomGroup;
    GeomTrianglesPC* tris=scinew GeomTrianglesPC;
    group->add(tris);

    int s=1;
    int i;
    double v1=nodes[0]->bc->value;
    Color cc1(cmap->lookup(v1)->diffuse);
    Point& pp1(nodes[0]->p);
    int j;
    for(j=0;j<nv;j++){
	int i2=1+(j%nv);
	int i3=1+((j+1)%nv);
	double v2=nodes[i2]->bc->value;
	double v3=nodes[i3]->bc->value;
	Color cc2(cmap->lookup(v2)->diffuse);
	Color cc3(cmap->lookup(v3)->diffuse);
	Point& pp2(nodes[i2]->p);
	Point& pp3(nodes[i3]->p);
	tris->add(pp1, cc1, pp2, cc2, pp3, cc3);
    }
    for(i=0;i<nu-3;i++){
	for(int j=0;j<nv;j++){
	    int i1=s+(j%nv);
	    int i2=s+((j+1)%nv);
	    int i3=i1+nv;
	    int i4=i2+nv;
	    double v1=nodes[i1]->bc->value;
	    double v2=nodes[i2]->bc->value;
	    double v3=nodes[i3]->bc->value;
	    double v4=nodes[i4]->bc->value;
	    Color cc1(cmap->lookup(v1)->diffuse);
	    Color cc2(cmap->lookup(v2)->diffuse);
	    Color cc3(cmap->lookup(v3)->diffuse);
	    Color cc4(cmap->lookup(v4)->diffuse);
	    Point& pp1(nodes[i1]->p);
	    Point& pp2(nodes[i2]->p);
	    Point& pp3(nodes[i3]->p);
	    Point& pp4(nodes[i4]->p);
	    tris->add(pp1, cc1, pp2, cc2, pp3, cc3);
	    tris->add(pp2, cc2, pp3, cc3, pp4, cc4);
	}
	s+=nv;
    }
    int last=nodes.size()-1;
    double v3=nodes[last]->bc->value;
    Color cc3(cmap->lookup(v3)->diffuse);
    Point& pp3(nodes[last]->p);
    for(j=0;j<nv;j++){
	int i1=s+(j%nv);
	int i2=s+((j+1)%nv);
	double v1=nodes[i1]->bc->value;
	double v2=nodes[i2]->bc->value;
	Color cc1(cmap->lookup(v1)->diffuse);
	Color cc2(cmap->lookup(v2)->diffuse);
	Point& pp1(nodes[i1]->p);
	Point& pp2(nodes[i2]->p);
	tris->add(pp1, cc1, pp2, cc2, pp3, cc3);
    }
    return group;
}

static Persistent* make_PointSurface()
{
    return scinew PointSurface(Point(0,0,0));
}

PersistentTypeID PointSurface::type_id("PointSurface", "Surface",
				       make_PointSurface);

PointSurface::PointSurface(const Point& pos)
: Surface(RepOther, 0), pos(pos)
{
}

PointSurface::~PointSurface()
{
}

PointSurface::PointSurface(const PointSurface& copy)
: Surface(copy), pos(copy.pos)
{
}

Surface* PointSurface::clone()
{
    return scinew PointSurface(*this);
}

int PointSurface::inside(const Point&)
{
    return 0;
}

void PointSurface::add_node(Array1<NodeHandle>& nodes,
			    char* id, const Point& p)
{
    Node* node=new Node(p);
    if(boundary_type == DirichletExpression){
	char str [200];
	ostrstream s(str, 200);
	s << id << " " << p.x() << " " << p.y() << " " << p.z() << '\0';
	clString retval;
	int err=TCL::eval(str, retval);
	if(err){
	    cerr << "Error evaluating boundary value" << endl;
	    return;
	}
	double value;
	if(!retval.get_double(value)){
	    cerr << "Bad result from boundary value" << endl;
	    return;
	}
	node->bc=scinew DirichletBC(this, value);
    }
    nodes.add(node);
}

void PointSurface::set_surfnodes(const Array1<NodeHandle>& nodes) {
    NOT_FINISHED("PointSurface::set_surfnodes");
}

void PointSurface::get_surfnodes(Array1<NodeHandle>& nodes)
{
    char id[100];
    if(boundary_type == DirichletExpression){
	// Format this string - we will use it later...
	char proc_string[1000];
	sprintf(id, "PointSurface%p", this);
	ostrstream proc(proc_string, 1000);
	proc << "proc " << id << " {x y z} { expr "
	     << boundary_expr << "}" << '\0';
	TCL::execute(proc_string);
    }
    add_node(nodes, id, pos);
    TCL::execute(clString("rename ")+id+" \"\"");
}

#define POINTSURFACE_VERSION 1

void PointSurface::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("PointSurface", POINTSURFACE_VERSION);
    Surface::io(stream);
    Pio(stream, pos);
    stream.end_class();
}

void PointSurface::construct_grid(int, int, int, const Point &, double)
{
    NOT_FINISHED("PointSurface::construct_grid");
}

void PointSurface::construct_grid()
{
    NOT_FINISHED("PointSurface::construct_grid");
}

GeomObj* PointSurface::get_obj(const ColorMapHandle& cmap)
{

    GeomPts* pts=scinew GeomPts(1);
    pts->pts[0] = pos.x();
    pts->pts[1] = pos.y();
    pts->pts[2] = pos.z();

    if(boundary_type == BdryNone)
	return pts;

    Array1<NodeHandle> nodes;
    get_surfnodes(nodes);

    // This is here twice, since get_surfnodes may reduce back to 
    // no BC's if there is an error...
    if(boundary_type == BdryNone)
	return pts;

//    WE NEED TO BUILD A NODE IN OUR CONSTRUCTOR, RIGHT??

    double v=nodes[0]->bc->value;
    Color c(cmap->lookup(v)->diffuse);

    return pts;
}

static Persistent* make_PointsSurface()
{
    return scinew PointsSurface();
}

PersistentTypeID PointsSurface::type_id("PointsSurface", "Surface",
				       make_PointsSurface);

PointsSurface::PointsSurface(const Array1<Point>& pos, const Array1<double>& val)
: Surface(PointsSurf, 0), pos(pos), val(val)

{
}

PointsSurface::PointsSurface()
: Surface(PointsSurf, 0)
{
}

PointsSurface::~PointsSurface()
{
}

PointsSurface::PointsSurface(const PointsSurface& copy)
: Surface(copy), pos(copy.pos)
{
}

Surface* PointsSurface::clone()
{
    return scinew PointsSurface(*this);
}

int PointsSurface::inside(const Point&)
{
    return 0;
}

#define POINTSSURFACE_VERSION 1

void PointsSurface::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("PointsSurface", POINTSSURFACE_VERSION);
    Surface::io(stream);
    Pio(stream, pos);
    Pio(stream, val);		    
    stream.end_class();
}

void PointsSurface::construct_grid(int, int, int, const Point &, double)
{
    NOT_FINISHED("PointsSurface::construct_grid");
}

void PointsSurface::construct_grid()
{
    NOT_FINISHED("PointsSurface::construct_grid");
}

GeomObj* PointsSurface::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("PointsSurface::get_obj");
    return 0;
}

void PointsSurface::get_surfnodes(Array1<NodeHandle>& nodes)
{
    for (int i=0; i<val.size(); i++) {
	Node* node=new Node(pos[i]);
	node->bc=scinew DirichletBC(this, val[i]);
	nodes.add(node);
    }
}

void PointsSurface::set_surfnodes(const Array1<NodeHandle>& nodes) {
    pos.resize(nodes.size());
    val.resize(nodes.size());
    for (int i=0; i<val.size(); i++) {
	pos[i]=nodes[i]->p;
	if (nodes[i]->bc) {
	    val[i]=nodes[i]->bc->value;
	} else val[i]=0;
    }
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>
template class Array1<Point>;

#endif
