
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
#include <Malloc/Allocator.h>
#include <Math/Trig.h>
#include <Math/TrigTable.h>

static Persistent* make_CylinderSurface()
{
    return scinew CylinderSurface(Point(0,0,0), Point(0,0,1),1,10,10,10);
}

PersistentTypeID CylinderSurface::type_id("CylinderSurface", "Surface",
					  make_CylinderSurface);

CylinderSurface::CylinderSurface(const Point& p1, const Point& p2,
				 double radius, int nu, int nv, int ndiscu)
: Surface(Other, 1),
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

void CylinderSurface::get_surfpoints(Array1<Point>& pts)
{
    pts.add(p1);
    SinCosTable tab(nv, 0, 2*Pi, radius);
    for(int i=1;i<ndiscu-1;i++){
	double r=double(i)/double(ndiscu-1);
	for(int j=0;j<nv;j++){
	    Point p(p1+(u*tab.sin(j)+v*tab.cos(j))*r);
	    pts.add(p);
	}
    }
    for(i=0;i<=nu;i++){
	double h=double(i)/double(nu)*height;
	for(int j=0;j<nv;j++){
	    Point p(p1+u*tab.sin(j)+v*tab.cos(j)+axis*h);
	    pts.add(p);
	}
    }
    for(i=ndiscu-2;i>=1;i--){
	double r=double(i)/double(ndiscu-1);
	for(int j=0;j<nv;j++){
	    Point p(p1+(u*tab.sin(j)+v*tab.cos(j))*r);
	    pts.add(p);
	}
    }
    pts.add(p2);
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

static Persistent* make_PointSurface()
{
    return scinew PointSurface(Point(0,0,0));
}

PersistentTypeID PointSurface::type_id("PointSurface", "Surface",
				       make_PointSurface);

PointSurface::PointSurface(const Point& pos)
: Surface(Other, 0), pos(pos)
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

void PointSurface::get_surfpoints(Array1<Point>& pts)
{
    pts.add(pos);
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

