
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

static Persistent* make_CylinderSurface()
{
    return new CylinderSurface(Point(0,0,0), Point(0,0,1),1,10,10,10);
}

PersistentTypeID CylinderSurface::type_id("CylinderSurface", "Surface",
					  make_CylinderSurface);

CylinderSurface::CylinderSurface(const Point& p1, const Point& p2,
				 double radius, int nu, int nv, int ndiscu)
: Surface(Other),
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
    return new CylinderSurface(*this);
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
#if 0
    for(int i=1;i<ndiscu-1;i++){
	
    }
    for(int i=0;i<=nu;i++){
	for(int j=0;j<nv;j++){
	    Point p(u*du+v*dv+p1);
	    pts.add(p);
	}
    }
    for(int i=ndiscu-2;i>=1;i++){
	
    }
#endif
    NOT_FINISHED("CylinderSurface::get_surfpoints");
    pts.add(p2);
}

#define CYLINDERSURFACE_VERSION 1

void CylinderSurface::io(Piostream& stream)
{
    int version=stream.begin_class("CylinderSurface", CYLINDERSURFACE_VERSION);
    Surface::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, radius);
    Pio(stream, nu);
    Pio(stream, nv);
    Pio(stream, ndiscu);
    stream.end_class();
}

