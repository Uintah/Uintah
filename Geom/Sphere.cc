
/*
 * Sphere.cc: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Sphere.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

GeomSphere::GeomSphere()
: GeomObj(1)
{
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: GeomObj(1), cen(cen), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomSphere::move(const Point& _cen, double _rad, int _nu, int _nv)
{
    cen=_cen;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomSphere::GeomSphere(const GeomSphere& copy)
: GeomObj(1), cen(copy.cen), rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomSphere::~GeomSphere()
{
}

void GeomSphere::adjust()
{
}

GeomObj* GeomSphere::clone()
{
    return new GeomSphere(*this);
}

void GeomSphere::get_bounds(BBox& bb)
{
    bb.extend(cen, rad);
}

void GeomSphere::make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>&)
{
    SinCosTable u(nu, 0, 2.*Pi);
    SinCosTable v(nv, 0, Pi, rad);
    double cx=cen.x();
    double cy=cen.y();
    double cz=cen.z();
    for(int i=0;i<nu-1;i++){
	double x0=u.sin(i);
	double y0=u.cos(i);
	double x1=u.sin(i+1);
	double y1=u.cos(i+1);
	Point l1, l2;
	for(int j=0;j<nv-1;j++){
	    double r0=v.sin(j);
	    double z0=v.cos(j);
	    Point p1(x0*r0+cx, y0*r0+cy, z0+cz);
	    Point p2(x1*r0+cx, y1*r0+cy, z0+cz);
	    if(j>0){
		GeomTri* t1=new GeomTri(l1, l2, p1);
		t1->set_matl(matl);
		free.add(t1);
		GeomTri* t2=new GeomTri(l2, p1, p2);
		t2->set_matl(matl);
		free.add(t2);
	    }
	    l1=p1;
	    l2=p2;
	}
    }
}
