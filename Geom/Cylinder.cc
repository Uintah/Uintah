
/*
 *  Cylinder.h: Cylinder Object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Cylinder.h>
#include <Classlib/NotFinished.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>
#include <Classlib/String.h>

GeomCylinder::GeomCylinder(int nu, int nv)
: GeomObj(), nu(nu), nv(nv), top(0,0,1), bottom(0,0,0), rad(1)
{
    adjust();
}

GeomCylinder::GeomCylinder(const Point& bottom, const Point& top,
			   double rad, int nu, int nv)
: GeomObj(), bottom(bottom), top(top), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomCylinder::move(const Point& _bottom, const Point& _top,
			double _rad, int _nu, int _nv)
{
    bottom=_bottom;
    top=_top;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomCylinder::GeomCylinder(const GeomCylinder& copy)
: GeomObj(copy), v1(copy.v1), v2(copy.v2), bottom(copy.bottom), top(copy.top),
  rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomCylinder::~GeomCylinder()
{
}

void GeomCylinder::adjust()
{
    axis=top-bottom;
    height=axis.length();
    if(height < 1.e-6){
	cerr << "Degenerate cylinder!\n";
    } else {
	axis.find_orthogonal(v1, v2);
    }
    v1*=rad;
    v2*=rad;

    Vector z(0,0,1);
    if(Abs(axis.y())+Abs(axis.x()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(axis, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, axis)/height;
    zrotangle=-Acos(cangle);
}

GeomObj* GeomCylinder::clone()
{
    return new GeomCylinder(*this);
}

void GeomCylinder::get_bounds(BBox& bb)
{
    bb.extend_cyl(bottom, axis, rad);
    bb.extend_cyl(top, axis, rad);
}

void GeomCylinder::get_bounds(BSphere& bs)
{
    Point cen(Interpolate(bottom, top, 0.5));
    double h2=height/2.;
    double r=Sqrt(h2*h2+rad*rad);
    bs.extend(cen, r);
}

void GeomCylinder::make_prims(Array1<GeomObj*>& free,
			      Array1<GeomObj*>&)
{
    SinCosTable u(nu, 0, 2.*Pi);
    for(int i=0;i<nv;i++){
	double z1=double(i)/double(nv);
	double z2=double(i+1)/double(nv);
	Point b1(bottom+axis*z1);
	Point b2(bottom+axis*z2);
	Point l1, l2;
	for(int j=0;j<nu;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Vector rv(v1*d1+v2*d2);
	    Point p1(b1+rv);
	    Point p2(b2+rv);
	    if(j>0){
		GeomTri* t1=new GeomTri(l1, l2, p1);
//		t1->set_matl(matl);
		free.add(t1);
		GeomTri* t2=new GeomTri(l2, p1, p2);
//		t2->set_matl(matl);
		free.add(t2);
	    }
	    l1=p1;
	    l2=p2;
	}
    }
}

void GeomCylinder::preprocess()
{
    NOT_FINISHED("GeomCylidner::preprocess");
}

void GeomCylinder::intersect(const Ray&, Material*,
			     Hit&)
{
    NOT_FINISHED("GeomCylinder::intersect");
}

// Capped Geometry....

GeomCappedCylinder::GeomCappedCylinder(int nu, int nv, int nvdisc)
: GeomCylinder(nu, nv), nvdisc(nvdisc)
{
}

GeomCappedCylinder::GeomCappedCylinder(const Point& bottom, const Point& top,
				       double rad, int nu, int nv, int nvdisc)
: GeomCylinder(bottom, top, rad, nu, nv), nvdisc(nvdisc)
{
}

GeomCappedCylinder::GeomCappedCylinder(const GeomCappedCylinder& copy)
: GeomCylinder(copy), nvdisc(copy.nvdisc)
{
}

GeomCappedCylinder::~GeomCappedCylinder()
{
}

GeomObj* GeomCappedCylinder::clone()
{
    return new GeomCappedCylinder(*this);
}

void GeomCappedCylinder::make_prims(Array1<GeomObj*>&,
				    Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomCappedCylinder::make_prims");
}

void GeomCappedCylinder::preprocess()
{
    NOT_FINISHED("GeomCappedCylinder::preprocess");
}

void GeomCappedCylinder::intersect(const Ray&, Material*,
			     Hit&)
{
    NOT_FINISHED("GeomCappedCylinder::intersect");
}
