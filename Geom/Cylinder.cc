
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
#include <Math/TrigTable.h>
#include <Math/Trig.h>
#include <Classlib/String.h>

GeomCylinder::GeomCylinder()
: GeomObj(1)
{
}

GeomCylinder::GeomCylinder(const Point& bottom, const Point& top,
			   double rad, int nu, int nv)
: GeomObj(1), bottom(bottom), top(top), rad(rad), nu(nu), nv(nv)
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
: GeomObj(1), bottom(copy.bottom), top(copy.top), rad(copy.rad), nu(copy.nu),
  nv(copy.nv), axis(copy.axis), v1(copy.v1), v2(copy.v2)
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
    if(Abs(axis.y()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,1,0);
    } else {
	zrotaxis=Cross(axis, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, axis)/height;
    zrotangle=Acos(cangle);
}

GeomObj* GeomCylinder::clone()
{
    return new GeomCylinder(*this);
}

void GeomCylinder::get_bounds(BBox& bb)
{
    NOT_FINISHED("GeomCylinder::get_bounds");
    bb.extend(bottom);
    bb.extend(top);
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
