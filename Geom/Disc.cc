
/*
 *  Disc.h:  Disc object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Disc.h>
#include <Classlib/NotFinished.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

GeomDisc::GeomDisc(int nu, int nv)
: GeomObj(1), nu(nu), nv(nv)
{
}

GeomDisc::GeomDisc(const Point& cen, const Vector& normal,
		   double rad, int nu, int nv)
: GeomObj(1), cen(cen), normal(normal), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomDisc::move(const Point& _cen, const Vector& _normal,
		    double _rad, int _nu, int _nv)
{
    cen=_cen;
    normal=_normal;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomDisc::GeomDisc(const GeomDisc& copy)
: GeomObj(1), cen(copy.cen), normal(copy.normal), rad(copy.rad), nu(copy.nu),
  nv(copy.nv), v1(copy.v1), v2(copy.v2)
{
    adjust();
}

GeomDisc::~GeomDisc()
{
}

GeomObj* GeomDisc::clone()
{
    return new GeomDisc(*this);
}

void GeomDisc::adjust()
{
    if(normal.length2() < 1.e-6){
	cerr << "Degenerate normal on Disc!\n";
    } else {
	normal.find_orthogonal(v1, v2);
    }
    normal.normalize();
    Vector z(0,0,1);
    if(Abs(normal.y()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(normal, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, normal);
    zrotangle=-Acos(cangle);
}

void GeomDisc::get_bounds(BBox& bb)
{
    NOT_FINISHED("GeomDisc::get_bounds");
    bb.extend(cen, rad);
}

void GeomDisc::make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>&)
{
    SinCosTable u(nu, 0, 2.*Pi);
    for(int i=0;i<nv;i++){
	double r1=rad*double(i)/double(nv);
	double r2=rad*double(i+1)/double(nv);
	Point l1, l2;
	for(int j=0;j<nu;j++){
	    double d1=u.sin(j);
	    double d2=u.cos(j);
	    Vector rv1a(v1*(d1*r1));
	    Vector rv1b(v2*(d2*r1));	
	    Vector rv1(rv1a+rv1b);
	    Point p1(cen+rv1);
	    Vector rv2a(v1*(d1*r2));
	    Vector rv2b(v2*(d2*r2));	
	    Vector rv2(rv2a+rv2b);
	    Point p2(cen+rv2);
	    if(j>0){
		if(i>0){
		    GeomTri* t1=new GeomTri(l1, l2, p1);
		    t1->set_matl(matl);
		    free.add(t1);
		}
		GeomTri* t2=new GeomTri(l2, p1, p2);
		t2->set_matl(matl);
		free.add(t2);
	    }
	    l1=p1;
	    l2=p2;
	}
    }
}

void GeomDisc::intersect(const Ray&, const MaterialHandle&,
			 Hit&)
{
    NOT_FINISHED("GeomDisc::intersect");
}
