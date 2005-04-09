
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
#include <Classlib/String.h>
#include <Geom/GeomRaytracer.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Geometry/Ray.h>
#include <Malloc/Allocator.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

Persistent* make_GeomDisc()
{
    return scinew GeomDisc;
}

PersistentTypeID GeomDisc::type_id("GeomDisc", "GeomObj", make_GeomDisc);

GeomDisc::GeomDisc(int nu, int nv)
: GeomObj(), n(0,0,1), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomDisc::GeomDisc(const Point& cen, const Vector& n,
		   double rad, int nu, int nv)
: GeomObj(), cen(cen), n(n), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomDisc::move(const Point& _cen, const Vector& _n,
		    double _rad, int _nu, int _nv)
{
    cen=_cen;
    n=_n;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomDisc::GeomDisc(const GeomDisc& copy)
: GeomObj(), v1(copy.v1), v2(copy.v2), cen(copy.cen), n(copy.n),
  rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomDisc::~GeomDisc()
{
}

GeomObj* GeomDisc::clone()
{
    return scinew GeomDisc(*this);
}

void GeomDisc::adjust()
{
    if(n.length2() < 1.e-6){
	cerr << "Degenerate normal on Disc!\n";
    } else {
	n.find_orthogonal(v1, v2);
    }
    n.normalize();
    Vector z(0,0,1);
    if(Abs(n.y()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(n, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, n);
    zrotangle=-Acos(cangle);
}

void GeomDisc::get_bounds(BBox& bb)
{
    bb.extend_cyl(cen, n, rad);
}

void GeomDisc::get_bounds(BSphere& bs)
{
    bs.extend(cen, rad*1.000001);
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
		    GeomTri* t1=scinew GeomTri(l1, l2, p1);
//		    t1->set_matl(matl);
		    free.add(t1);
		}
		GeomTri* t2=scinew GeomTri(l2, p1, p2);
//		t2->set_matl(matl);
		free.add(t2);
	    }
	    l1=p1;
	    l2=p2;
	}
    }
}

void GeomDisc::preprocess()
{
    // Nothing to do...
}

void GeomDisc::intersect(const Ray& ray, Material* matl,
			 Hit& hit)
{
    double tmp=Dot(n, ray.direction());
    if(tmp > -1.e-6 && tmp < 1.e-6)return; // Parallel to plane
    Vector v=cen-ray.origin();
    double t=Dot(n, v)/tmp;
    if(t<1.e-6)return;
    if(hit.hit() && t > hit.t())return;
    Point p(ray.origin()+ray.direction()*t);
    Vector vr(p-cen);
    if(vr.length2() < rad*rad){
	// Hit...
	hit.hit(t, this, matl);
    }
}

Vector GeomDisc::normal(const Point&, const Hit&)
{
    return n;
}

#define GEOMDISC_VERSION 1

void GeomDisc::io(Piostream& stream)
{
    stream.begin_class("GeomDisc", GEOMDISC_VERSION);
    GeomObj::io(stream);
    Pio(stream, cen);
    Pio(stream, n);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

bool GeomDisc::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomDisc::saveobj");
    return false;
}

