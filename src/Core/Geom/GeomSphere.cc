//static char *id="@(#) $Id$";

/*
 * GeomSphere.cc: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/GeomSphere.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Geom/GeomRaytracer.h>
#include <Geom/GeomSave.h>
#include <Geom/GeomTri.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Geometry/Ray.h>
#include <Malloc/Allocator.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomSphere()
{
    return scinew GeomSphere;
}

PersistentTypeID GeomSphere::type_id("GeomSphere", "GeomObj", make_GeomSphere);

GeomSphere::GeomSphere(int nu, int nv)
: GeomObj(), cen(0,0,0), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: GeomObj(), cen(cen), rad(rad), nu(nu), nv(nv)
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
: GeomObj(copy), cen(copy.cen), rad(copy.rad), nu(copy.nu), nv(copy.nv)
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
    return scinew GeomSphere(*this);
}

void GeomSphere::get_bounds(BBox& bb)
{
    bb.extend(cen, rad);
}

void GeomSphere::get_bounds(BSphere& bs)
{
    bs.extend(cen, rad*1.000001);
}

void GeomSphere::make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>&)
{
    SinCosTable u(nu, 0, 2.*Pi);
    SinCosTable v(nv, 0, Pi, rad);
    double cx=cen.x();
    double cy=cen.y();
    double cz=cen.z();

    for(int j=0;j<nv-1;j++){
	double r0=v.sin(j);
	double z0=v.cos(j);
	double r1=v.sin(j+1);
	double z1=v.cos(j+1);
	for(int i=0;i<nu-1;i++){
	    double x0=u.sin(i);
	    double y0=u.cos(i);
	    double x1=u.sin(i+1);
	    double y1=u.cos(i+1);
	    Point p1(x0*r0+cx, y0*r0+cy, z0+cz);
	    Point p2(x1*r0+cx, y1*r0+cy, z0+cz);
	    Point p3(x0*r1+cx, y0*r1+cy, z1+cz);
	    Point p4(x1*r1+cx, y1*r1+cy, z1+cz);
	    if(j<nv-2){
		GeomTri* t1=scinew GeomTri(p1, p3, p4);
//		t1->set_matl(matl);
		free.add(t1);
	    }
	    if(j>0){
		GeomTri* t2=scinew GeomTri(p1, p4, p2);
//		t2->set_matl(matl);
		free.add(t2);
	    }
	}
    }
}

void GeomSphere::preprocess()
{
    // Nothing to do...
}

void GeomSphere::intersect(const Ray& ray, Material* matl, Hit& hit)
{
  using namespace Geometry;

    Vector OC(cen-ray.origin());
    double tca=Dot(OC, ray.direction());
    double l2oc=OC.length2();
    double radius_sq=rad*rad;
    if(l2oc <= radius_sq){
	// Inside the sphere
	double t2hc=radius_sq-l2oc+tca*tca;
	double thc=Sqrt(t2hc);
	double t=tca+thc;
	hit.hit(t, this, matl);
    } else {
	if(tca < 0.0){
	    // Behind ray, no intersections...
	    return;
	} else {
	    double t2hc=radius_sq-l2oc+tca*tca;
	    if(t2hc <= 0.0){
		// Ray misses, no intersections...
		return;
	    } else {
		double thc=Sqrt(t2hc);
		hit.hit(tca-thc, this, matl);
		// hit.hit(tca+thc, this, ???);
	    }
	}
    }
}

Vector GeomSphere::normal(const Point& hitp, const Hit&)
{
    Vector normal(hitp-cen);
    normal.normalize();
    return normal;
}

#define GEOMSPHERE_VERSION 1

void GeomSphere::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

    stream.begin_class("GeomSphere", GEOMSPHERE_VERSION);
    GeomObj::io(stream);
    Pio(stream, cen);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

bool GeomSphere::saveobj(ostream& out, const clString& format,
			 GeomSave* saveinfo)
{
    cerr << "saveobj Sphere\n";
    if(format == "vrml" || format == "iv"){
	saveinfo->start_tsep(out);
	saveinfo->start_node(out, "Sphere");
	saveinfo->indent(out);
	out << "radius " << rad << "\n";
	saveinfo->end_node(out);
	saveinfo->end_tsep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->indent(out);
	out << "Translate " << cen.x() << " "  << cen.y() << " "  << cen.z() << "\n";
	saveinfo->indent(out);
	out << "Sphere " << rad << " " << -rad << " " << rad << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomSphere::saveobj");
	return false;
    }
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:52  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
