
/*
 * Torus.cc: Torus objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Torus.h>
#include <Geom/GeomRaytracer.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Geometry/Ray.h>
#include <Classlib/NotFinished.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

GeomTorus::GeomTorus(int nu, int nv)
: GeomObj(), nu(nu), nv(nv), cen(0,0,0), axis(0,0,1), rad1(1), rad2(.1)
{
}

GeomTorus::GeomTorus(const Point& cen, const Vector& axis,
		     double rad1, double rad2, int nu, int nv)
: GeomObj(), cen(cen), axis(axis), rad1(rad1), rad2(rad2), nu(nu), nv(nv)
{
    adjust();
}

void GeomTorus::move(const Point& _cen, const Vector& _axis,
		     double _rad1, double _rad2, int _nu, int _nv)
{
    cen=_cen;
    axis=_axis;
    rad1=_rad1;
    rad2=_rad2;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomTorus::GeomTorus(const GeomTorus& copy)
: GeomObj(copy), cen(copy.cen), axis(copy.axis),
  rad1(copy.rad1), rad2(copy.rad2), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomTorus::~GeomTorus()
{
}

void GeomTorus::adjust()
{
    axis.normalize();

    Vector z(0,0,1);
    if(Abs(axis.y())+Abs(axis.x()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(axis, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, axis);
    zrotangle=-Acos(cangle);
}

GeomObj* GeomTorus::clone()
{
    return new GeomTorus(*this);
}

void GeomTorus::get_bounds(BBox&)
{
    NOT_FINISHED("GeomTorus::get_bounds");
}

void GeomTorus::get_bounds(BSphere& bs)
{
    bs.extend(cen, (rad1+rad2)*1.000001);
}

void GeomTorus::make_prims(Array1<GeomObj*>&,
			    Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomTorus::make_prims");
}

void GeomTorus::preprocess()
{
    // Nothing to do...
}

void GeomTorus::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTorus::intersect");
}

Vector GeomTorus::normal(const Point&, const Hit&)
{
    NOT_FINISHED("GeomTorus::normal");
    return Vector(0,0,1);
}

