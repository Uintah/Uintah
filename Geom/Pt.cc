
/*
 * Pt.cc: Pts objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Pt.h>
#include <Classlib/String.h>
#include <Geom/GeomRaytracer.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Geometry/Ray.h>
#include <Malloc/Allocator.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>
#include <Classlib/NotFinished.h>

Persistent* make_GeomPts()
{
    return scinew GeomPts(0);
}

PersistentTypeID GeomPts::type_id("GeomPts", "GeomObj", make_GeomPts);

GeomPts::GeomPts(const GeomPts &copy)
: pts(copy.pts), have_normal(copy.have_normal), n(copy.n) {
}

GeomPts::GeomPts(int size)
: pts(0, size*3), have_normal(0)
{
}

GeomPts::GeomPts(int size, const Vector &n)
: pts(0, size*3), n(n), have_normal(1)
{
}

GeomPts::~GeomPts()
{
}

GeomObj* GeomPts::clone()
{
    return scinew GeomPts(*this);
}

void GeomPts::get_bounds(BBox& bb)
{
    for (int i=0; i<pts.size(); i+=3)
	bb.extend(Point(pts[i], pts[i+1], pts[i+2]));
}

void GeomPts::get_bounds(BSphere& bs)
{
    for (int i=0; i<pts.size(); i+=3)
	bs.extend(Point(pts[i], pts[i+1], pts[i+2]));
}

void GeomPts::make_prims(Array1<GeomObj*>&,
			    Array1<GeomObj*>&)
{
    // Nothing to do...
}

void GeomPts::preprocess()
{
    // Nothing to do...
}

void GeomPts::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("Can't render points yet.");
}

Vector GeomPts::normal(const Point&, const Hit&)
{
    if (have_normal) return n;
    else {
	NOT_FINISHED("These points don't have normals -- returning (1,0,0)");
	return(Vector(1,0,0));
    }
}

#define GEOMPTS_VERSION 2

void GeomPts::io(Piostream& stream)
{
    int version=stream.begin_class("GeomPts", GEOMPTS_VERSION);
    GeomObj::io(stream);
    Pio(stream, pts);
    if (version > 1) {
	Pio(stream, have_normal);
	Pio(stream, n);
    }
    stream.end_class();
}

bool GeomPts::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomPts::saveobj");
    return false;
}

