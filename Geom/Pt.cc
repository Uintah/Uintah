
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
    return new GeomPts(0);
}

PersistentTypeID GeomPts::type_id("GeomPts", "GeomObj", make_GeomPts);

GeomPts::GeomPts(const GeomPts &copy)
: pts(copy.pts) {
}

GeomPts::GeomPts(int size)
: pts(0, size)
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
    for (int i=0; i<pts.size(); i++)
	bb.extend(pts[i]);
}

void GeomPts::get_bounds(BSphere& bs)
{
    for (int i=0; i<pts.size(); i++)
	bs.extend(pts[i]);
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
    NOT_FINISHED("Don't know the normal to a point -- returning (1,0,0)");
    return(Vector(1,0,0));
}

#define GEOMPTS_VERSION 1

void GeomPts::io(Piostream& stream)
{
    stream.begin_class("GeomPts", GEOMPTS_VERSION);
    GeomObj::io(stream);
    Pio(stream, pts);
    stream.end_class();
}
