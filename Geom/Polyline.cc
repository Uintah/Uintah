
/*
 *  Polyline.cc: Polyline object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Polyline.h>
#include <Classlib/NotFinished.h>
#include <Geometry/BBox.h>
#include <Geom/Line.h>

GeomPolyline::GeomPolyline()
{
}

GeomPolyline::GeomPolyline(const GeomPolyline& copy)
: GeomVertexPrim(copy)
{
}

GeomPolyline::~GeomPolyline() {
}

GeomObj* GeomPolyline::clone()
{
    return scinew GeomPolyline(*this);
}

void GeomPolyline::make_prims(Array1<GeomObj*>&,
			      Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomPolyline::make_prims");
}

void GeomPolyline::preprocess()
{
    NOT_FINISHED("GeomPolyline::preprocess");
}

void GeomPolyline::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomPolyline::intersect");
}
