
/*
 *  TriStrip.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/TriStrip.h>
#include <Classlib/NotFinished.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>

GeomTriStrip::GeomTriStrip()
{
}

GeomTriStrip::GeomTriStrip(const GeomTriStrip& copy)
: GeomVertexPrim(copy)
{
}

GeomTriStrip::~GeomTriStrip() {
}

void GeomTriStrip::make_prims(Array1<GeomObj*>&,
			      Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomTriStrip::make_prims");
}

GeomObj* GeomTriStrip::clone()
{
    return scinew GeomTriStrip(*this);
}

void GeomTriStrip::preprocess()
{
    NOT_FINISHED("GeomTriStrip::preprocess");
}

void GeomTriStrip::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTriStrip::intersect");
}
