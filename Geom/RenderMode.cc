
/*
 * RenderMode.cc: RenderMode objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/RenderMode.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

GeomRenderMode::GeomRenderMode(DrawType drawtype, GeomObj* child)
: GeomObj(0), drawtype(drawtype), child(child)
{
}

GeomRenderMode::GeomRenderMode(const GeomRenderMode& copy)
: GeomObj(0), drawtype(copy.drawtype), child(copy.child->clone())
{
}

GeomRenderMode::~GeomRenderMode()
{
    if(child)
	delete child;
}

GeomObj* GeomRenderMode::clone()
{
    return new GeomRenderMode(*this);
}

void GeomRenderMode::get_bounds(BBox& bb)
{
    if(child)
	child->get_bounds(bb);
}

void GeomRenderMode::make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree)
{
    if(child)
	child->make_prims(free, dontfree);
}
