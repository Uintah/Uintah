
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
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

Persistent* make_GeomRenderMode()
{
    return new GeomRenderMode(GeomRenderMode::WireFrame, 0);
}

PersistentTypeID GeomRenderMode::type_id("GeomRenderMode", "GeomObj", make_GeomRenderMode);

GeomRenderMode::GeomRenderMode(DrawType drawtype, GeomObj* child)
: GeomContainer(child), drawtype(drawtype)
{
}

GeomRenderMode::GeomRenderMode(const GeomRenderMode& copy)
: GeomContainer(copy), drawtype(copy.drawtype)
{
}

GeomRenderMode::~GeomRenderMode()
{
    if(child)
	delete child;
}

GeomObj* GeomRenderMode::clone()
{
    return scinew GeomRenderMode(*this);
}

void GeomRenderMode::make_prims(Array1<GeomObj*>& free,
				Array1<GeomObj*>& dontfree)
{
    if(child)
	child->make_prims(free, dontfree);
}

void GeomRenderMode::intersect(const Ray&, Material*,
			       Hit&)
{
    NOT_FINISHED("GeomRenderMode::intersect");
}

#define GEOMRENDERMODE_VERSION 1

void GeomRenderMode::io(Piostream& stream)
{
    stream.begin_class("GeomRenderMode", GEOMRENDERMODE_VERSION);
    GeomContainer::io(stream);
    int tmp=drawtype;
    Pio(stream, tmp);
    if(stream.reading())
	drawtype=(DrawType)tmp;
    stream.end_class();
}
