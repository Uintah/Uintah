//static char *id="@(#) $Id$";

/*
 * GeomRenderMode.cc: RenderMode objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/GeomRenderMode.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Geom/GeomTri.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>
#include <Math/TrigTable.h>
#include <Math/Trig.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomRenderMode()
{
    return scinew GeomRenderMode(GeomRenderMode::WireFrame, 0);
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
    PersistentSpace::Pio(stream, tmp);
    if(stream.reading())
	drawtype=(DrawType)tmp;
    stream.end_class();
}

bool GeomRenderMode::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomRenderMode::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:43  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

