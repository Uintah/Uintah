//static char *id="@(#) $Id$";

/*
 *  Container.cc: Base class for container objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomContainer.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace GeomSpace {

PersistentTypeID GeomContainer::type_id("GeomContainer", "GeomObj", 0);

GeomContainer::GeomContainer(GeomObj* child)
: GeomObj(), child(child)
{
}

GeomContainer::GeomContainer(const GeomContainer& copy)
: GeomObj(copy), child(copy.child->clone())
{
}

GeomContainer::~GeomContainer()
{
    if(child)
	delete child;
}

void GeomContainer::get_bounds(BBox& bbox)
{
    child->get_bounds(bbox);
}

#define GEOMCONTAINER_VERSION 1

void GeomContainer::io(Piostream& stream)
{
    stream.begin_class("GeomContainer", GEOMCONTAINER_VERSION);
    GeomObj::io(stream);
    Pio(stream, child);
    stream.end_class();
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/23 07:06:33  sparker
// Fix IRIX build
//
// Revision 1.3  1999/08/17 23:50:19  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:06  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:38  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//
