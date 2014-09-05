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

#include <iostream>
using std::cerr;
using std::endl;

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

void GeomContainer::get_triangles( Array1<float> &v)
{
  if (child)
    child->get_triangles(v);
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
// Revision 1.4.2.3  2000/10/26 17:18:35  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5  2000/06/06 16:01:43  dahart
// - Added get_triangles() to several classes for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
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
