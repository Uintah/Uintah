//static char *id="@(#) $Id$";

/*
 *  GeomTetra.cc:  A tetrahedra object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomTetra.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomTetra()
{
    return scinew GeomTetra(Point(0,0,0), Point(0,0,1), Point(0,1,0), Point(1,0,0));
}

PersistentTypeID GeomTetra::type_id("GeomTetra", "GeomObj", make_GeomTetra);

GeomTetra::GeomTetra(const Point& p1, const Point& p2,
		     const Point& p3, const Point& p4)
: GeomObj(), p1(p1), p2(p2), p3(p3), p4(p4)
{
}

GeomTetra::GeomTetra(const GeomTetra& copy)
: GeomObj(copy), p1(copy.p1), p2(copy.p2), p3(copy.p3), p4(copy.p4)
{
}

GeomTetra::~GeomTetra()
{
}

GeomObj* GeomTetra::clone()
{
    return scinew GeomTetra(*this);
}

void GeomTetra::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
    bb.extend(p4);
}

#define GEOMTETRA_VERSION 1

void GeomTetra::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

    stream.begin_class("GeomTetra", GEOMTETRA_VERSION);
    GeomObj::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, p3);
    Pio(stream, p4);
    stream.end_class();
}

bool GeomTetra::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomTetra::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:25  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:13  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:52  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
