//static char *id="@(#) $Id$";

/*
 *  GeomPolyline.cc: Polyline object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomPolyline.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomPolyline()
{
    return scinew GeomPolyline;
}

PersistentTypeID GeomPolyline::type_id("GeomPolyline", "GeomObj", make_GeomPolyline);

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

#define GEOMPOLYLINE_VERSION 1

void GeomPolyline::io(Piostream& stream)
{
    stream.begin_class("GeomPolyline", GEOMPOLYLINE_VERSION);
    GeomVertexPrim::io(stream);
    stream.end_class();
}

bool GeomPolyline::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomPolyline::saveobj");
    return false;
}

GeomPolylineTC::GeomPolylineTC(int drawmode, double drawdist)
: drawmode(drawmode), drawdist(drawdist)
{
}

GeomPolylineTC::~GeomPolylineTC()
{
}

void GeomPolylineTC::add(double t, const Point& p, const Color& c)
{
  int s=data.size();
  data.grow(7);
  data[s]=t;
  data[s+1]=c.r();
  data[s+2]=c.g();
  data[s+3]=c.b();
  data[s+4]=p.x();
  data[s+5]=p.y();
  data[s+6]=p.z();
  bbox.extend(p);
}

GeomPolylineTC::GeomPolylineTC(const GeomPolylineTC& copy)
  : data(copy.data)
{
}

GeomObj* GeomPolylineTC::clone()
{
  return new GeomPolylineTC(*this);
}

void GeomPolylineTC::get_bounds(BBox& box)
{
  box.extend(bbox);
}

void GeomPolylineTC::io(Piostream& stream)
{
  using SCICore::PersistentSpace::Pio;

  SCICore::Geometry::Pio(stream, bbox);
  SCICore::Containers::Pio(stream, data);
}

bool GeomPolylineTC::saveobj(ostream&, const clString&, GeomSave*)
{
  NOT_FINISHED("GeomPolylineTC::saveobj");
  return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/10/07 02:07:43  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/29 00:46:55  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/28 17:54:41  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:23  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:10  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:42  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:52  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//
