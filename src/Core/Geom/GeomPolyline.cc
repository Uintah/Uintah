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

#include <Geom/GeomPolyline.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Geometry/BBox.h>
#include <Geom/GeomLine.h>
#include <Malloc/Allocator.h>

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

void GeomPolylineTC::get_bounds(BSphere&)
{
  NOT_FINISHED("GeomPolylineTC::get_bounds");
}

void GeomPolylineTC::make_prims(Array1<GeomObj*>&,
				Array1<GeomObj*>&)
{
  NOT_FINISHED("GeomPolylineTC::make_prims");
}

void GeomPolylineTC::preprocess()
{
  NOT_FINISHED("GeomPolylineTC::preprocess");
}

void GeomPolylineTC::intersect(const Ray&, Material*, Hit&)
{
  NOT_FINISHED("GeomPolylineTC::intersect");
}

void GeomPolylineTC::io(Piostream& stream)
{
  using SCICore::PersistentSpace::Pio;
  using SCICore::Containers::Pio;
  using SCICore::Geometry::Pio;

  Pio(stream, bbox);
  Pio(stream, data);
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
