/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Geom/GeomPolyline.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

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

  Pio(stream, bbox);
  Pio(stream, data);
}

} // End namespace SCIRun

