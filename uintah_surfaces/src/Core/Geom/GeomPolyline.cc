/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

