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

#include <Core/Geom/GeomTetra.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

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

    stream.begin_class("GeomTetra", GEOMTETRA_VERSION);
    GeomObj::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, p3);
    Pio(stream, p4);
    stream.end_class();
}

} // End namespace SCIRun

