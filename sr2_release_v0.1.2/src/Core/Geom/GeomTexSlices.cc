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
 *  GeomTexSlices.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Geom/GeomTexSlices.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#ifdef _WIN32
#include <string.h>
#else
#include <strings.h>
#endif
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomTexSlices()
{
    return scinew GeomTexSlices(0,0,0,Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomTexSlices::type_id("GeomTexSlices", "GeomObj", make_GeomTexSlices);

GeomTexSlices::GeomTexSlices(int nx, int ny, int nz, const Point& min,
			     const Point &max)
  : min(min), max(max), nx(nx), ny(ny), nz(nz), have_drawn(0), accum(0.1),
    bright(0.6)
{
    Xmajor.resize(nx, ny, nz);
    Ymajor.resize(ny, nx, nz);
    Zmajor.resize(nz, nx, ny);
}

GeomTexSlices::GeomTexSlices(const GeomTexSlices& copy)
  : GeomObj(copy)
{
}


GeomTexSlices::~GeomTexSlices()
{

}

void GeomTexSlices::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

GeomObj* GeomTexSlices::clone()
{
    return scinew GeomTexSlices(*this);
}

#define GeomTexSlices_VERSION 1

void GeomTexSlices::io(Piostream& stream)
{
    stream.begin_class("GeomTexSlices", GeomTexSlices_VERSION);
    GeomObj::io(stream);
    stream.end_class();
}    

} // End namespace SCIRun

