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
 *  HistogramTex.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (c) 199? SCI Group
 */

#include <Core/Geom/HistogramTex.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>

using std::ostream;

namespace SCIRun {

Persistent *make_HistogramTex() {
  return scinew HistogramTex( Point(0,0,0), Point(0,1,0), Point(1,1,0),
			   Point(1,0,0) );
}

PersistentTypeID HistogramTex::type_id("HistogramTex", "GeomObj", make_HistogramTex);

HistogramTex::HistogramTex(const Point &p1, const Point &p2,
			   const Point &p3, const Point &p4) : GeomObj()
{
  a = p1; b = p2; c = p3; d = p4;
}

HistogramTex::HistogramTex( const HistogramTex &copy ) : GeomObj(copy) {
  a = copy.a; b = copy.b;
  c = copy.c; d = copy.d;
}

HistogramTex::~HistogramTex()
{
}

GeomObj* HistogramTex::clone() {
  return scinew HistogramTex( *this );
}

void HistogramTex::get_bounds( BBox& bb ) {
  bb.extend( a );
  bb.extend( b );
  bb.extend( c );
  bb.extend( d );
}

#define HISTOGRAMTEX_VERSION 1

void HistogramTex::io(Piostream& stream) {

  stream.begin_class("HistogramTex", HISTOGRAMTEX_VERSION);
  GeomObj::io(stream);
  Pio(stream, a);
  Pio(stream, b);
  Pio(stream, c);
  Pio(stream, d);
  stream.end_class();
}

} // End namespace SCIRun

