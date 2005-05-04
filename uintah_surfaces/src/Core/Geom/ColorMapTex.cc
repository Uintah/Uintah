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
 *  ColorMapTex.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (c) 199? SCI Group
 */

#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomColorMap.h>
#include <Core/Geom/GeomQuads.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

Persistent *make_ColorMapTex()
{
  return scinew ColorMapTex( Point(0,0,0), Point(0,1,0), Point(1,1,0),
			     Point(1,0,0),  0);
}

PersistentTypeID ColorMapTex::type_id("ColorMapTex", "GeomObj",
				      make_ColorMapTex);


ColorMapTex::ColorMapTex(const Point &p1, const Point &p2,
			 const Point &p3, const Point &p4,
			 ColorMapHandle cmap)
  : GeomContainer(0)
{
  GeomFastQuads *quad = scinew GeomFastQuads();
  const double min = cmap->getMin();
  const double max = cmap->getMax();
  quad->add (p1, min, p2, max, p3, max, p4, min);
  child_ = scinew GeomColorMap(quad, cmap);
}


ColorMapTex::ColorMapTex( const ColorMapTex &copy )
  : GeomContainer(copy)
{
}


ColorMapTex::~ColorMapTex()
{
}


GeomObj*
ColorMapTex::clone()
{
  return scinew ColorMapTex( *this );
}


#define COLORMAPTEX_VERSION 2

void
ColorMapTex::io(Piostream& stream)
{
  const int version = stream.begin_class("ColorMapTex", COLORMAPTEX_VERSION);
  if (version > 1)
  {
    GeomContainer::io(stream);
  }
  else
  {
    // This is broken, but backwards compatable
    Point p1, p2, p3, p4;
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, p3);
    Pio(stream, p4);

    GeomFastQuads *quad = scinew GeomFastQuads();
    quad->add (p1, 0.0, p2, 1.0, p3, 1.0, p4, 0.0);
    child_ = scinew GeomColorMap(quad, 0);
  }
  stream.end_class();
}


} // End namespace SCIRun

