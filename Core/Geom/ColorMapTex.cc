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

#include "ColorMapTex.h"
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
using std::ostream;

namespace SCIRun {

Persistent *make_ColorMapTex()
{
  return scinew ColorMapTex( Point(0,0,0), Point(0,1,0), Point(1,1,0),
			     Point(1,0,0) );
}

PersistentTypeID ColorMapTex::type_id("ColorMapTex", "GeomObj", make_ColorMapTex);


ColorMapTex::ColorMapTex(const Point &p1, const Point &p2,
			 const Point &p3, const Point &p4)
  : GeomObj(),
    a_(p1),
    b_(p2),
    c_(p3),
    d_(p4),
    numcolors_(256)
{
  memset(texture_, 0, numcolors_ * 4);
}


ColorMapTex::ColorMapTex( const ColorMapTex &copy )
  : GeomObj(copy),
    a_(copy.a_),
    b_(copy.b_),
    c_(copy.c_),
    d_(copy.d_),
    numcolors_(copy.numcolors_)
{
  memcpy(texture_, copy.texture_, numcolors_ * 4);
}


ColorMapTex::~ColorMapTex()
{
}


GeomObj*
ColorMapTex::clone()
{
  return scinew ColorMapTex( *this );
}


void
ColorMapTex::get_bounds( BBox& bb )
{
  bb.extend(a_);
  bb.extend(b_);
  bb.extend(c_);
  bb.extend(d_);
}


#define COLORMAPTEX_VERSION 1

void
ColorMapTex::io(Piostream& stream)
{
  const int ver = stream.begin_class("ColorMapTex", COLORMAPTEX_VERSION);
  GeomObj::io(stream);
  Pio(stream, a_);
  Pio(stream, b_);
  Pio(stream, c_);
  Pio(stream, d_);
  if (ver > 1)
  {
    Pio(stream, numcolors_);
    //Pio(stream, texture_);
  }
  stream.end_class();
}


} // End namespace SCIRun

