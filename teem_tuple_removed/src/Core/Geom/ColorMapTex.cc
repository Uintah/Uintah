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

