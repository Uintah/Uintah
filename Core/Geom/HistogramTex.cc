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

#include "HistogramTex.h"
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

