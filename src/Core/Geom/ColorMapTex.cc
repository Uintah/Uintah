
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
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
using std::ostream;

namespace SCIRun {

Persistent *make_ColorMapTex() {
  return scinew ColorMapTex( Point(0,0,0), Point(0,1,0), Point(1,1,0),
			   Point(1,0,0) );
}

PersistentTypeID ColorMapTex::type_id("ColorMapTex", "GeomObj", make_ColorMapTex);

ColorMapTex::ColorMapTex(const Point &p1, const Point &p2,
		     const Point &p3, const Point &p4 ) : GeomObj()
{
  a = p1; b = p2; c = p3; d = p4;
}

ColorMapTex::ColorMapTex( const ColorMapTex &copy ) : GeomObj(copy) {
  a = copy.a; b = copy.b;
  c = copy.c; d = copy.d;
}

ColorMapTex::~ColorMapTex()
{
}

GeomObj* ColorMapTex::clone() {
  return scinew ColorMapTex( *this );
}

void ColorMapTex::get_bounds( BBox& bb ) {
  bb.extend( a );
  bb.extend( b );
  bb.extend( c );
  bb.extend( d );
}

#define COLORMAPTEX_VERSION 1

void ColorMapTex::io(Piostream& stream) {

  stream.begin_class("ColorMapTex", COLORMAPTEX_VERSION);
  GeomObj::io(stream);
  Pio(stream, a);
  Pio(stream, b);
  Pio(stream, c);
  Pio(stream, d);
  stream.end_class();
}

bool ColorMapTex::saveobj(ostream&, const clString&, GeomSave*) {
  NOT_FINISHED("ColorMapTex::saveobj");
  return false;
}

} // End namespace SCIRun

