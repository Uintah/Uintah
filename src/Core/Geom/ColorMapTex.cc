//static char *id="@(#) $Id$";

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
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/BBox.h>
using std::ostream;

namespace SCICore {
namespace GeomSpace {

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
  using SCICore::PersistentSpace::Pio;

  stream.begin_class("ColorMapTex", COLORMAPTEX_VERSION);
  GeomObj::io(stream);
  SCICore::Geometry::Pio(stream, a);
  SCICore::Geometry::Pio(stream, b);
  SCICore::Geometry::Pio(stream, c);
  SCICore::Geometry::Pio(stream, d);
  stream.end_class();
}

bool ColorMapTex::saveobj(ostream&, const clString&, GeomSave*) {
  NOT_FINISHED("ColorMapTex::saveobj");
  return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
