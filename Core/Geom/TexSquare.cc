//static char *id="@(#) $Id$";

/*
 *  TexSquare.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (c) 199? SCI Group
 */

#include "TexSquare.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

Persistent *make_TexSquare() {
  return scinew TexSquare( Point(0,0,0), Point(0,1,0), Point(1,1,0),
			   Point(1,0,0) );
}

PersistentTypeID TexSquare::type_id("TexSquare", "GeomObj", make_TexSquare);

TexSquare::TexSquare(const Point &p1, const Point &p2,
		     const Point &p3, const Point &p4 ) : GeomObj()
{
  a = p1; b = p2; c = p3; d = p4;
}

TexSquare::TexSquare( const TexSquare &copy ) : GeomObj(copy) {
  a = copy.a; b = copy.b;
  c = copy.c; d = copy.d;
}

TexSquare::~TexSquare()
{
}

void TexSquare::set_texture( unsigned char *tex, int num, int w ) {
  int i;
  width = w;
  numcolors = num;
  texture = new unsigned char[numcolors*width*3];
  for( i = 0; i < numcolors*width*3; i++ )
    texture[i] = tex[i];
}

GeomObj* TexSquare::clone() {
  return scinew TexSquare( *this );
}

void TexSquare::get_bounds( BBox& bb ) {
  bb.extend( a );
  bb.extend( b );
  bb.extend( c );
  bb.extend( d );
}

void TexSquare::get_bounds(BSphere&) {
  NOT_FINISHED("TexSquare::get_bounds");
}

void TexSquare::make_prims( Array1<GeomObj*>&, Array1<GeomObj*>&)
{
  NOT_FINISHED("TexSquare::make_prims");
}

void TexSquare::preprocess() {
  NOT_FINISHED("TexSquare::preprocess");
}

void TexSquare::intersect(const Ray&, Material*, Hit&) {
  NOT_FINISHED("TexSquare::intersect");
}

#define TEXSQUARE_VERSION 1

void TexSquare::io(Piostream& stream) {
  using SCICore::PersistentSpace::Pio;
  using SCICore::Geometry::Pio;

  stream.begin_class("TexSquare", TEXSQUARE_VERSION);
  GeomObj::io(stream);
  Pio(stream, a);
  Pio(stream, b);
  Pio(stream, c);
  Pio(stream, d);
  stream.end_class();
}

bool TexSquare::saveobj(ostream&, const clString&, GeomSave*) {
  NOT_FINISHED("TexSquare::saveobj");
  return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:24  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:53  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:57  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//
