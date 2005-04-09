#include "TexSquare.h"
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <Geometry/BBox.h>

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


