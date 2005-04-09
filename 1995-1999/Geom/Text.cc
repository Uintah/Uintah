
/*
 *  Text.cc:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>
#include <Geom/Save.h>
#include <Geom/Text.h>



int    GeomText::init = 1;
GLuint GeomText::fontbase = 0;

static Persistent* make_GeomText()
{
    return scinew GeomText;
}

PersistentTypeID GeomText::type_id("GeomText", "GeomObj", make_GeomText);

GeomText::GeomText()
  : GeomObj()
{
}

GeomText::GeomText( const clString &text, const Point &at, const Color &c)
: GeomObj(), text(text), at(at), c(c)
{
}

GeomText::GeomText(const GeomText& copy)
: GeomObj(copy)
{
  init = copy.init;
  fontbase = copy.fontbase;
  text = copy.text;
  at = copy.at;
  c = copy.c;
}


GeomObj* GeomText::clone()
{
    return scinew GeomText(*this);
}

void GeomText::get_bounds(BBox& in_bb)
{
  in_bb.extend( at );
}

void GeomText::get_bounds(BSphere& in_sphere)
{
  in_sphere.extend( at );
}


void GeomText::make_prims(Array1<GeomObj*>&,
			 Array1<GeomObj*>&)
{
}


GeomText::~GeomText()
{
}

void GeomText::reset_bbox()
{
}

void GeomText::preprocess()
{
}

void GeomText::intersect(const Ray&, Material*, Hit& )
{
}

#define GEOMGROUP_VERSION 1

void GeomText::io(Piostream& stream)
{
    stream.begin_class("GeomText", GEOMGROUP_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, at);
    Pio(stream, text);
    stream.end_class();
}

bool GeomText::saveobj(ostream&, const clString&, GeomSave*)
{
  return 0;
}

