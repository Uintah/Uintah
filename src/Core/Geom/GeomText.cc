//static char *id="@(#) $Id$";

/*
 *  GeomText.cc:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/BSphere.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geom/GeomText.h>

namespace SCICore {
namespace GeomSpace {

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
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

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


} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:14  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:45  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:53  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

