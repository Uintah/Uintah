//static char *id="@(#) $Id$";

/*
 *  GeomBox.cc:  A box object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Geom/GeomBox.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>
#include <Math/MinMax.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomBox()
{
    return scinew GeomBox(Point(0,0,0), Point(1,1,1), 1);
}

PersistentTypeID GeomBox::type_id("GeomBox", "GeomObj", make_GeomBox);

GeomBox::GeomBox(const Point& p, const Point& q, int op) : GeomObj()
{
  using SCICore::Geometry::Min;
  using SCICore::Geometry::Max;

  min = Min( p, q );
  max = Max( p, q );

  for (int i=0; i<6; i++ )
    opacity[i] = op;
}

GeomBox::GeomBox(const GeomBox& copy)
: GeomObj(copy)
{
  min = copy.min;
  max = copy.max;
  for (int s=0; s<6; s++)
    opacity[s] = copy.opacity[s];
}

GeomBox::~GeomBox()
{
}

GeomObj* GeomBox::clone()
{
    return scinew GeomBox(*this);
}

void GeomBox::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

void GeomBox::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomBox::get_bounds");
}

void GeomBox::make_prims(Array1<GeomObj*>&, Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomBox::preprocess");
}

void GeomBox::preprocess()
{
    NOT_FINISHED("GeomBox::preprocess");
}

void GeomBox::intersect(const Ray&, Material*,
			  Hit&)
{
    NOT_FINISHED("GeomBox::intersect");
}

#define GEOMBOX_VERSION 1

void GeomBox::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

    stream.begin_class("GeomBox", GEOMBOX_VERSION);
    GeomObj::io(stream);
    Pio(stream, min);
    Pio(stream, max);
    
    for ( int j=0; j<6; j++ )
      Pio(stream, opacity[j]);
    stream.end_class();
}

bool GeomBox::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomBox::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:37  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:49  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

