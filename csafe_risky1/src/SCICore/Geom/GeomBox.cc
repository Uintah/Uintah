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

#include <SCICore/Geom/GeomBox.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MinMax.h>
#include <iostream>
using std::ostream;

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

#define GEOMBOX_VERSION 1

void GeomBox::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomBox", GEOMBOX_VERSION);
    GeomObj::io(stream);
    SCICore::Geometry::Pio(stream, min);
    SCICore::Geometry::Pio(stream, max);
    
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
// Revision 1.6  1999/10/07 02:07:40  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/29 00:46:53  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/28 17:54:39  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:19  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:05  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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

