//static char *id="@(#) $Id$";

/*
 *  Geom.cc: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/GeomObj.h>
#include <Geometry/Vector.h>
#include <iostream.h>

namespace SCICore {
namespace GeomSpace {

PersistentTypeID GeomObj::type_id("GeomObj", "Persistent", 0);

GeomObj::GeomObj()
: parent(0)
{
}

GeomObj::GeomObj(const GeomObj&)
: parent(0)
{
}

GeomObj::~GeomObj()
{
}

void GeomObj::reset_bbox()
{
    // Nothing to do, by default.
}

Vector GeomObj::normal(const Point&, const Hit&)
{
    cerr << "ERROR: GeomObj::normal() shouldn't get called!!!\n";
    return Vector(0,0,1);
}

void GeomObj::set_parent(GeomObj* p)
{
    if(parent){
	cerr << "Warning: Object already has parent!\n";
    }
    parent=p;
}

void GeomObj::io(Piostream&)
{
    // Nothing for now...
}

} // End namespace GeomSpace
} // End namespace SCICore

namespace SCICore {
namespace PersistentSpace {

using namespace GeomSpace;

void Pio( Piostream & stream, GeomObj *& obj )
{
    Persistent* tmp=obj;
    stream.io(tmp, GeomObj::type_id);
    if(stream.reading())
	obj=(GeomObj*)tmp;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:40  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
