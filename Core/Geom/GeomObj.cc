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

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {

PersistentTypeID GeomObj::type_id("GeomObj", "Persistent", 0);

GeomObj::GeomObj(int id) : id(id),
  _id(0x1234567,0x1234567,0x1234567)
{
}

GeomObj::GeomObj(IntVector i)
  :id( 0x1234567 ), _id(i)
{
}

GeomObj::GeomObj(int id_int, IntVector i)
  :id( id_int ), _id(i)
{
}

GeomObj::GeomObj(const GeomObj&)
{
}

GeomObj::~GeomObj()
{
}

void GeomObj::get_triangles( Array1<float> &)
{
  cerr << "GeomObj::get_triangles - no triangles" << endl;
}

void GeomObj::reset_bbox()
{
    // Nothing to do, by default.
}

void GeomObj::io(Piostream&)
{
    // Nothing for now...
}

void Pio( Piostream & stream, GeomObj *& obj )
{
    Persistent* tmp=obj;
    stream.io(tmp, GeomObj::type_id);
    if(stream.reading())
	obj=(GeomObj*)tmp;
}

} // End namespace SCIRun
