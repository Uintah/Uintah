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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Vector.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace SCICore {
namespace GeomSpace {

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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.5.2.3  2000/10/26 17:18:36  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.8  2000/09/11 22:14:46  bigler
// Added constructors that take an int and IntVector to allow unique
// identification in 4 dimensions.
//
// Revision 1.7  2000/08/09 18:21:14  kuzimmer
// Added IntVector indexing to GeomObj & GeomSphere
//
// Revision 1.6  2000/06/06 16:01:45  dahart
// - Added get_triangles() to several classes for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
// Revision 1.5  2000/01/03 20:12:36  kuzimmer
//  Forgot to check in these files for picking spheres
//
// Revision 1.4  1999/09/08 02:26:50  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/17 23:50:22  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:09  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:40  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
