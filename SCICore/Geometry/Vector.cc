//static char *id="@(#) $Id$";

/*
 *  Vector.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MiscMath.h>
#include <iostream.h>
#include <stdio.h>

namespace SCICore {
namespace Geometry {

clString
Vector::string() const
{
#if 0
    return clString("[")
	+to_string(_x)+clString(", ")
	    +to_string(_y)+clString(", ")
		+to_string(_z)+clString("]");
#endif
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", _x, _y, _z);
    return clString(buf);
}

void
Vector::find_orthogonal(Vector& v1, Vector& v2) const
{
    Vector v0(Cross(*this, Vector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(*this, Vector(0,1,0));
    }
    v1=Cross(*this, v0);
    v1.normalize();
    v2=Cross(*this, v1);
    v2.normalize();
}

Vector
Vector::normal() const
{
   Vector v(*this);
   v.normalize();
   return v;			// 
}

ostream& operator<<( ostream& os, const Vector& v )
{
   os << v.string();
   return os;
}

istream& operator>>( istream& is, Vector& v)
{
  double x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Vector(x,y,z);
  return is;
}

int
Vector::operator== ( const Vector& v ) const
{
    return v._x == _x && v._y == _y && v._z == _z;
}

void
Pio(Piostream& stream, Vector& p)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}

void Vector::rotz90(const int c)
{
    // Rotate by c*90 degrees counter clockwise
    switch(c%4){
    case 0:
	// 0 degrees, do nothing
	break;
    case 1:
	// 90 degrees
	{
	    double newx=-_y;
	    _y=_x;
	    _x=newx;
	}
	break;
    case 2:
	// 180 degrees
	_x=-_x;
	_y=-_y;
	break;
    case 3:
	// 270 degrees
	{
	    double newy=-_x;
	    _x=_y;
	    _y=newy;
	}
	break;
    }
}


} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:53  sparker
// Various #include cleanups
//
// Revision 1.3  1999/09/04 06:01:53  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.2  1999/08/17 06:39:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:57  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:11:00  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

