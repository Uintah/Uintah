
/*
 *  Vector.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Vector_h
#define Geometry_Vector_h 1

#include <SCICore/share/share.h>

#include <SCICore/share/share.h>

#include <SCICore/Util/Assert.h>
#include <SCICore/Containers/String.h>

#ifdef KCC
#include <iosfwd.h>  // Forward declarations for KCC C++ I/O routines
#else
class ostream;
class istream;
#endif

namespace SCICore {
namespace Geometry {

using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;

class Point;

class SCICORESHARE Vector {
    double _x,_y,_z;
#if SCI_ASSERTION_LEVEL >= 4
    int uninit;
#endif
public:
    inline Vector(double x, double y, double z): _x(x), _y(y), _z(z)
#if SCI_ASSERTION_LEVEL >= 4
	, uninit(0)
#endif
	    { }
    inline Vector(const Vector&);
    inline Vector();
    double length() const;
    double length2() const;
    friend SCICORESHARE inline double Dot(const Vector&, const Vector&);
    friend SCICORESHARE inline double Dot(const Point&, const Vector&);
    friend SCICORESHARE inline double Dot(const Vector&, const Point&);
    Vector& operator=(const Vector&);

    // checks if one vector is exactly the same as another
    int operator==(const Vector&) const;

    Vector operator*(const double) const;
    Vector& operator*=(const double);
    Vector operator/(const double) const;
    Vector operator/(const Vector&) const;
    Vector operator+(const Vector&) const;
    Vector& operator+=(const Vector&);
    Vector operator-() const;
    Vector operator-(const Vector&) const;
    Vector& operator-=(const Vector&);
    inline double normalize();
    Vector normal() const;
    friend SCICORESHARE inline Vector Cross(const Vector&, const Vector&);
    friend SCICORESHARE inline Vector Abs(const Vector&);
    void x(double);
    inline double x() const;
    void y(double);
    inline double y() const;
    void z(double);
    inline double z() const;

    void u(double);
    inline double u() const;
    void v(double);
    inline double v() const;
    void w(double);
    inline double w() const;

    void rotz90(const int);
    Point point() const;
    
    clString string() const;
    
    friend class Point;
    friend class Transform;
    
    friend SCICORESHARE inline Vector Interpolate(const Vector&, const Vector&, double);
    
    void find_orthogonal(Vector&, Vector&) const;
    
    friend SCICORESHARE void Pio( Piostream&, Vector& );

    inline Point asPoint() const;
    inline double minComponent() const {
	if(_x<_y){
	    if(_x<_z)
		return _x;
	    else
		return _z;
	} else {
	    if(_y<_z)
		return _y;
	    else
		return _z;
	}
    }
    inline double maxComponent() const {
	if(_x>_y){
	    if(_x>_z)
		return _x;
	    else
		return _z;
	} else {
	    if(_y>_z)
		return _y;
	    else
		return _z;
	}
    }
};

SCICORESHARE ostream& operator<<(ostream& os, const Vector& p);
SCICORESHARE istream& operator>>(istream& os, Vector& p);

} // End namespace Geometry
} // End namespace SCICore

#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Geometry {

#include "Vector.icc"

inline Point Vector::asPoint() const {
    return Point(_x,_y,_z);
}

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:57  mcq
// Initial commit
//
// Revision 1.6  1999/07/09 00:27:40  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.5  1999/07/07 21:11:00  dav
// added beginnings of support for g++ compilation
//
// Revision 1.4  1999/06/21 23:52:32  dav
// updated makefiles.main
//
// Revision 1.3  1999/05/06 19:56:17  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:19  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif
