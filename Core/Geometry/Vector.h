
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

#include <SCICore/Util/Assert.h>
#include <SCICore/Math/Expon.h>

#include <iosfwd>

namespace SCICore {
    namespace Containers {
	class clString;
    }
    namespace PersistentSpace {
	class Piostream;
    }

namespace Geometry {

using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;

class Point;

class SCICORESHARE Vector {
    double _x,_y,_z;
public:
    inline explicit Vector(const Point&);
    inline Vector(double x, double y, double z): _x(x), _y(y), _z(z)
	    { }
    inline Vector(const Vector&);
    inline Vector();
    inline double length() const;
    inline double length2() const;
    friend SCICORESHARE inline double Dot(const Vector&, const Vector&);
    friend SCICORESHARE inline double Dot(const Point&, const Vector&);
    friend SCICORESHARE inline double Dot(const Vector&, const Point&);
    inline Vector& operator=(const Vector&);

    inline double& operator()(int idx) {
	// Ugly, but works
	return (&_x)[idx];
    }

    inline double operator()(int idx) const {
	// Ugly, but works
	return (&_x)[idx];
    }

    // checks if one vector is exactly the same as another
    int operator==(const Vector&) const;

    inline Vector operator*(const double) const;
    inline Vector operator*(const Vector&) const;
    inline Vector& operator*=(const double);
    inline Vector operator/(const double) const;
    inline Vector operator/(const Vector&) const;
    inline Vector& operator/=(const double);
    inline Vector operator+(const Vector&) const;
    inline Vector& operator+=(const Vector&);
    inline Vector operator-() const;
    inline Vector operator-(const Vector&) const;
    inline Vector& operator-=(const Vector&);
    inline double normalize();
    Vector normal() const;
    friend SCICORESHARE inline Vector Cross(const Vector&, const Vector&);
    friend SCICORESHARE inline Vector Abs(const Vector&);
    inline void x(double);
    inline double x() const;
    inline void y(double);
    inline double y() const;
    inline void z(double);
    inline double z() const;

    inline void u(double);
    inline double u() const;
    inline void v(double);
    inline double v() const;
    inline void w(double);
    inline double w() const;

    void rotz90(const int);
    inline Point point() const;
    
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

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Vector& p);
SCICORESHARE std::istream& operator>>(std::istream& os, Vector& p);

} // End namespace Geometry
} // End namespace SCICore

// This cannot be above due to circular dependencies
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Geometry {

inline Vector::Vector(const Point& p)
    : _x(p._x), _y(p._y), _z(p._z)
{
}

inline Vector::Vector()
{
}

inline Vector::Vector(const Vector& p)
{
    _x=p._x;
    _y=p._y;
    _z=p._z;
}

inline double Vector::length2() const
{
    return _x*_x+_y*_y+_z*_z;
}

inline Vector& Vector::operator=(const Vector& v)
{
    _x=v._x;
    _y=v._y;
    _z=v._z;
    return *this;
}

inline Vector Vector::operator*(const double s) const
{
    return Vector(_x*s, _y*s, _z*s);
}

inline Vector Vector::operator/(const double d) const
{
    return Vector(_x/d, _y/d, _z/d);
}

inline Vector Vector::operator/(const Vector& v2) const
{
    return Vector(_x/v2._x, _y/v2._y, _z/v2._z);
}

inline Vector Vector::operator+(const Vector& v2) const
{
    return Vector(_x+v2._x, _y+v2._y, _z+v2._z);
}

inline Vector Vector::operator*(const Vector& v2) const
{
    return Vector(_x*v2._x, _y*v2._y, _z*v2._z);
}

inline Vector Vector::operator-(const Vector& v2) const
{
    return Vector(_x-v2._x, _y-v2._y, _z-v2._z);
}

inline Vector& Vector::operator+=(const Vector& v2)
{
    _x+=v2._x;
    _y+=v2._y;
    _z+=v2._z;
    return *this;
}

inline Vector& Vector::operator-=(const Vector& v2)
{
    _x-=v2._x;
    _y-=v2._y;
    _z-=v2._z;
    return *this;
}

inline Vector Vector::operator-() const
{
    return Vector(-_x,-_y,-_z);
}

inline double Vector::length() const
{
    return Sqrt(_x*_x+_y*_y+_z*_z);
}

inline Vector Abs(const Vector& v)
{
    double x=v._x<0?-v._x:v._x;
    double y=v._y<0?-v._y:v._y;
    double z=v._z<0?-v._z:v._z;
    return Vector(x,y,z);
}

inline Vector Cross(const Vector& v1, const Vector& v2)
{
    return Vector(
	v1._y*v2._z-v1._z*v2._y,
	v1._z*v2._x-v1._x*v2._z,
	v1._x*v2._y-v1._y*v2._x);
}

inline Vector Interpolate(const Vector& v1, const Vector& v2,
			  double weight)
{
    double weight1=1.0-weight;
    return Vector(
	v2._x*weight+v1._x*weight1,
	v2._y*weight+v1._y*weight1,
	v2._z*weight+v1._z*weight1);
}

inline Vector& Vector::operator*=(const double d)
{
    _x*=d;
    _y*=d;
    _z*=d;
    return *this;
}

inline Vector& Vector::operator/=(const double d)
{
    _x/=d;
    _y/=d;
    _z/=d;
    return *this;
}

inline void Vector::x(double d)
{
    _x=d;
}

inline double Vector::x() const
{
    return _x;
}

inline void Vector::y(double d)
{
    _y=d;
}

inline double Vector::y() const
{
    return _y;
}

inline void Vector::z(double d)
{
    _z=d;
}

inline double Vector::z() const
{
    return _z;
}



inline void Vector::u(double d)
{
    _x=d;
}

inline double Vector::u() const
{
    return _x;
}

inline void Vector::v(double d)
{
    _y=d;
}

inline double Vector::v() const
{
    return _y;
}

inline void Vector::w(double d)
{
    _z=d;
}

inline double Vector::w() const
{
    return _z;
}

inline Point Vector::point() const
{
    return Point(_x,_y,_z);
}

inline double Dot(const Vector& v1, const Vector& v2)
{
    return v1._x*v2._x+v1._y*v2._y+v1._z*v2._z;
}

inline double Dot(const Vector& v, const Point& p)
{
    return v._x*p._x+v._y*p._y+v._z*p._z;
}

inline
double Vector::normalize()
{
    double l2=_x*_x+_y*_y+_z*_z;
    double l=Sqrt(l2);
    ASSERT(l>0.0);
    _x/=l;
    _y/=l;
    _z/=l;
    return l;
}

inline Point Vector::asPoint() const {
    return Point(_x,_y,_z);
}

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.9  2000/07/06 00:05:06  tan
// Made const works for operator()
//
// Revision 1.8  2000/07/05 21:38:47  tan
// Added /= operator.
//
// Revision 1.7  2000/06/15 20:43:19  sparker
// Added "inline" statements in class file
//
// Revision 1.6  2000/04/12 22:56:00  sparker
// Added IntVector (a vector of you-guess-what)
// Added explicit ctors from point to vector and vice-versa
//
// Revision 1.5  2000/01/26 01:32:52  sparker
// Added new stuff for C-SAFE
//
// Revision 1.4  1999/10/07 02:07:57  sparker
// use standard iostreams and complex type
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
