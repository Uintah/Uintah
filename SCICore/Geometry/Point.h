
/*
 *  Point.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Point_h
#define Geometry_Point_h 1

#include <SCICore/share/share.h>
#include <SCICore/Math/MinMax.h>

#include <iosfwd>

namespace SCICore {
    namespace Containers {
	class clString;
    }
    namespace PersistentSpace {
	class Piostream;
    }
    namespace Tester {
	class RigorousTest;
    }
    
namespace Geometry {

using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::Tester::RigorousTest;

class Vector;

class SCICORESHARE Point {
    double _x,_y,_z;
public:
    inline explicit Point(const Vector& v);
    inline Point(double x, double y, double z): _x(x), _y(y), _z(z)
	    {}
    Point(double, double, double, double);
    inline Point(const Point&);
    inline Point();
    int operator==(const Point&) const;
    int operator!=(const Point&) const;
    inline Point& operator=(const Point&);
    inline Vector operator-(const Point&) const;
    inline Point operator+(const Vector&) const;
    inline Point operator-(const Vector&) const;
    inline Point operator*(double) const;
    inline Point& operator*=(const double);
    inline Point& operator+=(const Vector&);
    inline Point& operator-=(const Vector&);
    inline Point& operator/=(const double);
    inline Point operator/(const double) const;
    inline Point operator-() const;
    inline void x(const double);
    inline double x() const;
    inline void y(const double);
    inline double y() const;
    inline void z(const double);
    inline double z() const;
    inline Vector vector() const;
    inline Vector asVector() const;
    
    clString string() const;
    
    friend SCICORESHARE class Vector;
    friend SCICORESHARE inline double Dot(const Point&, const Point&);
    friend SCICORESHARE inline double Dot(const Vector&, const Point&);
    friend SCICORESHARE inline double Dot(const Point&, const Vector&);
//    friend inline double Dot(const Point&, const Vector&);
    friend SCICORESHARE inline Point Min(const Point&, const Point&);
    friend SCICORESHARE inline Point Max(const Point&, const Point&);
    friend SCICORESHARE Point Interpolate(const Point&, const Point&, double);
    friend SCICORESHARE Point AffineCombination(const Point&, double,
				   const Point&, double,
				   const Point&, double,
				   const Point&, double);
    friend SCICORESHARE Point AffineCombination(const Point&, double,
				   const Point&, double,
				   const Point&, double);
    friend SCICORESHARE Point AffineCombination(const Point&, double,
				   const Point&, double);
    friend SCICORESHARE void Pio( Piostream&, Point& );


    // is one point within a small interval of another?

    int Overlap( double a, double b, double e );
    int InInterval( Point a, double epsilon );
    
    static void test_rigorous(RigorousTest* __test);


};

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Point& p);
SCICORESHARE std::istream& operator>>(std::istream& os, Point& p);

} // End namespace Geometry
} // End namespace SCICore

// This cannot be above due to circular dependencies
#include <SCICore/Geometry/Vector.h>

namespace SCICore {
namespace Geometry {

inline Point::Point(const Vector& v)
    : _x(v._x), _y(v._y), _z(v._z)
{
}

inline Point::Point(const Point& p)
{
    _x=p._x;
    _y=p._y;
    _z=p._z;
}

inline Point::Point()
{
}

inline Point& Point::operator=(const Point& p)
{
    _x=p._x;
    _y=p._y;
    _z=p._z;
    return *this;
}

inline Vector Point::operator-(const Point& p) const
{
    return Vector(_x-p._x, _y-p._y, _z-p._z);
}

inline Point Point::operator-() const
{
    return Point(-_x, -_y, -_z);
}

inline Point Point::operator-(const Vector& v) const
{
    return Point(_x-v._x, _y-v._y, _z-v._z);
}

inline Point& Point::operator-=(const Vector& v)
{
    _x-=v._x;
    _y-=v._y;
    _z-=v._z;
    return *this;
}

inline Point Point::operator+(const Vector& v) const
{
    return Point(_x+v._x, _y+v._y, _z+v._z);
}

inline Point Point::operator*(double d) const
{
    return Point(_x*d, _y*d, _z*d);
}

inline Point& Point::operator*=(const double d)
{
    _x*=d;_y*=d;_z*=d;
    return *this;
}

inline Point& Point::operator+=(const Vector& v)
{
    _x+=v._x;
    _y+=v._y;
    _z+=v._z;
    return *this;
}

inline Point& Point::operator/=(const double d)
{
    _x/=d;
    _y/=d;
    _z/=d;
    return *this;
}

inline Point Point::operator/(const double d) const
{
    return Point(_x/d,_y/d,_z/d);
}

inline void Point::x(const double d)
{
    _x=d;
}

inline double Point::x() const
{
    return _x;
}

inline void Point::y(const double d)
{
    _y=d;
}

inline double Point::y() const
{
    return _y;
}

inline void Point::z(const double d)
{
    _z=d;
}

inline double Point::z() const
{
    return _z;
}

// THIS ONE SHOULD BE REMOVED
inline Vector Point::vector() const
{
    return Vector(_x,_y,_z);
}

inline Vector Point::asVector() const
{
    return Vector(_x,_y,_z);
}

inline Point Min(const Point& p1, const Point& p2)
{
  using SCICore::Math::Min;

  double x=Min(p1._x, p2._x);
  double y=Min(p1._y, p2._y);
  double z=Min(p1._z, p2._z);
  return Point(x,y,z);
}

inline Point Max(const Point& p1, const Point& p2)
{
  using SCICore::Math::Max;

  double x=Max(p1._x, p2._x);
  double y=Max(p1._y, p2._y);
  double z=Max(p1._z, p2._z);
  return Point(x,y,z);
}

inline double Dot(const Point& p, const Vector& v)
{
    return p._x*v._x+p._y*v._y+p._z*v._z;
}

inline double Dot(const Point& p1, const Point& p2)
{
    return p1._x*p2._x+p1._y*p2._y+p1._z*p2._z;
}

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
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
// Revision 1.4  1999/10/07 02:07:56  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/04 06:01:52  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.2  1999/08/17 06:39:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:56  mcq
// Initial commit
//
// Revision 1.5  1999/07/09 00:27:39  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.4  1999/07/07 21:10:59  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:56:16  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:18  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif //ifndef Geometry_Point_h
