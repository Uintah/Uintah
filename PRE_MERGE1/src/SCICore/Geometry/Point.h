
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

#include <share/share.h>

#include <Util/Assert.h>
#include <Tester/RigorousTest.h>
#include <Persistent/Persistent.h>
#include <Containers/String.h>

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
using SCICore::Tester::RigorousTest;

class Vector;

class SHARE Point {
    double _x,_y,_z;
#if SCI_ASSERTION_LEVEL >= 4
    int uninit;
#endif
public:
    inline Point(double x, double y, double z): _x(x), _y(y), _z(z)
#if SCI_ASSERTION_LEVEL >= 4
	, uninit(0)
#endif
	    {}
    Point(double, double, double, double);
    inline Point(const Point&);
    inline Point();
    int operator==(const Point&) const;
    int operator!=(const Point&) const;
    Point& operator=(const Point&);
    Vector operator-(const Point&) const;
    Point operator+(const Vector&) const;
    Point operator-(const Vector&) const;
    Point operator*(double) const;
    Point& operator*=(const double);
    Point& operator+=(const Vector&);
    Point& operator-=(const Vector&);
    Point& operator/=(const double);
    Point operator/(const double) const;
    Point operator-() const;
    void x(const double);
    inline double x() const;
    void y(const double);
    inline double y() const;
    void z(const double);
    inline double z() const;
    Vector vector() const;
    
    clString string() const;
    
    friend class Vector;
    friend inline double Dot(const Point&, const Point&);
    friend inline double Dot(const Vector&, const Point&);
    friend inline double Dot(const Point&, const Vector&);
//    friend inline double Dot(const Point&, const Vector&);
    friend inline Point Min(const Point&, const Point&);
    friend inline Point Max(const Point&, const Point&);
    friend Point Interpolate(const Point&, const Point&, double);
    friend Point AffineCombination(const Point&, double,
				   const Point&, double,
				   const Point&, double,
				   const Point&, double);
    friend Point AffineCombination(const Point&, double,
				   const Point&, double,
				   const Point&, double);
    friend Point AffineCombination(const Point&, double,
				   const Point&, double);
    friend void Pio( Piostream&, Point& );


    // is one point within a small interval of another?

    int Overlap( double a, double b, double e );
    int InInterval( Point a, double epsilon );
    
    static void test_rigorous(RigorousTest* __test);


};

SHARE ostream& operator<<(ostream& os, const Point& p);
SHARE istream& operator>>(istream& os, Point& p);

} // End namespace Geometry
} // End namespace SCICore

#include <Geometry/Vector.h>
#include <Math/MinMax.h>

namespace SCICore {
namespace Geometry {

#include "Point.icc"

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
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
