#ifndef SCI_Wangxl_Datatypes_Point3_h
#define SCI_Wangxl_Datatypes_Point3_h

#include <Core/Geometry/Point.h>

namespace Wangxl {

using namespace SCIRun;

class Point3 : public Point {
public:
  Point3() : Point() {}
  Point3( double x, double y, double z ) : Point ( x, y, z ) {}
  ~Point3() {}
  void set( int i, double val ) { if ( i == 0 ) x(val); else if ( i == 1 ) y(val); else if ( i ==2 ) z(val); }
 double operator [] ( const int i ) const { if ( i == 0 ) return x(); else if ( i == 1 ) return y(); else return z(); }
 Point3& operator = ( const Point3& p ) {// copy constructor
    if ( &p != this ) {
      x(p.x() );
      y(p.y() );
      z(p.z() );
    }
    return *this;
  }
 Point3& operator = ( const Point& p ) {// copy constructor
   x(p.x() );
   y(p.y() );
   z(p.z() );
   return *this;
 }
};

}

#endif
