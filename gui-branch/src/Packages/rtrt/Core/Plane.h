#ifndef PLANE_H
#define PLANE_H 1

#include "Point.h"

namespace rtrt {

class Plane {
public:
    Point inplane;
    Vector normal;
    inline Plane(const Point& p, const Vector& v);
    inline double distance( const Point& p );
    inline double scaled_distance( const Point& p );
    inline double cos_angle( const Vector& v ) const;
};



inline Plane::Plane(const Point& p, const Vector& v) {
     inplane = p;
     normal = v;
     normal.normalize();
}

inline double Plane::distance( const Point& p ) {
    return fabs( normal.dot( p-inplane ) ) / normal.length() ; 
}

inline double Plane::scaled_distance( const Point& p ) {
    return fabs( normal.dot( p-inplane ) );
}

#if 0
inline double Plane::cos_angle( const Vector& v ) {
    return normal.dot( v ) / (normal.length()*v.length())  ;
}
#else
inline double Plane::cos_angle( const Vector& v ) const {
    return normal.dot( v );
}
#endif

} // end namespace rtrt

#endif
