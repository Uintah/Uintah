#ifndef PLANE_H
#define PLANE_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>

namespace rtrt {
class Plane;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Plane*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Dot;

class Plane : public SCIRun::Persistent {
public:
  Point inplane;
  Vector normal;
  inline Plane(const Point& p, const Vector& v);
  virtual ~Plane();
  Plane() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::Plane*&);

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
  return fabs( Dot(normal, p-inplane) ) / normal.length() ; 
}

inline double Plane::scaled_distance( const Point& p ) {
  return fabs( Dot(normal, p-inplane ) );
}

#if 0
inline double Plane::cos_angle( const Vector& v ) {
  return Dot(normal, v ) / (normal.length()*v.length())  ;
}
#else
inline double Plane::cos_angle( const Vector& v ) const {
  return Dot(normal, v );
}
#endif

} // end namespace rtrt

#endif
