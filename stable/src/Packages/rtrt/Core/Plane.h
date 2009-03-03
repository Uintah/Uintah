/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
