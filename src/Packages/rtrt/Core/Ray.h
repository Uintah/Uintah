
#ifndef RAY_H
#define RAY_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {

  using namespace SCIRun;
  
class Ray {
    Point  _origin;
    Vector _direction;
public:
    inline Ray() {
    }
    inline Ray(const Point& origin, const Vector& dir) : _origin(origin), 
      _direction(dir)
    {
    }
	
    inline Point origin() const {
	return _origin;
    }
    inline Vector direction() const {
	return _direction;
    }
    inline void set_origin(const Point& p) {
	_origin = p;
    }
    inline void set_direction(const Vector& v) {
	_direction = v;
    }
};

} // end namespace rtrt

#endif
