
#ifndef RAY_H
#define RAY_H 1

#include "Point.h"
#include "Vector.h"

namespace rtrt {

class Ray {
    Point o;
    Vector d;
public:
    inline Ray() {
    }
    inline Ray(const Point& o, const Vector& d) : o(o), d(d)
    {
    }
	
    inline Point origin() const {
	return o;
    }
    inline Vector direction() const {
	return d;
    }
    inline void set_origin(const Point& p) {
	o=p;
    }
    inline void set_direction(const Vector& v) {
	d=v;
    }
};

} // end namespace rtrt

#endif
