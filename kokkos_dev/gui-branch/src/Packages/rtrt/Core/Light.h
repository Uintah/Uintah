
#ifndef LIGHT_H
#define LIGHT_H 1

#include "Point.h"
#include "Color.h"
#include "Array1.h"

namespace rtrt {

class Light {
protected:
    Point pos;
    Color color;
    Array1<Vector> beamdirs;
public:
    double radius;
    Light(const Point&, const Color&, double radius);
    inline const Point& get_pos() const {
	return pos;
    }
    inline const Color& get_color() const {
	return color;
    }
    inline Array1<Vector>& get_beamdirs() {
	return beamdirs;
    }
};


} // end namespace rtrt

#endif
