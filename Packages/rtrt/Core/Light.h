
#ifndef LIGHT_H
#define LIGHT_H 1

#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

  using namespace SCIRun;
  
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
