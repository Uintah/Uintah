
#ifndef LIGHT_H
#define LIGHT_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;

class Light {
public:

  std::string name_;
  double      radius;

  Light(const Point&, const Color&, double radius, double intensity = 1.0 );

  inline const Point& get_pos() const { return pos; }
  inline const Color& get_color() const { return currentColor_; }
  inline       float  get_intensity() const { return intensity_; }
  inline Array1<Vector>& get_beamdirs() { return beamdirs; }

  void updateIntensity( double intensity );
  inline const Color& getOrigColor() const { return origColor_; }

  void setIndex(int index) { this->index=index; }
  int getIndex() const { return index; }

protected:

  // Parameters for changing light value:
  float intensity_;
  Color currentColor_; // Original Color * intensity
  Color origColor_;

  Point pos;
  Color color;
  Array1<Vector> beamdirs;
  int index;

};


} // end namespace rtrt

#endif
