
#ifndef PHONGLIGHT_H
#define PHONGLIGHT_H 1

#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Material.h>

namespace rtrt {

class PhongLight : public Light
{

 protected:
  Vector direction_;
  int pow_;
 public:

  PhongLight(const Color &c, const Point &p, double r, const Vector &direction, int pow) : Light(p,c,r), direction_(direction), pow_(pow)
  {
  }
  virtual ~PhongLight() {}
  void setDirection(const Vector &dir) { direction_ = dir; }
  virtual void updatePosition( const Point & newPos, const Vector &offset, const Vector &fwd) { direction_=fwd; Light::updatePosition(newPos, offset, fwd); }
  virtual Color get_color(const Vector &v) { return currentColor_*Max(ipow(Dot(direction_,-v),pow_),0.0); }
};

} // end namespace

#endif
