
#ifndef LIGHTOBJECT_H
#define LIGHTOBJECT_H 1

/*
 * This is a light which when viewed will use the internal geometry to
 * view it rather than a sphere.
 */

#include <Packages/rtrt/Core/Light.h>

namespace rtrt {

class LightObject : public Light
{

 protected:

  Object *light_geom;

 public:

  LightObject(Object *obj, const Point &p, const Color &c, double r)
    : Light(p,c,r), light_geom(obj)
  {
  }
  virtual ~LightObject() {}

  virtual Object *getSphere() const { return light_geom; }
};

} // end namespace

#endif
