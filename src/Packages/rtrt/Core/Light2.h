
#ifndef LIGHT2_H
#define LIGHT2_H 1

#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Satellite.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <Packages/rtrt/Core/Group.h>

namespace rtrt {

class Light2 : public Light
{

 protected:

  Group *light_group;

 public:

  Light2(Material *m, const Color &c, const Point &p, double r, double h=4,
         const Vector &up=Vector(0,0,1)) : Light(p,c,r)
  {
    light_group = new Group();
    LightMaterial *lm = new LightMaterial(c);
    light_group->add( new Sphere(new HaloMaterial(lm, h), p, r*1.3) );
    light_group->add( new Satellite("sun light",m,p,r,up) );
  }
  virtual ~Light2() {}

  virtual Object *getSphere() const { return light_group; }
};

} // end namespace

#endif
