
#ifndef PORTALMATERIAL_H
#define PORTALMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>

namespace rtrt {

class PortalMaterial : public Material
{

 protected:

  Transform portal_;

 public:

  PortalMaterial() { portal_.load_identity(); }
  virtual ~PortalMaterial() {}

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx)
  {
    Ray pray(portal_.project(ray.origin()+ray.direction()*hit.min_t),
             portal_.project(ray.direction()));

    cx->worker->traceRay(result, pray, depth+1,  atten,
                         accumcolor, cx);
  }

  Point project(const Point &p) 
  {
    return Point(portal_.project(p));
  }

  Vector project(const Vector &v)
  {
    return Vector(portal_.project(v));
  }

  void set(const Point &a, const Vector &au, const Vector &av,
           const Point &b, const Vector &bu, const Vector &bv)
  {
    // a is local coordinates, b is opposite end coordinates
    portal_.load_identity();
    portal_.pre_translate(a-b);
    portal_.rotate(bu, au);
    portal_.rotate(bv, av);
    portal_.pre_scale( Vector(au.length()/bu.length(), 
                              av.length()/bv.length(), 
                              1) );
  }
};

} // end namespace

#endif
