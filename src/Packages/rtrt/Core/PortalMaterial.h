
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
  bool      valid_;

 public:

  PortalMaterial() : valid_(false) { portal_.load_identity(); }
  virtual ~PortalMaterial() {}

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx)
  {
    UVMapping* map=hit.hit_obj->get_uvmapping();
    UV uv;
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    map->uv(uv, hitpos, hit);
    Color diffuse;
    double u=uv.u();
    double v=uv.v();
    if (valid_ && (u>.02 && u<.98) && (v>.02 && v<.98)) {
      Ray pray(portal_.project(hitpos), portal_.project(ray.direction()));
      
      cx->worker->traceRay(result, pray, depth+1,  atten,
                           accumcolor, cx);
    } else {
      result = Color(.6,0,.6);
    }
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
    //portal_.rotate(au,bu);
    //portal_.rotate(av,bv);
    portal_.pre_translate(b-a);
    portal_.pre_scale( Vector(au.length()/bu.length(), 
                              av.length()/bv.length(), 
                              1) );
    valid_ = true;
    portal_.print();
  }
};

} // end namespace

#endif
