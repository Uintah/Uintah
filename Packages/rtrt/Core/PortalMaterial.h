
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
  bool      attached_;

 public:

  PortalMaterial() : valid_(false), attached_(false) 
    { portal_.load_identity(); }
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
    if (attached_ && (u>.02 && u<.98) && (v>.02 && v<.98)) {
      Ray pray(portal_.project(hitpos), portal_.project(ray.direction()));
      
      cx->worker->traceRay(result, pray, depth+1,  atten,
                           accumcolor, cx);
    } else {
      result = Color(.1,.1,.65);
    }
  }

  Transform *get_portal() { return &portal_; }
  bool valid() { return valid_; }
  bool attached() { return attached_; }
  void attached(bool a) { attached_ = a; }

  Point project(const Point &p) 
  {
    return Point(portal_.project(p));
  }

  Vector project(const Vector &v)
  {
    return Vector(portal_.project(v));
  }

  void print() { portal_.print(); }

  void set(const Point &a, const Vector &au, const Vector &av)
  {
    portal_.load_identity();
    Vector aw(Cross(au,av));
    portal_.load_basis(a,au,av,aw/aw.length());
    valid_ = true;
    attached_ = false;
  }

  void attach(PortalMaterial *mat)
  {
    if (mat->valid()) {
      Transform temp(portal_);
      Transform *other_end = mat->get_portal();
      portal_.change_basis(*other_end);
      other_end->change_basis(temp);

      portal_.print();
      other_end->print();
      
      mat->attached(true);
      attached_ = true;
    }
  }
};

} // end namespace

#endif



