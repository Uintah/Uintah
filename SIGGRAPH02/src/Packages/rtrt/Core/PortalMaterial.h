
#ifndef PORTALMATERIAL_H
#define PORTALMATERIAL_H 1

#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Material.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <Core/Thread/Thread.h>

namespace rtrt {
class PortalMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::PortalMaterial*&);
}

namespace rtrt {

using SCIRun::Cross;

class PortalMaterial : public Material
{

 protected:

  // local end basis
  Point     p_;
  Vector    u_;
  Vector    v_;

  // other end basis
  Point     oe_p_;
  Vector    oe_u_;
  Vector    oe_v_;

  Transform portal_,other_end_;

  bool      attached_;

 public:

  PortalMaterial(const Point &p, const Vector &u, const Vector &v) 
    : Material(), p_(p), u_(u), v_(v), attached_(false) 
    { portal_.load_basis(p_,u_,v_,Cross(u_,v_)); }
  virtual ~PortalMaterial() {}

  PortalMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, PortalMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx)
  {
    UVMapping* map=hit.hit_obj->get_uvmapping();
    UV uv;
    // hitpos is global hit coords
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    map->uv(uv, hitpos, hit);
    Color diffuse;
    // (u,v) is local hit coords
    double u=uv.u();
    double v=uv.v();
    if (attached_ && (u>.02 && u<.98) && (v>.02 && v<.98)) {

      Point p2(oe_p_+oe_u_*u+oe_v_*v);
      
      Vector v1(portal_.unproject(ray.direction()));
      Vector v2(other_end_.project(v1));

      Ray pray(p2,v2);

      cx->worker->traceRay(result, pray, depth+1,  atten,
                           accumcolor, cx);
    } else {
      result = Color(.1,.1,.65);
    }
  }

  bool attached() { return attached_; }

  void print() { portal_.print(); }

  void attach(const Point &b, const Vector &bu, const Vector &bv)
  {
    oe_p_ = b;
    oe_u_ = bu;
    oe_v_ = bv;
    other_end_.load_basis(oe_p_,oe_u_,oe_v_,Cross(oe_u_,oe_v_));

    attached_ = true;
  }
};

} // end namespace

#endif



