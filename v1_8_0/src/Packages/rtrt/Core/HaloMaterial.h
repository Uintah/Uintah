
#ifndef HALOMATERIAL_H
#define HALOMATERIAL_H 1

#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <math.h>

namespace rtrt {
class HaloMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::HaloMaterial*&);
}

namespace rtrt {

class HaloMaterial : public Material 
{

 protected:

  InvisibleMaterial transparent_;
  Material          *fg_;
  double            pow_;

 public:

  HaloMaterial(Material *fg, double pow) 
    : transparent_(), fg_(fg), pow_(pow) {}
  virtual ~HaloMaterial() {}

  HaloMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, HaloMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
                     const HitInfo& hit, int depth,
                     double atten, const Color& accumcolor,
                     Context* cx)
  {
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    double percent=-Dot(obj->normal(hitpos,hit), ray.direction());
    //if (percent<0) percent=0;
    percent = pow(percent,pow_);
    fg_->shade(result,ray,hit,depth,atten,accumcolor,cx);
    Color fg = result;
    transparent_.shade(result,ray,hit,depth,atten,accumcolor,cx);
    Color bg = result;
    result = (fg*percent)+(bg*(1.-percent));
  }
};

} // end namespace

#endif
