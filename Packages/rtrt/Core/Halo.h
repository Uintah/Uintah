
#ifndef HALO_H
#define HALO_H 1

#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <math.h>

namespace rtrt {

class Halo : public Material 
{

 protected:

  InvisibleMaterial transparent_;
  Material          *fg_;
  double            pow_;

 public:

  Halo(Material *fg, double pow) 
    : transparent_(), fg_(fg), pow_(pow) {}
  virtual ~Halo() {}

  virtual void shade(Color& result, const Ray& ray,
                     const HitInfo& hit, int depth,
                     double atten, const Color& accumcolor,
                     Context* cx)
  {
    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    normal.normalize();
    Vector eye(-ray.direction());
    eye.normalize();
    double percent=Dot(normal,eye);
    percent = pow(percent,pow_);
    Color original=result;
    transparent_.shade(result,ray,hit,depth,atten,accumcolor,cx);
    Color bg = result;
    result = original;
    fg_->shade(result,ray,hit,depth,atten,accumcolor,cx);
    Color fg = result;

    result = fg*percent+bg*(1-percent);
  }
};

} // end namespace

#endif
