

#ifndef HARDSHADOWS_H
#define HARDSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
  class HardShadows : public ShadowBase {
    int shadow_cache_offset;
  public:
    HardShadows();
    virtual ~HardShadows();
    virtual void preprocess(Scene* scene, int& pp_offset, int& scratchsize);
    virtual bool lit(const Point& hitpos, Light* light,
		     const Vector& light_dir, double dist, Color& atten,
		     int depth, Context* cx);
  };
}

#endif
