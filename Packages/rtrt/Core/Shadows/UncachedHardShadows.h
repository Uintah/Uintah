

#ifndef UNCACHEDHARDSHADOWS_H
#define UNCACHEDHARDSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
  class UncachedHardShadows : public ShadowBase {
  public:
    UncachedHardShadows();
    virtual ~UncachedHardShadows();
    virtual bool lit(const Point& hitpos, Light* light,
		     const Vector& light_dir, double dist, Color& atten,
		     int depth, Context* cx);
  };
}

#endif
