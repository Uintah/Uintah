

#ifndef MULTISAMPLESOFTSHADOWS_H
#define MULTISAMPLESOFTSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
  class MultiSampleSoftShadows : public ShadowBase {
  public:
    MultiSampleSoftShadows();
    virtual ~MultiSampleSoftShadows();
    virtual bool lit(const Point& hitpos, Light* light,
		     const Vector& light_dir, double dist, Color& atten,
		     int depth, Context* cx);
  };
}

#endif
