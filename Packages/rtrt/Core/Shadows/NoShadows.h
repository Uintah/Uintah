

#ifndef NOSHADOWS_H
#define NOSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
  class NoShadows : public ShadowBase {
  public:
    NoShadows();
    virtual ~NoShadows();
    virtual bool lit(const Point& hitpos, Light* light,
		     const Vector& light_dir, double dist, Color& atten,
		     int depth, Context* cx);
  };
}

#endif
