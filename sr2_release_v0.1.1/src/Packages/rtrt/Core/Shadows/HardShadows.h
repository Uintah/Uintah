

#ifndef HARDSHADOWS_H
#define HARDSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
class HardShadows;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::HardShadows*&);
}

namespace rtrt {
class HardShadows : public ShadowBase {
  int shadow_cache_offset;
public:
  HardShadows();
  virtual ~HardShadows();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, HardShadows*&);

  virtual void preprocess(Scene* scene, int& pp_offset, int& scratchsize);
  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx);
};
}

#endif
