

#ifndef SINGLESAMPLESOFTSHADOWS_H
#define SINGLESAMPLESOFTSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
class SingleSampleSoftShadows;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::SingleSampleSoftShadows*&);
}

namespace rtrt {
class SingleSampleSoftShadows : public ShadowBase {
  int shadow_cache_offset;
public:
  SingleSampleSoftShadows();
  virtual ~SingleSampleSoftShadows();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, SingleSampleSoftShadows*&);

  virtual void preprocess(Scene* scene, int& pp_offset, int& scratchsize);
  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx);
};
}

#endif
