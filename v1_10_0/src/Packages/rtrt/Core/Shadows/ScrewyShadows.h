

#ifndef SCREWYSHADOWS_H
#define SCREWYSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
class ScrewyShadows;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::ScrewyShadows*&);
}

namespace rtrt {
class ScrewyShadows : public ShadowBase {
  int shadow_cache_offset;
public:
  ScrewyShadows();
  virtual ~ScrewyShadows();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, ScrewyShadows*&);

  virtual void preprocess(Scene* scene, int& pp_offset, int& scratchsize);
  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx);
};
}

#endif
