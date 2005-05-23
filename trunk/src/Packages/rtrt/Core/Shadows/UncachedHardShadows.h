

#ifndef UNCACHEDHARDSHADOWS_H
#define UNCACHEDHARDSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
class UncachedHardShadows;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::UncachedHardShadows*&);
}

namespace rtrt {
class UncachedHardShadows : public ShadowBase {
public:
  UncachedHardShadows();
  virtual ~UncachedHardShadows();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, UncachedHardShadows*&);

  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx);
};
}

#endif
