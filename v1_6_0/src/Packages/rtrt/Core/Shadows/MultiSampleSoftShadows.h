

#ifndef MULTISAMPLESOFTSHADOWS_H
#define MULTISAMPLESOFTSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
class MultiSampleSoftShadows;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::MultiSampleSoftShadows*&);
}

namespace rtrt {
class MultiSampleSoftShadows : public ShadowBase {
public:
  MultiSampleSoftShadows();
  virtual ~MultiSampleSoftShadows();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, MultiSampleSoftShadows*&);

  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx);
};
}

#endif
