

#ifndef NOSHADOWS_H
#define NOSHADOWS_H

#include <Packages/rtrt/Core/Shadows/ShadowBase.h>

namespace rtrt {
class NoShadows;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::NoShadows*&);
}

namespace rtrt {
class NoShadows : public ShadowBase {
public:
  NoShadows();
  virtual ~NoShadows();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, NoShadows*&);

  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx);
};
}

#endif
