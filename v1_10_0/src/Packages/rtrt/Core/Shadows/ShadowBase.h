
#ifndef SHADOWBASE_H
#define SHADOWBASE_H

#include <Core/Persistent/Persistent.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace rtrt {

using SCIRun::Point;
using SCIRun::Vector;

class Light;
class Color;
class Context;
class Scene;
class ShadowBase;


enum ShadowType { No_Shadows = 0, Single_Soft_Shadow, Hard_Shadows,
		  Glass_Shadows, Soft_Shadows, Uncached_Shadows };
}

namespace SCIRun {
void Pio(Piostream&, rtrt::ShadowBase*&);
}

namespace rtrt {

class ShadowBase : public SCIRun::Persistent {
  const char* name;
public:
  ShadowBase();
  virtual ~ShadowBase();

  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::ShadowBase*&);
  
  virtual void preprocess(Scene* scene, int& pp_offset, int& scratchsize);
  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx) = 0;
  void setName(const char* name) {
    this->name=name;
  }
  const char* getName() {
    return name;
  }

  static char * shadowTypeNames[];

};

} // end namespace rtrt

#endif

