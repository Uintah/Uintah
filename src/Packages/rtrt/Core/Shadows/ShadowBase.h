
#ifndef SHADOWBASE_H
#define SHADOWBASE_H

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

enum ShadowType { No_Shadows = 0, Single_Soft_Shadow, Hard_Shadows,
		  Glass_Shadows, Soft_Shadows, Uncached_Shadows };

class ShadowBase {
  const char* name;
public:
  ShadowBase();
  virtual ~ShadowBase();

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

