#ifndef __HVOLUMEMATHERIAL_H__
#define __HVOLUMEMATHERIAL_H__

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace rtrt {

  struct Context;
  class HitInfo;
  class Scene;
  class Ray;
  class Stats;
  class Worker;
  class VolumeDpy;

class HVolumeMaterial: public Material {
  VolumeDpy *vdpy;
  ScalarTransform1D<float,float> *f1_to_f2;
  ScalarTransform1D<float,Material*> *f2_to_material;
  
public:
  HVolumeMaterial(VolumeDpy *dpy, ScalarTransform1D<float,float> *f1_to_f2,
		  ScalarTransform1D<float,Material*> *f2_to_material);
  virtual ~HVolumeMaterial() {}
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};
  
} // end namespace rtrt

#endif // __HVOLUMEMATHERIAL_H__
