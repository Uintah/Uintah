#ifndef __HVOLUMEMATHERIAL_H__
#define __HVOLUMEMATHERIAL_H__

#include "Color.h"
#include "Array1.h"
#include "Material.h"
#include "ScalarTransform1D.h"

namespace rtrt {

  struct Context;
  class Point;
  class HitInfo;
  class Scene;
  class Ray;
  class Stats;
  class Vector;
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
  
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};
  
} // end namespace rtrt

#endif // __HVOLUMEMATHERIAL_H__
