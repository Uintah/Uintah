#ifndef __HVOLUMEMATHERIAL_H__
#define __HVOLUMEMATHERIAL_H__

#define USE_STF 1

#include "Color.h"
#include "Array1.h"
#include "Material.h"
#ifdef USE_STF
#include "ScalarTransform1D.h"
#endif

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
  
#ifndef USE_STF
  class HVolumeMaterial;
  
class HVolumeTransferFunct {
  Array1<HVolumeMaterial*> colors;
  Material** matls;
  int nmatls;
  
  float datamin, datamax;
  float scale;
public:
  HVolumeTransferFunct(Material **matls, int nmatls):
    matls(matls), nmatls(nmatls)
  {}
  
  Material* index(const float val);
  
  void add(HVolumeMaterial *hvcolor);
  
  // this must be called after all the HVolumeMaterial's have been added
  // and before rendering starts.
  void compute_min_max();
};

#endif // ifndef USE_STF
  
class HVolumeMaterial: public Material {
  VolumeDpy *vdpy;
#ifdef USE_STF
  ScalarTransform1D<float,float> *f1_to_f2;
  ScalarTransform1D<float,Material*> *f2_to_material;
#else
  Material** matls;
  float datamin, datamax;
  float scale;
  int size_minus_1;
  Array1<float> data;
  HVolumeTransferFunct *transfer;
#endif  
  
public:
#ifdef USE_STF
  HVolumeMaterial(VolumeDpy *dpy, ScalarTransform1D<float,float> *f1_to_f2,
		  ScalarTransform1D<float,Material*> *f2_to_material);
#else
  HVolumeMaterial(VolumeDpy *dpy, Array1<float> indata, float datamin,
		  float datamax, HVolumeTransferFunct *trans);
#endif
  virtual ~HVolumeMaterial() {}
  
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  
#ifndef USE_STF
  void get_min_max(float &in_min, float &in_max);
#endif
};
  
} // end namespace rtrt

#endif // __HVOLUMEMATHERIAL_H__
