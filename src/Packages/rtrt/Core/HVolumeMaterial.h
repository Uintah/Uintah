#ifndef __HVOLUMEMATHERIAL_H__
#define __HVOLUMEMATHERIAL_H__

#include "Color.h"
#include "Array1.h"
#include "Material.h"

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
  
class HVolumeMaterial: public Material {
  VolumeDpy *vdpy;
  Material** matls;
  float datamin, datamax;
  float scale;
  int size_minus_1;
  Array1<float> data;
  HVolumeTransferFunct *transfer;
  
public:
  HVolumeMaterial(VolumeDpy *dpy, Array1<float> indata, float datamin,
		  float datamax, HVolumeTransferFunct *trans);
  virtual ~HVolumeMaterial() {}
  
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  
  void get_min_max(float &in_min, float &in_max);
};
  
} // end namespace rtrt

#endif // __HVOLUMEMATHERIAL_H__
