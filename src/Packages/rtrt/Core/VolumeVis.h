#ifndef __RTRT_VOLUMEVIS_H__
#define __RTRT_VOLUMEVIS_H__

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Point.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <stdlib.h>

namespace rtrt {

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

class VolumeVis : public Object, public Material {
protected:
  Vector diag;
  BrickArray3<float> data;
  //ScalarTransform1D<float,Material*> *matl_transform;
  float data_min, data_max, data_diff_inv;
  int nx, ny, nz;
  Point min, max;
  Material** matls;
  int nmatls;
  float *alphas;
  int nalphas;
  //  ScalarTransform1D<float,float> *alpha_transform;
  float delta_x2, delta_y2, delta_z2;
  int bound(const int val, const int min, const int max);
  
public:
  VolumeVis(BrickArray3<float>& data, float data_min, float data_max,
	    int nx, int ny, int nz,
	    Point min, Point max, Material** matls, int nmatls,
	    float *alphas, int nalphas);
  virtual ~VolumeVis();
  virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Light* light, const Ray& lightray,
			       HitInfo& hit, double dist, Color& atten,
			       DepthStats* st, PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};
  
} // end namespace rtrt

#endif
