#ifndef __RTRT_VOLUMEVIS_H__
#define __RTRT_VOLUMEVIS_H__

#include "Object.h"
#include "Material.h"
#include "Point.h"
#include "BrickArray3.h"
#include <stdlib.h>

namespace rtrt {

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

struct Voxel {
  float val;
  //  unsigned short gradient_index;
};

class VolumeVis : public Object, public Material {
protected:
  Vector diag;
  BrickArray3<Voxel> data;
  float data_min, data_max, data_diff_inv;
  int nx, ny, nz;
  Point min, max;
  Material** matls;
  int nmatls;
  float *alphas;
  int nalphas;
  float delta_x2, delta_y2, delta_z2;
  int bound(const int val, const int min, const int max);
  Vector gradient(const int x, const int y, const int z);
  Vector compute_gradient(const int x, const int y, const int z);
  unsigned short get_index(const Vector &v);
  Vector get_vector(const unsigned short index);
  
public:
  VolumeVis(BrickArray3<Voxel>& data, float data_min, float data_max,
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
