
#ifndef SPHERE_H
#define SPHERE_H 1

#include "Object.h"
#include "Material.h"
#include "Point.h"
#include "Array3.h"
#include <stdlib.h>

namespace rtrt {

  class VolumeVis : public Object, public Material {
  protected:
    Vector datadiag;
    Vector hierdiag;
    Vector ihierdiag;
    Vector sdiag;
    Array3<float> data;
    float data_min, data_max, data_diff;
    int nx, ny, nz;
    Point min, max;
    Material** matls;
    int nmatls;
    int bound(const int val, const int min, const int max);
      
  public:
    VolumeVis(Array3<float>& data, float data_min, float data_max,
	      int nx, int ny, int nz,
	      Point min, Point max, Material** matls, int nmatls);
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
