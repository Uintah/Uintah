#ifndef __RTRT_VOLUMEVIS_H__
#define __RTRT_VOLUMEVIS_H__

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Point.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <stdlib.h>

namespace rtrt {

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

class VolumeVis : public Object, public Material {
protected:
  Vector diag;
  Vector inv_diag;
  BrickArray3<float> data;
  //ScalarTransform1D<float,Material*> *matl_transform;
  float data_min, data_max, data_diff_inv;
  int nx, ny, nz;
  Point min, max;
  Array1<Color*> matls;
  int nmatls;
  double spec_coeff, ambient, diffuse, specular;
  Array1<float> alphas;
  int nalphas;
  //  ScalarTransform1D<float,float> *alpha_transform;
  float delta_x2, delta_y2, delta_z2;
  int bound(const int val, const int min, const int max);
  Color color(const Vector &N, const Vector &V, const Vector &L, 
	      const Color &object_color, const Color &light_color);
public:
  VolumeVis(BrickArray3<float>& data, float data_min, float data_max,
	    int nx, int ny, int nz,
	    Point min, Point max, const Array1<Color*> &matls, int nmatls,
	    const Array1<float> &alphas, int nalphas, double spec_coeff,
	    double ambient, double diffuse, double specular);
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
