
#ifndef SLICE_H
#define SLICE_H 1

#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {

class PlaneDpy;

template<class T, class A, class B>
class Slice : public VolumeBase, public Material {
protected:
  PlaneDpy* pdpy;
  Point min;
  Vector datadiag;
  Vector sdiag;
  Vector isdiag;
  int nx,ny,nz;
  A blockdata;
  T datamin, datamax;
  double d;
  Vector n;
  Array1<Material*> matls;
public:
  Slice(VolumeDpy* dpy, PlaneDpy* plane, HVolume<T,A,B>* share);
  virtual ~Slice();
  virtual void io(SCIRun::Piostream &stream);
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_hist(int nhist, int* hist,
			    float datamin, float datamax);
  virtual void get_minmax(float& min, float& max);
  virtual void animate(double t, bool& changed);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
