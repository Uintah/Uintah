
#ifndef BV2_H
#define BV2_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {

struct BV2Tree;

class BV2 : public Object {
  Object* obj;

  BV2Tree* top;
  BV2Tree* top_light;
  BV2Tree* make_tree(int nprims, Object** prims, double maxradius);
public:
  BV2(Object* obj);
  virtual ~BV2();
  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
  virtual void intersect(Ray& ray,
			 HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif
