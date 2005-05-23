
#ifndef CELLGROUP_H
#define CELLGROUP_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Group.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array1.h>



namespace rtrt {

class CellGroup : public Object {
  Array1<BBox> bboxes;
  Array1<Object *> bbox_objs;
  Array1<Object *> non_bbox_objs;
public:
  void add_bbox_obj(Object *o, BBox b) { bboxes.add(b); bbox_objs.add(o); }
  void add_bbox_obj(Object *o) { BBox b; o->compute_bounds(b, 0); bboxes.add(b); bbox_objs.add(o); }
  void add_non_bbox_obj(Object *o) { non_bbox_objs.add(o); }
  CellGroup();
  virtual ~CellGroup();

  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }

  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray,
			       HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext*);
  virtual void softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				    double dist, Color& atten, DepthStats* st,
				    PerProcessorContext* ppc);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);
  virtual void collect_prims(Array1<Object*>& prims);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual bool interior_value( double& value, const Ray &ray, const double t);
};

} // end namespace rtrt

#endif
