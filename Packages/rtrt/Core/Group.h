
#ifndef GROUP_H
#define GROUP_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Geometry/Transform.h>

namespace rtrt {
  class Group;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Group*&);
}

namespace rtrt {

class Group : public Object {
public:
  bool was_processed;
  BBox bbox;
  bool all_children_are_groups;
  Array1<Object*> objs;
  Group();
  virtual ~Group();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Group*&);

  virtual Vector normal(const Point&, const HitInfo& hit);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);

  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);
  void add(Object* obj);
  int add2(Object* obj);
  void remove2(int idx);
  void prime(int n);
  int numObjects() { return objs.size(); }
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void collect_prims(Array1<Object*>& prims);
  void transform(Transform &);

  virtual bool interior_value(double& ret_val, const Ray &ref, const double t);
};

} // end namespace rtrt

#endif

