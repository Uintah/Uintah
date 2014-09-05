
#ifndef CUTPLANE_H
#define CUTPLANE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BBox.h>

namespace rtrt {

class PlaneDpy;
  
class CutPlane : public Object {
    Object* child;
    Point cen;
    Vector n;
    double d;
    PlaneDpy* dpy;
  BBox child_bbox;
  bool active;
  bool use_material;
public:
    CutPlane(Object* child, const Point& cen, const Vector& n);
    CutPlane(Object* child, PlaneDpy* dpy);
    CutPlane(Object* child, const Vector& n, const double d);
    virtual ~CutPlane();
    virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void animate(double t, bool& changed);
    virtual void preprocess(double radius, int& pp_offset, int& scratchsize);
  void update_displacement(double newd) { d = newd; }
  void update_normal(const Vector &newn) { n = newn; }
  void update_active_state(const bool newstate) { active = newstate; }
  void update_usemat_state(const bool newstate) { use_material = newstate; }
};

} // end namespace rtrt

#endif
