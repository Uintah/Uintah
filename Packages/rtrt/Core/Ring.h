
#ifndef RING_H
#define RING_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>

namespace rtrt {

class Ring : public Object {
    Point cen;
    Vector n;
    double d;
    double radius;
    double thickness;
public:
    Ring(Material* matl, const Point& cen, const Vector& n, double radius,
	 double thickness);
    virtual ~Ring();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
  virtual void light_intersect(const Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
};

} // end namespace rtrt

#endif
