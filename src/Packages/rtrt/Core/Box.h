
#ifndef BOX_H
#define BOX_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {

class Box : public Object {
protected:
  Point min, max;
public:
    Box(Material* matl, const Point& min, const Point& max);
    virtual ~Box();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void print(ostream& out);
};

} // end namespace rtrt

#endif
