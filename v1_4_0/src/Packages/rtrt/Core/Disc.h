
#ifndef DISC_H
#define DISC_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>

namespace rtrt {

class Disc : public Object {
    Point cen;
    Vector n;
    double d;
    double radius;
public:
    Disc(Material* matl, const Point& cen, const Vector& n, double radius);
    virtual ~Disc();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
};

} // end namespace rtrt

#endif
