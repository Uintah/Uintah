
#ifndef PARALLELOGRAM_H
#define PARALLELOGRAM_H 1

#include "Object.h"
#include "UVMapping.h"
#include "Point.h"

namespace rtrt {

class Parallelogram : public Object, public UVMapping {
    Point anchor;
    Vector u,v;
    Vector n;
    double d;
    double d1;
    double d2;
    Vector un, vn;
    double du, dv;
public:
    Parallelogram(Material* matl, const Point& cen, const Vector& u, const Vector& v);
    virtual ~Parallelogram();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void uv(UV& uv, const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
};

} // end namespace rtrt

#endif
