
#ifndef RECT_H
#define RECT_H 1

#include "Object.h"
#include "UVMapping.h"
#include "Point.h"

namespace rtrt {

class Rect : public Object, public UVMapping {
    Point cen;
    Vector u,v;
    Vector n;
    double d;
    double d1;
    double d2;
    Vector un, vn;
    double du, dv;
public:
    Rect(Material* matl, const Point& cen, const Vector& u, const Vector& v);
    virtual ~Rect();
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
