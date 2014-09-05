
#ifndef TRI_H
#define TRI_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>

namespace rtrt {

class Tri : public Object {
    Point p1, p2, p3;
    Vector n;
    double d;
    Vector e1p, e2p, e3p;
    Vector e1, e2, e3;
    double e1l, e2l, e3l;
    bool bad;
public:
    inline bool isbad() {
	return bad;
    }
    Tri(Material* matl, const Point& p1, const Point& p2, const Point& p3);
    virtual ~Tri();
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
