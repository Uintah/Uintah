
#ifndef Cylinder_H
#define Cylinder_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Transform.h>
#include <stdlib.h>

namespace rtrt {

class Cylinder : public Object {
protected:
    Point top;
    Point bottom;
    double radius;
    Transform xform;
    Transform ixform;
public:
    Cylinder(Material* matl, const Point& bottom, const Point& top, double radius);
    virtual ~Cylinder();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void print(ostream& out);
};

} // end namespace rtrt

#endif
