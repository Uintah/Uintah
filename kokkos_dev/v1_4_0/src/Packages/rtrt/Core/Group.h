
#ifndef GROUP_H
#define GROUP_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

class Group : public Object {
protected:
    Array1<Object*> objs;
public:
    Group();
    virtual ~Group();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    virtual void multi_light_intersect(Light* light, const Point& orig,
				       const Array1<Vector>& dirs,
				       const Array1<Color>& attens,
				       double dist,
				       DepthStats* st, PerProcessorContext* ppc);
    void add(Object* obj);
    void prime(int n);
    int numObjects() { return objs.size(); }
    virtual void animate(double t, bool& changed);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_bounds(BBox&, double offset);
    virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif
