
#ifndef TIMEOBJ_H
#define TIMEOBJ_H 1

#include "Object.h"
#include "Array1.h"

namespace rtrt {

class TimeObj : public Object {
protected:
    int cur;
    Array1<Object*> objs;
    double rate;
    int num_processors;
public:
    TimeObj(double rate);
    virtual ~TimeObj();
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
    virtual void animate(double t, bool& changed);
    void parallel_preprocess(int proc);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_bounds(BBox&, double offset);
    virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif
