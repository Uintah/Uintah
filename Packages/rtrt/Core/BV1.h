
#ifndef BV1_H
#define BV1_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {

struct BV1Tree;
struct BoundedObject;

class BV1 : public Object {
    Object* obj;

    BV1Tree* normal_tree;
    BV1Tree* light_tree;

    BV1Tree* make_tree(double maxradius);
    void make_tree(int nprims, Object** prims, double* slabs);
    void finishit(double* slabs, Array1<Object*>& prims, int primStart);
public:
    BV1(Object* obj);
    virtual ~BV1();
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
    virtual void animate(double t, bool& changed);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_bounds(BBox&, double offset);
    virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif
