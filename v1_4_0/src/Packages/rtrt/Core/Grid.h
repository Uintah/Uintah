
#ifndef Grid_H
#define Grid_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>

namespace rtrt {

struct GridTree;
struct BoundedObject;

class Grid : public Object {
    Object* obj;
    BBox bbox;
    int nx, ny, nz;
    Object** grid;
    int* counts;
    int nsides;
public:
    Grid(Object* obj, int nside);
    virtual ~Grid();
    virtual void intersect(const Ray& ray,
			   HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    void add(Object* obj);
    virtual void animate(double t, bool& changed);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_bounds(BBox&, double offset);
    virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif
