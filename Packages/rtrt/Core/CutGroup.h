
#ifndef CUTGROUP_H
#define CUTGROUP_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Group.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>

//This is a replacement for the buggy CutPlane object.

//It is a group so that related objects, like the parts of the visible woman, can be
//treated together.
//Multiple cutting planes in a scene are fine, multiple cutting planes on the same
//object are not allowed.


//see also, CutVolumeDpy, CutMaterial, CutPlaneDpy, and ColorMap.

namespace rtrt {

#define CUTGROUPDIST 48
#define CUTGROUPPTR 56
//Note, this uses 
//8 bytes at HitInfo.scratchpad+CUTGROUPDIST for CutPlane distance and
//4 bytes at HitInfo.scratchpad+CUTGROUPPTR for CutGroup detection
//If your scene uses them for something else and also has cutgroups in it there will be
//conflicts.

class CutGroup : public Group {
    Vector n;
    double d;
    CutPlaneDpy* dpy;
    bool on;
public:
    CutGroup(const Vector& n, const Point& cen);
    CutGroup(CutPlaneDpy *dpy);

    virtual ~CutGroup();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void sub_intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void light_intersect(const Ray& ray, HitInfo& hit, Color& atten,
				 DepthStats* st, PerProcessorContext* ppc);
    virtual void multi_light_intersect(Light* light, const Point& orig,
				       const Array1<Vector>& dirs,
				       const Array1<Color>& attens,
				       double dist,
				       DepthStats* st, PerProcessorContext* ppc);
    virtual void animate(double t, bool& changed);
    virtual void collect_prims(Array1<Object*>& prims);
    bool interior_value(double& ret_val, const Ray &ref, const double t);
};

} // end namespace rtrt

#endif
