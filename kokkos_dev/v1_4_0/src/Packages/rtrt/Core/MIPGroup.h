
#ifndef MIPGROUP_H
#define MIPGROUP_H 1

#include <Packages/rtrt/Core/Group.h>

namespace rtrt {

class MIPGroup : public Group {
public:
    MIPGroup();
    virtual ~MIPGroup();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
};

} // end namespace rtrt

#endif
