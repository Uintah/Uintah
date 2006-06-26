
#ifndef MIPGROUP_H
#define MIPGROUP_H 1

#include <Packages/rtrt/Core/Group.h>

namespace rtrt {

class MIPGroup : public Group {
public:
  MIPGroup();
  virtual ~MIPGroup();
  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
};

} // end namespace rtrt

#endif
