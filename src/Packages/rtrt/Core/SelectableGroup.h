
#ifndef SELECTABLEGROUP_H
#define SELECTABLEGROUP_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Group.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array1.h>

/*
Added for the hologram room demo.
We need an object which can switch from time to time, into something else.
*/

namespace rtrt {

class SelectableGroup : public Group {
  int child; //which sub is currently showing
  bool autoswitch; //should animate automatically switch showing child?
  float autoswitch_secs; //how many second should autoswitch dwell on each child?
public:

  SelectableGroup(float secs=1.0);
  virtual ~SelectableGroup();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				    double dist, Color& atten, DepthStats* st,
				    PerProcessorContext* ppc);
  virtual void light_intersect(Ray& ray,
			       HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext*);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);
  virtual void collect_prims(Array1<Object*>& prims);
  virtual void animate(double t, bool& changed);


  inline void Child(int i) { if (i<objs.size()) child = i; /*neg means none*/ };
  inline void nextChild() { 
    autoswitch = false; 
    child = child + 1; 
    child = child % objs.size();
  };
  inline void Autoswitch(bool b) {autoswitch = b;};
  inline int Autoswitch() {if (autoswitch) return 1; else return 0;};
  inline void toggleAutoswitch() {autoswitch = !autoswitch;};
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize) {
    Group::preprocess(maxradius, pp_offset, scratchsize);
    if (objs.size() == 0)
      child = -1;
  }

};

} // end namespace rtrt

#endif
