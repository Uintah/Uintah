
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
  // Remember that child < 0 means that no child is selected
  int child; //which sub is currently showing
  
  bool autoswitch; //should animate automatically switch showing child?
  bool no_skip; // This ignores autoswitch_secs and does the next object
public:
  float autoswitch_secs; //how many second should autoswitch dwell on each child?

  SelectableGroup(float secs=1.0);
  virtual ~SelectableGroup();

  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }

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


  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual bool interior_value( double& value, const Ray &ray, const double t)
  { return objs[child]->interior_value(value,ray,t); }

  // Returns the currently active child object.
  Object * getCurrentChild();


  // Interfaces for gui
  inline void SetAutoswitch(bool b) {autoswitch = b;}
  inline int GetAutoswitch() {if (autoswitch) return 1; else return 0;}
  inline void toggleAutoswitch() {autoswitch = !autoswitch;}

  inline void SetNoSkip(bool b) { no_skip = b;}
  inline int GetNoSkip() { if (no_skip) return 1; else return 0; }
  inline void toggleNoSkip() {no_skip = !no_skip;}

  
  inline int GetChild() { return child; }
  // Remember that setting child less than 0 will not select any child.
  inline void SetChild(int i) { if (i<objs.size()) child = i;}
  inline void nextChild() { 
    autoswitch = false; 
    child++;
    if (child == objs.size())
      child = 0;
  };

};

} // end namespace rtrt

#endif
