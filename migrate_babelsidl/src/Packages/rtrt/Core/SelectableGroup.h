
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
  int gui_child; // which child the gui wants to see next
  int internal_child; // Maintains the current child position
                      // including all the stuff for the repeat_last
                      // and min_child and max_child offsets.

  bool autoswitch; //should animate automatically switch showing child?
  bool no_skip; // This ignores autoswitch_secs and does the next object
  bool frame_for_frame; // This will do a simple cycling

  int repeat_last; // Number of times to repeat the last timestep
                   // (defaults to 1).
  int min_child; // First timestep to render (defaults to 0).
  int max_child; // Last timestep to render (defaults to objs.size()-1).
public:
  float autoswitch_secs; // Number of second autoswitch should dwell on
                         // each child.

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

  inline void SetFrameForFrame(bool b) { frame_for_frame = b; }
  inline int GetFrameForFrame() { return frame_for_frame? 1: 0; }
  inline void toggleFrameForFrame() { frame_for_frame = !frame_for_frame; }
  
  inline int GetChild() { return gui_child; }
  // Remember that setting child less than 0 will not select any child.
  void SetChild(int i);
  void nextChild();

  // More GUI interfaces
  inline void SetRepeatLast(int rl) { if (rl > 0) repeat_last = rl; }
  inline int  GetRepeatLast() { return repeat_last; }

  // This will clamp the assignment to [0..max_child]
  inline void SetMinChild(int mc) {
    if (mc < 0) min_child = 0;
    else if (mc > max_child) min_child = max_child;
    else min_child = mc;
  }
  inline int  GetMinChild() { return min_child; }

  // This will clamp the assignment to [min_child..objs.size()-1].
  inline void SetMaxChild(int mc) {
    if (mc < min_child) max_child = min_child;
    else if (mc >= objs.size()) max_child = objs.size()-1;
    else max_child = mc;
  }
  inline int  GetMaxChild() { return max_child; }
};

} // end namespace rtrt

#endif
