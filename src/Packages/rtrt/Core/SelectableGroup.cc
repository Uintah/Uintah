
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/HitInfo.h>

#include <sci_values.h>

// #include <sgi_stl_warnings_off.h>
// #include <iostream>
// #include <sgi_stl_warnings_on.h>

using namespace rtrt;
//using namespace std;

SelectableGroup::SelectableGroup(float secs)
  : Group(),
    child(0), gui_child(0),
    autoswitch(true), no_skip(false), autoswitch_secs(secs)
{
}

SelectableGroup::~SelectableGroup()
{
}

//override most of Groups methods to only look at the selected child
void SelectableGroup::light_intersect(Ray& ray, HitInfo& hit,
			    Color& atten, DepthStats* st,
			    PerProcessorContext* ppc)
{
  if (objs.size() > 0) {
    objs[child]->light_intersect(ray, hit, atten, st, ppc);
  }
}

void SelectableGroup::softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				 double dist, Color& atten, DepthStats* st,
				 PerProcessorContext* ppc)
{
  if (objs.size() > 0) {
    objs[child]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
  }
}

void
SelectableGroup::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext* ppc)
{
  if (objs.size() > 0) {
    objs[child]->intersect(ray, hit, st, ppc);
  }
}

void
SelectableGroup::multi_light_intersect(Light* light, const Point& orig,
				       const Array1<Vector>& dirs,
				       const Array1<Color>& attens,
				       double dist,
				       DepthStats* st,
				       PerProcessorContext* ppc)
{
  if (objs.size() > 0) {
    objs[child]->multi_light_intersect(light, orig, dirs, attens,dist,st,ppc);
  }
}

void
SelectableGroup::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
  Group::preprocess(maxradius, pp_offset, scratchsize);
  if (objs.size() == 0) {
    gui_child = child = -1;
  }
}

void
SelectableGroup::animate(double t, bool& changed)
{
  // This is the state when there are not objects to animate.
  if (objs.size() < 0 || child < 0) return;

  //  cerr << "SelectableGroup::animate: gui_child = "<<gui_child<<", child = "<<child;
  // If gui_child doesn't match child, take that and return.
  if (gui_child != child) {
    child = gui_child;
    changed = true;
    objs[child]->animate(t, changed);
    return;
  }

  // At this point gui_child and child should be equal.  We could be
  // tempted to use gui_child to do operations on, but since we access
  // it several times, we don't want inconsistencies.  Therefore we
  // will use child.  We will then overwrite whatever is in gui_child
  // with child, so any changes made during this block of code to
  // gui_chage will be ignored.

  // Automatic cycling of child based on the clock passed in with t
  if (autoswitch) {
    int old_child = child;
    int sec = (int)(t/autoswitch_secs);
    int new_child = sec%objs.size();
    // Should probably watch for changes and then pass back changed
    if (new_child != old_child) {
      changed = true;
      // child has changed, force it to be the next child
      if (no_skip) {
	gui_child = child = (old_child+1)%objs.size();
      } else {
        gui_child = child = new_child;
      }
    }
  }

  objs[child]->animate(t, changed);
  //  cerr << " --> "<<gui_child << ", "<<child<<"\n";
}

void
SelectableGroup::collect_prims(Array1<Object*>& prims)
{
  // Do not let the acceleration structure go under this,
  // or everything will show
  prims.add(this);
}

Object *
SelectableGroup::getCurrentChild()
{
  // Since we have maintained that child is a valid index into objs, we
  // only need to check it for validity.
  if( child >= 0 )
    return objs[child];
  else
    return NULL;
}

void
SelectableGroup::SetChild(int i) {
  //  cerr << "SelectableGroup::SetChild: gui_child = "<<gui_child;
  if (i < objs.size() && i >= 0)
    gui_child = i;
  //  cerr << " --> "<<gui_child << "\n";
}

void
SelectableGroup::nextChild() {
  //  cerr << "SelectableGroup::nextChild: gui_child = "<<gui_child;
  gui_child = (gui_child+1) % objs.size();
  //  cerr << " --> "<<gui_child << "\n";
}
