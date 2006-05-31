
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/HitInfo.h>

#include <sci_values.h>

// #include <sgi_stl_warnings_off.h>
// #include <iostream>
// #include <sgi_stl_warnings_on.h>

using namespace rtrt;
// using namespace std;

SelectableGroup::SelectableGroup(float secs)
  : Group(),
    child(0), gui_child(0), internal_child(0),
    autoswitch(true), no_skip(false),
    repeat_last(1), min_child(0), max_child(0),
    autoswitch_secs(secs)
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
  if (objs.size() != 0) {
    // Set the max_child to be the last one
    max_child = objs.size() - 1;
  } else {
    gui_child = child = -1;
  }
}

void
SelectableGroup::animate(double t, bool& changed)
{
  // This is the state when there are not objects to animate.
  if (objs.size() < 0 || child < 0) return;

//   cerr << "SelectableGroup::animate: gui_child = "<<gui_child<<", child = "<<child<<", internal_child = "<<internal_child;

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
  // gui_child will be ignored.

  // Automatic cycling of child based on the clock passed in with t
  if (autoswitch) {
    if (!frame_for_frame) {
      int sec = (int)(t/autoswitch_secs);
      int num_frames = max_child-min_child + repeat_last;
      int new_internal_child = sec%num_frames + min_child;
      //     cerr << ", num_frames = "<<num_frames<<", new_internal_child = "<<new_internal_child;
      if (new_internal_child != internal_child) {
        changed = true;
        // child has changed, force it to be the next child in need be.
        if (no_skip) {
          internal_child++;
        } else {
          internal_child = new_internal_child;
        }
        // Now update gui_child and child
        if (internal_child > max_child) {
          // We are repeating, but should we loop yet
          if (internal_child >= max_child+repeat_last) {
            gui_child = child = internal_child = min_child;
          } else {
            gui_child = child = max_child;
          }
        } else {
          // else do nothing, because we are fine.
          gui_child = child = internal_child;
        }
      }
    } else {
      internal_child++;
      changed = true;
      // Now update gui_child and child
      if (internal_child > max_child) {
        // We are repeating, but should we loop yet
        if (internal_child >= max_child+repeat_last) {
          gui_child = child = internal_child = min_child;
        } else {
          gui_child = child = max_child;
        }
      } else {
        // else do nothing, because we are fine.
        gui_child = child = internal_child;
      }
    }
  }

  objs[child]->animate(t, changed);
//   cerr << " --> "<<gui_child << ", "<<child<<", "<<internal_child<<"\n";
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
//   cerr << "SelectableGroup::SetChild: i = "<<i<<", gui_child = "<<gui_child<<", internal_child = "<<internal_child;
  if (i < min_child) {
    gui_child = internal_child = min_child;
  } else if (i > max_child) {
    // In the repeating phase.  Figure out where.
    if (i >= max_child+repeat_last) {
      // Past repeating phase, so clamp
      internal_child = max_child+repeat_last-1;
    } else {
      // Still in repeating phase
      internal_child = i;
    }
    gui_child = max_child;
  } else {
    // Simple assignment
    gui_child = internal_child = i;
  }
//   cerr << " --> "<<gui_child <<", "<<internal_child<<"\n";
}

void
SelectableGroup::nextChild() {
//   cerr << "SelectableGroup::nextChild: gui_child = "<<gui_child<<", internal_child = "<<internal_child;
  // Increment internal_child.
  internal_child++;
  // Now check to see where internal_child ended up.
  if (internal_child > max_child) {
    // We are repeating, but should we loop yet
    if (internal_child >= max_child+repeat_last) {
      gui_child = internal_child = min_child;
    } else {
      gui_child = max_child;
    }
  } else {
    // else just assign, because we are fine.
    gui_child = internal_child;
  }
//   cerr << " --> "<<gui_child <<", "<<internal_child<<"\n";
}
