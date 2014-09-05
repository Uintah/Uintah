
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <values.h>

using namespace rtrt;

SelectableGroup::SelectableGroup(float secs)
  : Group(), autoswitch_secs(secs), autoswitch(true)
{
  child = 0;
}

SelectableGroup::~SelectableGroup()
{
}

//override most of Groups methods to only look at the selected child
void SelectableGroup::light_intersect(Ray& ray, HitInfo& hit,
			    Color& atten, DepthStats* st,
			    PerProcessorContext* ppc)
{
  //double min_t = hit.min_t;
  //if (!bbox.intersect(ray, min_t)) return;

  if (child >=0) objs[child]->light_intersect(ray, hit, atten, st, ppc);
}

void SelectableGroup::softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				 double dist, Color& atten, DepthStats* st,
				 PerProcessorContext* ppc)
{
  if (child >=0) objs[child]->light_intersect(ray, hit, atten, st, ppc);
}

void SelectableGroup::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
  //double min_t = hit.min_t;
  //if (!bbox.intersect(ray, min_t)) return;
  if (child >=0) objs[child]->intersect(ray, hit, st, ppc);
}

void SelectableGroup::multi_light_intersect(Light* light, const Point& orig,
					    const Array1<Vector>& dirs,
					    const Array1<Color>& attens,
					    double dist,
					    DepthStats* st, PerProcessorContext* ppc)
{
  if (child >=0) objs[child]->multi_light_intersect(light, orig, dirs, attens,
						    dist, st, ppc);
}

void SelectableGroup::animate(double t, bool& changed)
{
  //automatic cycling of child based on the clock passed in with t
  if (autoswitch) {
    int sec = (int)(t/autoswitch_secs);
    int ochild = child;
    child = sec%objs.size(); //should probably watch for changes and then pass back changed
    if ((child-ochild)) changed = true;
  }

  //animate all of them even if they aren't showing
  for(int i=0;i<objs.size();i++){
    objs[i]->animate(t, changed);
  }
}

void SelectableGroup::collect_prims(Array1<Object*>& prims)
{
  //do not let the acceleration structure go under this, or everything will show
  prims.add(this);
}



