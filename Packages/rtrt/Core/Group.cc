
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <values.h>

using namespace rtrt;

Group::Group()
    : Object(0)
{
  was_processed = false;
}

Group::~Group()
{
}

void Group::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
  for(int i=0;i<objs.size();i++){
    objs[i]->intersect(ray, hit, st, ppc);
  }
}

void Group::light_intersect(const Ray& ray, HitInfo& hit, Color& atten,
			    DepthStats* st, PerProcessorContext* ppc)
{
  for(int i=0;i<objs.size();i++){
    objs[i]->light_intersect(ray, hit, atten, st, ppc);
    if(hit.was_hit)
      return;
  }
}

void Group::softshadow_intersect(Light* light, const Ray& ray, HitInfo& hit,
				 double dist, Color& atten, DepthStats* st,
				 PerProcessorContext* ppc)
{
  for(int i=0;i<objs.size();i++){
    objs[i]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
    if(hit.was_hit)
      return;
  }
}


void Group::multi_light_intersect(Light* light, const Point& orig,
				  const Array1<Vector>& dirs,
				  const Array1<Color>& attens,
				  double dist,
				  DepthStats* st, PerProcessorContext* ppc)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->multi_light_intersect(light, orig, dirs, attens,
				       dist, st, ppc);
    }
}

Vector Group::normal(const Point&, const HitInfo&)
{
    cerr << "Error: Group normal should not be called!\n";
    return Vector(0,0,0);
}


void Group::add(Object* obj)
{
    objs.add(obj);
}

void Group::animate(double t, bool& changed)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->animate(t, changed);
    }
}

void Group::collect_prims(Array1<Object*>& prims)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->collect_prims(prims);
    }
}

void Group::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
  if (!was_processed) {
    all_children_are_groups=1;
    for(int i=0;i<objs.size();i++) {
      objs[i]->preprocess(maxradius, pp_offset, scratchsize);
      objs[i]->compute_bounds(bbox, 1E-5);
      if (dynamic_cast<Group *>(objs[i]) == 0) all_children_are_groups=0;
    }
    was_processed = true;
  }
}


void Group::compute_bounds(BBox& bb, double offset)
{
  if (!was_processed)
    for(int i=0;i<objs.size();i++) {
      objs[i]->compute_bounds(bb, offset);
    }
  else
    bb.extend(bbox);
}

void Group::prime(int n)
{
    int nobjs=objs.size();
    objs.resize(n);
    objs.resize(nobjs);
}

void Group::transform(Transform& T)
{
  for (int i=0;i<objs.size();i++) {
    objs[i]->transform(T);
  }

}
