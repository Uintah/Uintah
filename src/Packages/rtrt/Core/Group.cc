
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <values.h>

using namespace rtrt;

Group::Group()
    : Object(0)
{
}

Group::~Group()
{
}

void Group::light_intersect(Light* light, const Ray& ray, HitInfo& hit,
			    double dist, Color& atten, DepthStats* st,
			    PerProcessorContext* ppc)
{
  double min_t = MAXDOUBLE;
  if (!bbox.intersect(ray, min_t)) return;

  int i;
  for(i=0;i<objs.size();i++){
    objs[i]->light_intersect(light, ray, hit, dist, atten, st, ppc);
  }
}

void Group::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
#if 1
  for(int i=0;i<objs.size();i++){
    objs[i]->intersect(ray, hit, st, ppc);
  }
#else
  double min_t = MAXDOUBLE;
  if (!bbox.intersect(ray, min_t)) return;
  
  int i,j;
  if (!all_children_are_groups) {
    for(i=0;i<objs.size();i++){
      objs[i]->intersect(ray, hit, st, ppc);
    }
    return;
  }

  // sort kids' bboxes so we can test intersections against the closest ones 
  //   first
  Array1<std::pair<int, double> > bbox_dist(objs.size());
  for (i=0; i<objs.size(); i++) {
    Group *g = dynamic_cast<Group*>(objs[i]);
    min_t = MAXDOUBLE;
    g->bbox.intersect(ray, min_t);
    bbox_dist[i].first = i;
    bbox_dist[i].second = min_t;
  }
  int swapi;
  double swapd;
  for (i=0; i<objs.size()-1; i++) {
    for (j=i+1; j<objs.size(); j++) {
      if (bbox_dist[j].second < bbox_dist[i].second) {
	swapi=bbox_dist[j].first;
	swapd=bbox_dist[j].second;
	bbox_dist[j].first=bbox_dist[i].first;
	bbox_dist[j].second=bbox_dist[i].second;
	bbox_dist[i].first=swapi;
	bbox_dist[i].second=swapd;
      }
    }
  }
  for (i=0; i<objs.size(); i++) {
    if (hit.min_t <= bbox_dist[i].second) return;
    objs[bbox_dist[i].first]->intersect(ray, hit, st, ppc);
  }
#endif
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
  all_children_are_groups=1;
  for(int i=0;i<objs.size();i++) {
    objs[i]->preprocess(maxradius, pp_offset, scratchsize);
    objs[i]->compute_bounds(bbox, pp_offset);
    if (dynamic_cast<Group *>(objs[i]) == 0) all_children_are_groups=0;
  }
  if (all_children_are_groups) {
    cerr << "YES!  ALL CHILDREN ARE GROUPS!\n";
  }
}


void Group::compute_bounds(BBox& bb, double offset)
{
  bb.extend(bbox);
}

void Group::prime(int n)
{
    int nobjs=objs.size();
    objs.resize(n);
    objs.resize(nobjs);
}
