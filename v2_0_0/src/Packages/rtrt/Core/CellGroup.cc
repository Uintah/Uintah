#include <Packages/rtrt/Core/CellGroup.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <values.h>

using namespace rtrt;

CellGroup::CellGroup()
  : Object(0)
{
}

CellGroup::~CellGroup()
{
}

void CellGroup::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
  int b_idx=-1;
  int i;
  int was_already_hit=hit.was_hit;
  double min_t = hit.min_t;
  for (i=0; i<bboxes.size(); i++)
    if (bboxes[i].contains_point(ray.origin()))
      b_idx=i;
  if (b_idx!=-1) {
    bbox_objs[b_idx]->intersect(ray, hit, st, ppc);
    if (hit.was_hit && (!was_already_hit || hit.min_t<min_t)) return;
  }
  for (i=0; i<bbox_objs.size(); i++)
    if (i != b_idx)
      bbox_objs[i]->intersect(ray, hit, st, ppc);
  for (i=0; i<non_bbox_objs.size(); i++)
    non_bbox_objs[i]->intersect(ray, hit, st, ppc);
}

void CellGroup::light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			    DepthStats* st, PerProcessorContext* ppc)
{
  int b_idx=-1;
  int i;
  for (i=0; i<bboxes.size(); i++)
    if (bboxes[i].contains_point(ray.origin()))
      b_idx=i;
  if (b_idx!=-1) {
    bbox_objs[b_idx]->light_intersect(ray, hit, atten, st, ppc);
    if (hit.was_hit) return;
  }
  for (i=0; i<bbox_objs.size(); i++)
    if (i != b_idx) {
      bbox_objs[i]->light_intersect(ray, hit, atten, st, ppc);
      if (hit.was_hit) return;
    }
  for (i=0; i<non_bbox_objs.size(); i++) {
    non_bbox_objs[i]->light_intersect(ray, hit, atten, st, ppc);
    if (hit.was_hit) return;
  }
}

void CellGroup::softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				 double dist, Color& atten, DepthStats* st,
				 PerProcessorContext* ppc)
{
  int b_idx=-1;
  int i;
  for (i=0; i<bboxes.size(); i++)
    if (bboxes[i].contains_point(ray.origin()))
      b_idx=i;
  if (b_idx!=-1) {
    bbox_objs[b_idx]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
    if (hit.was_hit) return;
  }
  for (i=0; i<bbox_objs.size(); i++)
    if (i != b_idx) {
      bbox_objs[i]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
      if (hit.was_hit) return;
    }
  for (i=0; i<non_bbox_objs.size(); i++) {
    non_bbox_objs[i]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
    if (hit.was_hit) return;
  }
}


void CellGroup::multi_light_intersect(Light* light, const Point& orig,
				  const Array1<Vector>& dirs,
				  const Array1<Color>& attens,
				  double dist,
				  DepthStats* st, PerProcessorContext* ppc)
{
  int i;
  for (i=0; i<bbox_objs.size(); i++)
    bbox_objs[i]->multi_light_intersect(light, orig, dirs, attens, dist, st, ppc);
  for (i=0; i<non_bbox_objs.size(); i++)
    non_bbox_objs[i]->multi_light_intersect(light, orig, dirs, attens, dist, st, ppc);
}

Vector CellGroup::normal(const Point&, const HitInfo&)
{
    cerr << "Error: Group normal should not be called!\n";
    return Vector(0,0,0);
}

void CellGroup::animate(double t, bool& changed)
{
  int i;
  for (i=0; i<bbox_objs.size(); i++)
    bbox_objs[i]->animate(t, changed);
  for (i=0; i<non_bbox_objs.size(); i++)
    non_bbox_objs[i]->animate(t, changed);
}

void CellGroup::collect_prims(Array1<Object*>& prims)
{
  int i;
  for (i=0; i<bbox_objs.size(); i++)
    bbox_objs[i]->collect_prims(prims);
  for (i=0; i<non_bbox_objs.size(); i++)
    non_bbox_objs[i]->collect_prims(prims);
}

void CellGroup::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
  int i;
  for(i=0;i<bbox_objs.size();i++) {
    bbox_objs[i]->preprocess(maxradius, pp_offset, scratchsize);
  }
  for(i=0;i<non_bbox_objs.size();i++) {
    non_bbox_objs[i]->preprocess(maxradius, pp_offset, scratchsize);
  }
}

void CellGroup::compute_bounds(BBox& bb, double offset)
{
  int i;
  for(i=0;i<bbox_objs.size();i++)
    bbox_objs[i]->compute_bounds(bb, offset);
  for(i=0;i<non_bbox_objs.size();i++)
    non_bbox_objs[i]->compute_bounds(bb, offset);
}

bool CellGroup::interior_value(double& ret_val, const Ray &ref, 
				   const double t)
{
  int i;
  for(i=0;i<bbox_objs.size();i++)
    if (bbox_objs[i]->interior_value(ret_val, ref, t)) 
      return true;
  for(i=0;i<non_bbox_objs.size();i++)
    if (non_bbox_objs[i]->interior_value(ret_val, ref, t)) 
      return true;
  return false;
}
