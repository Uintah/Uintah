
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/PlaneDpy.h>

#include <Core/Math/MiscMath.h>

#include <iostream>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

CutPlane::CutPlane(Object* child, const Point& cen, const Vector& n)
  : Object(0), child(child), cen(cen), n(n), dpy(0),
    active(true), use_material(true)
{
    this->n.normalize();
    d=Dot(this->n, cen);
}

CutPlane::CutPlane(Object* child, PlaneDpy* dpy)
  : Object(0), child(child), dpy(dpy),
    active(dpy->active), use_material(dpy->use_material)
{
    n=dpy->n;
    d=dpy->d;
}

CutPlane::~CutPlane()
{
}

void CutPlane::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext* ppc)
{
  if (active) {
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
      // perpendicular to the plane, so there are no intersections
      return;
    double t=(d-Dot(n, orig))/dt;
    HitInfo newhit;
    double plane=Dot(n, orig)-d;
    if(plane > 0){
      child->intersect(ray, newhit, st, ppc);
      // On near side of plane...so try to intersect the object first.
      // Only record a hit if the object's intersection point was before
      // the plane.
      if (newhit.was_hit && ( t<0 || ((newhit.min_t < t) && (newhit.min_t < hit.min_t)))) {
	//	if(t < 0 || (newhit.was_hit && newhit.min_t < t)){
	hit=newhit;
      } else if (use_material && (get_matl() != 0)) {
	// the plane has a material, so use it :)
	if (child_bbox.contains_point(ray, t))
	  hit.hit(this,t);
      }
    } else {
      // On far side of plane...
      if(t<=0)
	// the plane is behind the ray origin
	return;
      if (use_material && (get_matl() != 0)) {
	if (child_bbox.contains_point(ray, t)) {
	  // We hit the plane before we can get to the child
	  hit.hit(this,t);
	} else {
	  // Need to compute intersection with the child
	  Point p(orig+dir*t);
	  Ray newray(p, dir);
	  child->intersect(newray, newhit, st, ppc);
	  if(newhit.was_hit){
	    hit=newhit;
	    hit.min_t+=t;
	  }
	}
      } else {
	// Shoot the ray out
	// On far side of plane...so create a new ray that starts at the
	// location on the cut plane and send it out.
	Point p(orig+dir*t);
	Ray newray(p, dir);
	child->intersect(newray, newhit, st, ppc);
	if(newhit.was_hit){
	  hit=newhit;
	  hit.min_t+=t;
	}
      }
    }
  } else {
    // just call the intersection on the child
    child->intersect(ray, hit, st, ppc);
  }
}

Vector CutPlane::normal(const Point&, const HitInfo&)
{
  //    cerr << "Error: Group normal should not be called!\n";
  return n.normal();
}

void CutPlane::light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc)
{
  if (active) {
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
      // perpendicular to the plane, so there are no intersections
      return;
    double t=(d-Dot(n, orig))/dt;
    HitInfo newhit;
    Color newatten(1,1,1);
    Point p(orig+dir*t);
    double plane=Dot(n, orig)-d;
    if(plane > 0){
      newhit.min_t = hit.min_t;
      child->light_intersect(ray, newhit, newatten, st, ppc);
      // On near side of plane...so try to intersect the object first.
      // Only record a hit if the object's intersection point was before
      // the plane.
      if (newhit.was_hit && ( t<0 || ((newhit.min_t < t) && (newhit.min_t < hit.min_t)))) {
	//	if(t<0 || (newhit.was_hit && newhit.min_t < t)){
	hit=newhit;
	atten=newatten;
      } else if (use_material && (get_matl() != 0)) {
	// the plane has a material, so use it :)
	hit.hit(this,t);
      }
    } else {
      if(t<0)
	// the plane is behind the ray origin
	return;
      if (use_material && (get_matl() != 0)) {
	if (child_bbox.contains_point(ray, t)) {
	  // We hit the plane before we can get to the child
	  hit.hit(this,t);
	} else {
	  // Need to compute intersection with the child
	  Point p(orig+dir*t);
	  Ray newray(p, dir);
	  child->intersect(newray, newhit, st, ppc);
	  if(newhit.was_hit){
	    hit=newhit;
	    hit.min_t+=t;
	  }
	}
      } else {
	// Shoot the ray out
	// On far side of plane...so create a new ray that starts at the
	// location on the cut plane and send it out.
	Ray newray(p, dir);
	newhit.min_t = hit.min_t-dt;
	child->light_intersect(newray, newhit, newatten, st, ppc);
	if (newhit.was_hit && ((newhit.min_t + t) < hit.min_t)) {
	  //	if(newhit.was_hit){
	  hit=newhit;
	  hit.min_t+=t;
	  atten=newatten;
	}
      }
    }
  } else {
    // just call the intersection on the child
    child->intersect(ray, hit, st, ppc);
  }
}

void CutPlane::compute_bounds(BBox& bbox, double offset)
{
  child->compute_bounds(child_bbox, offset);
  bbox.extend(child_bbox);
}

void CutPlane::preprocess(double radius, int& pp_offset, int& scratchsize)
{
    child->preprocess(radius, pp_offset, scratchsize);
}

void CutPlane::animate(double t, bool& changed)
{
    if(dpy){
	if(dpy->n != n ||
	   dpy->d != d ||
	   dpy->active != active ||
	   dpy->use_material != use_material) {
	  changed = true;
	  n = dpy->n;
	  d = dpy->d;
	  active = dpy->active;
	  use_material = dpy->use_material;
	}
    }
    child->animate(t, changed);
}
