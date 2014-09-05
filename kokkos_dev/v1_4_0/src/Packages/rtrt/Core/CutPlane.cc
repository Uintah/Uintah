
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <iostream>

using namespace rtrt;

CutPlane::CutPlane(Object* child, const Point& cen, const Vector& n)
    : Object(0), child(child), cen(cen), n(n), dpy(0)
{
    this->n.normalize();
    d=Dot(this->n, cen);
}

CutPlane::CutPlane(Object* child, PlaneDpy* dpy)
    : Object(0), child(child), dpy(dpy)
{
    n=dpy->n;
    d=dpy->d;
}

CutPlane::~CutPlane()
{
}

void CutPlane::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext* ppc)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    HitInfo newhit;
    double plane=Dot(n, orig)-d;
    if(plane > 0){
	child->intersect(ray, newhit, st, ppc);
	// On near side of plane...
	if(t < 0 || (newhit.was_hit && newhit.min_t < t)){
	    hit=newhit;
	}
    } else {
	// On far side of plane...
	if(t<=0)
	    return;
	Point p(orig+dir*t);
	Ray newray(p, dir);
	child->intersect(newray, newhit, st, ppc);
	if(newhit.was_hit){
	    hit=newhit;
	    hit.min_t+=t;
	}
    }
}

Vector CutPlane::normal(const Point&, const HitInfo&)
{
    cerr << "Error: Group normal should not be called!\n";
    return Vector(0,0,0);
}

void CutPlane::light_intersect(Light* light, const Ray& ray,
			   HitInfo& hit, double dist, Color& atten,
			   DepthStats* st, PerProcessorContext* ppc)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    HitInfo newhit;
    Color newatten(1,1,1);
    Point p(orig+dir*t);
    double plane=Dot(n, orig)-d;
    if(plane > 0){
	child->light_intersect(light, ray, newhit, dist, newatten, st, ppc);
	// On near side of plane...
	if(t<0 || (newhit.was_hit && newhit.min_t < t)){
	    hit=newhit;
	    atten=newatten;
	}
    } else {
	if(t<0)
	    return;
	// On far side of plane...
	Ray newray(p, dir);
	child->light_intersect(light, newray, newhit, dist-t, newatten, st, ppc);
	if(newhit.was_hit){
	    hit=newhit;
	    hit.min_t+=t;
	    atten=newatten;
	}
    }
}

void CutPlane::compute_bounds(BBox& bbox, double offset)
{
    child->compute_bounds(bbox, offset);
}

void CutPlane::preprocess(double radius, int& pp_offset, int& scratchsize)
{
    child->preprocess(radius, pp_offset, scratchsize);
}

void CutPlane::animate(double t, bool& changed)
{
    if(dpy){
	if(dpy->n != n || dpy->d != d){
	    changed=true;
	    n=dpy->n;
	    d=dpy->d;
	}
    }
    child->animate(t, changed);
}
