
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <iostream>
#include <values.h>

using namespace rtrt;
using namespace SCIRun;

CutGroup::CutGroup(const Vector& n, const Point& cen)
    : Group(), n(n), dpy(0)
{
    this->n.normalize();
    d=Dot(this->n, cen);
    on = true;
    this->set_matl(this);
    animdpy = false;
}

CutGroup::CutGroup(CutPlaneDpy* dpy, bool animdpy)
  : Group(), dpy(dpy), animdpy(animdpy)
{
    n=dpy->n;
    d=dpy->d;
    on = true;
    this->set_matl(this);
}

CutGroup::~CutGroup()
{
}

void 
CutGroup::io(SCIRun::Piostream &str)
{
  ASSERTFAIL("Pio for CutGroup not implemented");
}

void CutGroup::softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				 double dist, Color& atten, DepthStats* st,
				 PerProcessorContext* ppc)
{
  light_intersect(ray, hit, atten, st, ppc);
}

void CutGroup::light_intersect(Ray& ray, HitInfo& hit,
			    Color& atten, DepthStats* st,
			    PerProcessorContext* ppc)
{
  //note: the light_intersect methods of some objects (ex Spheres) return no t info,
  //so there is no way to use them to check for shadows that may be cut
  //so I resort to using intersect, and setting Atten to Color(0,0,0)
  //this may have implications for soft shadows

  //a note about lighting, when you cut into an object, the cutting plane does not block
  //light rays, so shading on the non cut side can change
  //there seems to be no way to (in all cases) cast shadow rays within the object
  //differently than from other objects





  //see intersect for an idea of what is going on here
  double min_t = MAXDOUBLE;
  if (!bbox.intersect(ray, min_t)) return;

  if (on) {
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    Point p(orig+dir*t);

    HitInfo newhit;
    double plane=Dot(n, orig)-d;
    if(plane > 0){
	// On near side of plane...
	for(int i=0;i<objs.size();i++){
	  objs[i]->intersect(ray, newhit, st, ppc);
	}
	if (newhit.was_hit && ( t<0 || ((newhit.min_t < t) && (newhit.min_t < hit.min_t)))) {
	  hit = newhit;
	  atten=Color(0,0,0);
	  *((double *)(hit.scratchpad+CUTGROUPDIST)) = (t>0)?t:newhit.min_t;
	  *((Object **)(hit.scratchpad+CUTGROUPPTR)) = hit.hit_obj;
	  Point p = ray.origin() + ray.direction()*hit.min_t;
	  *(Vector*)hit.scratchpad = hit.hit_obj->normal(p,hit);
	  hit.hit_obj = this;
	}
    } else {
	// On far side of plane...
	if(t<=0)
	    return;
	Ray newray(p, dir);
	for(int i=0;i<objs.size();i++){
	  objs[i]->intersect(newray, newhit, st, ppc);
	}
	if (newhit.was_hit && ((newhit.min_t + t) < hit.min_t)) {
	  hit=newhit;
	  hit.min_t+=t;
	  atten = Color(0,0,0);
	  *((double *)(hit.scratchpad+CUTGROUPDIST)) = t;
	  *((Object **)(hit.scratchpad+CUTGROUPPTR)) = hit.hit_obj;
	  Point p = ray.origin() + ray.direction()*hit.min_t;
	  *(Vector*)hit.scratchpad = hit.hit_obj->normal(p,hit);
	  hit.hit_obj = this;
	}
    }
  } else {
    for(int i=0;i<objs.size();i++){
      objs[i]->light_intersect(ray, hit, atten, st, ppc);
    }
  }
}

void CutGroup::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
  //basic idea, check if we hit the cuttplane first. If we do check the interiors.
  double min_t = MAXDOUBLE;
  if (!bbox.intersect(ray, min_t)) return;

  if (on) {
    //how far to the plane?
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
      return;
    double t=(d-Dot(n, orig))/dt;
    Point p(orig+dir*t); //this is the insection with the plane
    
    //create a temporary record in case we only hit a child beyond the plane
    HitInfo newhit;
    double plane=Dot(n, orig)-d;
    if(plane > 0){
      // On near side of plane

      //what might we hit?
      for(int i=0;i<objs.size();i++){
	objs[i]->intersect(ray, newhit, st, ppc);
      }
      
      //are they beyond the plane?
      if (newhit.was_hit && ( t<0 || ((newhit.min_t < t) && (newhit.min_t < hit.min_t)))) {
	hit=newhit;
	//store this object, and the plane distance for CutMaterial's use
	//"(t>0)?t:newhit.min_t;" necessary when eye is between plane and obj
	*((double *)(hit.scratchpad+CUTGROUPDIST)) = (t>0)?t:newhit.min_t; 
	// you could probably do nested cut planes by prioritizing the t's here
	// you might need to store a min and max t to do it
	*((Object **)(hit.scratchpad+CUTGROUPPTR)) = hit.hit_obj;
	Point p = ray.origin() + ray.direction()*hit.min_t;
	*(Vector*)hit.scratchpad = hit.hit_obj->normal(p,hit);
	hit.hit_obj = this;
      }

    } else {
      // On far side of plane... (possibly looking inside cut objects)

      if(t<=0)
	return;

      //create a new ray that starts at the plane
      Ray newray(p, dir);
      
      //can we hit anything from there?
      for(int i=0;i<objs.size();i++){
	objs[i]->intersect(newray, newhit, st, ppc);
      }
      //if so is it closer to us than anything previous?
      if (newhit.was_hit && ((newhit.min_t + t) < hit.min_t)) {
	hit=newhit;
	hit.min_t+=t;
	//store this object, and the plane distance for CutMaterial's use
	*((double *)(hit.scratchpad+CUTGROUPDIST)) = t;
	*((Object **)(hit.scratchpad+CUTGROUPPTR)) = hit.hit_obj;
	Point p = ray.origin() + ray.direction()*hit.min_t;
	*(Vector*)hit.scratchpad = hit.hit_obj->normal(p,hit);	
	hit.hit_obj = this;
      }

    }
  } else {
    //cutt plane is turned off process as normal
    Object *ohit = hit.hit_obj;
    for(int i=0;i<objs.size();i++){
      objs[i]->intersect(ray, hit, st, ppc);
    }
    //you still need to store this info, the cutmaterial will need it
    if (hit.hit_obj != ohit) {
      *((double *)(hit.scratchpad+CUTGROUPDIST)) = hit.min_t;
      *((Object **)(hit.scratchpad+CUTGROUPPTR)) = hit.hit_obj;
      Point p = ray.origin() + ray.direction()*hit.min_t;
      *(Vector*)hit.scratchpad = hit.hit_obj->normal(p,hit);
      hit.hit_obj = this;
    }
  }
}

void CutGroup::sub_intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			     PerProcessorContext* ppc)
{
  //used by CutMaterial to find insidedness
  //we don't want that the back ray to escape through the cutting plane
  for(int i=0;i<objs.size();i++){
    objs[i]->intersect(ray, hit, st, ppc);
  }
}

void CutGroup::multi_light_intersect(Light* light, const Point& orig,
				  const Array1<Vector>& dirs,
				  const Array1<Color>& attens,
				  double dist,
				  DepthStats* st, PerProcessorContext* ppc)
{
  //This should be optimized. 
  if (on) {
    HitInfo dummy;
    for(int j = 0; j<dirs.size(); j++) {
      Ray jray(orig, dirs[j]);
      light_intersect( 
		      jray, 
		      dummy,
		      attens[j],
		      st, 
		      ppc);
    }
  } else {
    for(int i=0;i<objs.size();i++){
      objs[i]->multi_light_intersect(light, orig, dirs, attens,
				     dist, st, ppc);
    }
  }
}


void CutGroup::animate(double t, bool& changed)
{
  //receive updates from the PlaneDpy
  if(dpy){
    if (animdpy) dpy->animate(t);

    if(dpy->n != n || dpy->d != d || dpy->on != on){
      changed=true;
      n=dpy->n;
      d=dpy->d;
      on=dpy->on;
    }
  }
  //animate each child
  for(int i=0;i<objs.size();i++){
    objs[i]->animate(t, changed);
  }
}

void CutGroup::collect_prims(Array1<Object*>& prims)
{
  //do not let the acceleration structure go under this, or there will be no cutting
  prims.add(this);
}

bool CutGroup::interior_value(double& ret_val, const Ray &ref, const double t)
{
  //to get the correct interior from a collection of objects (like vfem)
  //you have to try all of them, and let each determine if it is the real holder
  //otherwise you will pick the wrong one when looking through on item into another
  for(int i=0;i<objs.size();i++){
    if (objs[i]->interior_value(ret_val, ref, t)) return true;
  }
  return false;
}



void CutGroup::shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx) {

  //cerr << "USING CG AS A MAT" << endl;
  Object *o = *((Object **)(hit.scratchpad+CUTGROUPPTR));
  Material *mat = o->get_matl();
  mat->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

Vector CutGroup::normal(const Point& p, const HitInfo& hit)
{
  Vector* n=(Vector*)hit.scratchpad;
  return *n;  
}


