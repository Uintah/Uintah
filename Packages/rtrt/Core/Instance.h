#ifndef INSTANCE_H
#define INSTANCE_H

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>

// Known problems:
// Only translations will work because normal isn't transformed
// Scales will not work because ray parameter isn't transformed
// Why are all the virtual functions in this file?  We should make it's
//  very own .cc file
// Any nested material that might need the scratchpad will be quite broken
// Faster by calling o->obj->intersect
// Do they cast correct shadows?

namespace rtrt {

  class Instance: public Object, public Material
{
public:
  struct InstanceHit {
    Vector normal;
    Object* obj;
  };
    InstanceWrapperObject *o;
    Transform *t;
    BBox bbox;

    Instance(InstanceWrapperObject* o, Transform* trans) 
	: Object(this), o(o), t(trans) 
	{
	    if (!t->inv_valid())
		t->compute_imat();

	    o->compute_bounds(bbox,1E-5);

	    Point cmin = bbox.min(), cmax=bbox.max();

	    bbox.transform_inplace(t);

	    cmin = bbox.min();
	    cmax=bbox.max();

	}

    Instance(InstanceWrapperObject* o, Transform* trans, BBox& b) 
	: Object(this), o(o), t(trans)
	{
	    if (!t->inv_valid())
		t->compute_imat();

	    bbox = b.transform(t);
	    
	}
	    
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext* ppc)
	{

	    Ray tray;

	    ray.transform(t,tray);

	    double min_t = hit.min_t;

	    o->intersect(tray,hit,st,ppc);

	    // if the ray hit one of our objects....
	    if (min_t > hit.min_t)
	    {
	      InstanceHit* i = (InstanceHit*)(hit.scratchpad);
	      Point p = ray.origin() + hit.min_t*ray.direction();
	      i->normal = hit.hit_obj->normal(p,hit);
	      i->obj = hit.hit_obj;
	      hit.hit_obj = this;
	    }
	}
    
  virtual void light_intersect(Light* /*light*/, const Ray& ray,
			       HitInfo& hit, double /*dist*/, Color& /*atten*/,
			       DepthStats* st, PerProcessorContext* ppc)
	{
	    intersect(ray,hit,st,ppc);
	}
    
    virtual Vector normal(const Point&, const HitInfo& hit)
	{
	  InstanceHit* i = (InstanceHit*)(hit.scratchpad);
	  t->project_normal_inplace(i->normal);
	  i->normal.normalize();
	  return i->normal;
	}

  virtual void compute_bounds(BBox& b, double /*offset*/)
	{
	    b.extend(bbox);
	}

    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize)
	{
	  o->preprocess(maxradius,pp_offset,scratchsize);
	}
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx) {
      InstanceHit* i = (InstanceHit*)(hit.scratchpad);
      Material *mat = i->obj->get_matl();
      mat->shade(result, ray, hit, depth, atten, accumcolor, cx);
    }
  };
}
#endif
