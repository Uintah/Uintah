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

namespace rtrt {

class Instance: public Object
{
public:
    
    InstanceWrapperObject *o;
    Transform *t;
    BBox bbox;

    Instance(InstanceWrapperObject* o, Transform* trans) 
	: Object(0), o(o), t(trans) 
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
	: Object(0), o(o), t(trans)
	{
	    if (!t->inv_valid())
		t->compute_imat();

	    bbox = b.transform(t);
	    
	}
	    
    virtual Material* get_matl(HitInfo& hit) const 
	{
	    Material **mat = (Material **)&hit.scratchpad[sizeof(Vector)];
	    return *mat;
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
	    	Vector *n = (Vector *)hit.scratchpad;
		Point p = ray.origin() + hit.min_t*ray.direction();
		*n = hit.hit_obj->normal(p,hit);
		
		Material **mat = (Material **)&hit.scratchpad[sizeof(Vector)];
		*mat = hit.hit_obj->get_matl();

		hit.hit_obj = this;

	    }
	}
    
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext* ppc)
	{
	    intersect(ray,hit,st,ppc);
	}
    
    virtual Vector normal(const Point& p, const HitInfo& hit)
	{
	    Vector* n=(Vector*)hit.scratchpad;
	    t->project_normal_inplace(*n);
	    (*n).normalize();
	    return *n;
	}

    virtual void compute_bounds(BBox& b, double offset)
	{
	    b.extend(bbox);
	}

    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize)
	{
	  o->preprocess(maxradius,pp_offset,scratchsize);
	}
};
}
#endif
