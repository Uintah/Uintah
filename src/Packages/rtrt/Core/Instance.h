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
// Scales may not work because ray parameter isn't transformed --
// the direction vector scale is adjusted instead.
// Why are all the virtual functions in this file?  We should make it's
//  very own .cc file

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

	    //cerr << "B IBBMIN" << bbox.min() << endl;
	    //cerr << "B IBBMAX" << bbox.max() << endl;

	    bbox.transform_inplace(t);

	    //cerr << "A IBBMIN" << bbox.min() << endl;
	    //cerr << "A IBBMAX" << bbox.max() << endl;

	    this->set_matl(this);
	}

    Instance(InstanceWrapperObject* o, Transform* trans, BBox& b) 
	: Object(this), o(o), t(trans)
	{
	    if (!t->inv_valid())
		t->compute_imat();

	    bbox = b.transform(t);
	    
	    this->set_matl(this);
	}
	    
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext* ppc)
	{

	  double min_t = hit.min_t;
	  if (!bbox.intersect(ray, min_t)) return;	  
	  min_t = hit.min_t;

	  Ray tray;
	  HitInfo thit=hit;
	  
	  ray.transform(t,tray);

	  o->intersect(tray,thit,st,ppc);
	  
	  // if the ray hit one of our objects....
	  if (min_t > thit.min_t)
	    {
	      InstanceHit* i = (InstanceHit*)(thit.scratchpad);
	      Point p = ray.origin() + thit.min_t*ray.direction();
	      i->normal = thit.hit_obj->normal(p,thit);
	      i->obj = thit.hit_obj;
	      thit.hit_obj = this;
	      hit = thit;
	    }	      
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

    virtual void animate(double t, bool& changed) {
      o->animate(t, changed);
    }

    bool interior_value(double& ret_val, const Ray &ref, const double _t) {
      Ray tray;

      ref.transform(t,tray);

      return o->interior_value(ret_val, tray, _t);
    }

  };
}
#endif
