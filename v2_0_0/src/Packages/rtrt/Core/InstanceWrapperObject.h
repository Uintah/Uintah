#ifndef INSTANCE_WRAPPER_OBJECT_H
#define INSTANCE_WRAPPER_OBJECT_H

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>

namespace rtrt {
  class InstanceWrapperObject;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::InstanceWrapperObject*&);
}

namespace rtrt {

class InstanceWrapperObject : public SCIRun::Persistent {
  
 public:

  Object* obj;
  BBox bb;
  bool was_processed;
  bool computed_bbox;

  InstanceWrapperObject(Object* obj) :
    obj(obj) 
    {
      was_processed = false;
      computed_bbox = false;
    }

  // Force a particlular bounding box
  InstanceWrapperObject(Object* obj, BBox &b) :
    obj(obj), bb(b)
    {
      was_processed = false;
      computed_bbox = true;
    }

  InstanceWrapperObject() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, InstanceWrapperObject*&);

  void preprocess(double maxradius, int& pp_offset, int& scratchsize)
    {
      if (!was_processed) {
	obj->preprocess(maxradius,pp_offset,scratchsize);
	was_processed = true;
	if (!computed_bbox) {
	  obj->compute_bounds(bb,1E-5);
	  computed_bbox = true;
	}
      }
    }
   
  inline void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			PerProcessorContext* ppc)
    {
      obj->intersect(ray, hit, st,ppc);
    }

  void compute_bounds(BBox& bbox, double offset)
    {
      if (!computed_bbox) {
	obj->compute_bounds(bb,offset);
	computed_bbox = true;
      }
      bbox.extend(bb);
    }

  virtual void animate(double t, bool& changed) {
    obj->animate(t, changed);
  }

  bool interior_value(double& ret_val, const Ray &ref, const double t) {
    return obj->interior_value(ret_val, ref, t);
  }

};
} // end namespace rtrt
#endif

