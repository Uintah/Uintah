#ifndef DUMMYOBJECT_H
#define DUMMYOBJECT_H

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace rtrt { 

using SCIRun::Vector;
using SCIRun::Point;

class DummyObject : public Object
{
  Object* obj;
  Vector n;

public:
    
  DummyObject(Object* obj, Material* m) 
    : Object(m), obj(obj)
  {}
  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
  void SetNormal(Vector& v)
  {
    n = v;
  }
  void SetObject(Object* o)
  {
    obj=o;
  }
  virtual void compute_bounds(BBox& bbox, double offset)
  {
    obj->compute_bounds(bbox, offset);
  }
    
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* ppc)
  {
    obj->intersect(ray, hit, st, ppc);
  }
    
  virtual void light_intersect(Ray& ray,
			       HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc)
  {
    obj->light_intersect(ray, hit, atten, st, ppc);
  }
    
  virtual Vector normal(const Point&, const HitInfo& )
  {
    return n;
  }
    
};
} // end namespace rtrt

#endif

    
