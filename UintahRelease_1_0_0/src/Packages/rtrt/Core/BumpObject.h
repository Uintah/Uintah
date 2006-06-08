
#ifndef BumpObject_H
#define BumpObject_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {

class BumpObject : public Object {
protected:
  Vector norm;
public:
  BumpObject(Vector &v, UVMapping *uv);
  ~BumpObject(); 
  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);

  virtual Vector normal(const Point&, const HitInfo&) {return norm;};
  virtual void compute_bounds(BBox&, double) { };
  virtual void print(ostream&) { };
};

} // end namespace rtrt

#endif
