
#ifndef RING_H
#define RING_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>

namespace rtrt {
class Ring;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Ring*&);
}

namespace rtrt {

class Ring : public Object, public UVMapping {
protected:
  Point cen;
  Vector n;
  double d;
  double radius;
  double thickness;
public:
  Ring(Material* matl, const Point& cen, const Vector& n, double radius,
       double thickness);
  virtual ~Ring();
    
  Ring() : Object(0) {} // for Pio.

  virtual void uv(UV& uv, const Point&, const HitInfo& hit);

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Ring*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
};

} // end namespace rtrt

#endif
