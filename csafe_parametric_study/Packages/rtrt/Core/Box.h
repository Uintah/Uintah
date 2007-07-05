
#ifndef BOX_H
#define BOX_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {
class Box;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Box*&);
}

namespace rtrt {

class Box : public Object {
protected:
  Point min, max;
public:
  Box(Material* matl, const Point& min, const Point& max);
  virtual ~Box();

  Box() : Object(0) {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Box*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
};

} // end namespace rtrt

#endif
