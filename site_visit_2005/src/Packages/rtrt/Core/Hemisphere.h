
#ifndef HEMISPHERE_H
#define HEMISPHERE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <stdlib.h>

namespace rtrt {
class Hemisphere;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Hemisphere*&);
}

namespace rtrt {

class Hemisphere : public Object {
protected:
  Point cen;
  Vector orientation;
  double radius;
public:

  Hemisphere(Material* matl, const Point& cen, const Vector& orientation, double radius);
  virtual ~Hemisphere();

  Hemisphere() : Object(0) {} // for Pio.
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Hemisphere*&);
  
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
  void updatePosition( const Point & pos );
  void updateOrientation( const Vector & orientation );
};

} // end namespace rtrt

#endif
