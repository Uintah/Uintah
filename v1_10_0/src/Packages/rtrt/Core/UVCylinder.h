
#ifndef UVCylinder_H
#define UVCylinder_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Transform.h>
#include <stdlib.h>

namespace rtrt {
class UVCylinder;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::UVCylinder*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Transform;

class UVCylinder : public Object, public UVMapping {
protected:
  Point top;
  Point bottom;
  double radius;
  Vector tex_scale;
  Transform xform;
  Transform ixform;
public:
  UVCylinder(Material* matl, const Point& bottom, const Point& top, 
	     double radius);
  UVCylinder() : Object(0), UVMapping() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, UVCylinder*&);

  virtual ~UVCylinder();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
  virtual void uv(UV&, const Point&, const HitInfo&);
  void set_tex_scale(const Vector &v) { tex_scale = v; }
};

} // end namespace rtrt

#endif
