
#ifndef DISC_H
#define DISC_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Point.h>

namespace rtrt {
  class Disc;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Disc*&);
}

namespace rtrt {

class Disc : public Object, public UVMapping {
protected:
  Point cen;
  Vector n;
  double d;
  double radius;
  Vector tex_scale;
  Transform xform;
public:
  Disc(Material* matl, const Point& cen, const Vector& n, double radius);
  virtual ~Disc();

  Disc() : Object(0) {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Disc*&);
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void uv(UV&, const Point&, const HitInfo&);
  void set_tex_scale(const Vector &v) { tex_scale = v; }
};

} // end namespace rtrt

#endif
