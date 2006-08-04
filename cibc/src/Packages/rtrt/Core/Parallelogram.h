
#ifndef PARALLELOGRAM_H
#define PARALLELOGRAM_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Point.h>

namespace rtrt {
class Parallelogram;
}
namespace SCIRun {
void Pio(Piostream&, rtrt::Parallelogram*&);
}

namespace rtrt {

class Parallelogram : public Object, public UVMapping {
protected:
  Point anchor;
  Vector u,v;
  Vector n;
  double d;
  double d1;
  double d2;
  Vector un, vn;
  double du, dv;
public:
  Parallelogram() : Object(0), UVMapping() {}
  Parallelogram(Material* matl, const Point& cen, const Vector& u, 
		const Vector& v);
  virtual ~Parallelogram();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Parallelogram*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void uv(UV& uv, const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);

  Point  get_anchor() { return anchor; }
  Vector get_u() { return u; }
  Vector get_v() { return v; }
};

} // end namespace rtrt

#endif
