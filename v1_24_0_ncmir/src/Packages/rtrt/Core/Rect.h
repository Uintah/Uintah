
#ifndef RECT_H
#define RECT_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Point.h>

namespace rtrt {
class Rect;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Rect*&);
}

namespace rtrt {

class Rect : public Object, public UVMapping {
  Point cen;
  Vector u,v;
  Vector n;
  double d;
  double d1;
  double d2;
  Vector un, vn;
  double du, dv;
  Vector tex_scale;
public:
  Rect(Material* matl, const Point& cen, const Vector& u, const Vector& v);
  virtual ~Rect();
  Rect() : Object(0), UVMapping() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Rect*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void uv(UV& uv, const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  void set_tex_scale(const Vector &v) { tex_scale = v; }
};

} // end namespace rtrt

#endif
