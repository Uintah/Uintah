/*
  Has a child whose intersection is computed when the value associated
  with the object has a higher value than the global threshold.

  Author: James Bigler (bigler@cs.utah.edu)
  Date:   July 16, 2002

*/

#ifndef GLYPH_H
#define GLYPH_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {
class Glyph;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Glyph*&);
}

namespace rtrt {

class Glyph : public Object {
protected:
  Object *child;
  float value;
public:
  Glyph(Object *obj, const float value);
  virtual ~Glyph();

  Glyph() : Object(0) {} // for Pio.
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::Sphere*&);
  
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* cx);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);

  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
};

} // end namespace rtrt

#endif
