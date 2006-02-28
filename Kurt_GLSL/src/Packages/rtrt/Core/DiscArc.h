
#ifndef UVDISC_H
#define UVDISC_H 1

#include <Packages/rtrt/Core/Disc.h>

namespace rtrt {

class DiscArc : public Disc {
protected:
  double theta0, theta1;
public:
  virtual ~DiscArc();
  DiscArc(Material* matl, const Point& cen, const Vector& n, double radius);
  inline void set_arc(double t0, double t1) { theta0=t0; theta1=t1; }
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
};

} // end namespace rtrt

#endif
