
#ifndef SEALAMBERTIAN_H
#define SEALAMBERTIAN_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/TimeVaryingCheapCaustics.h>

namespace rtrt {
class SeaLambertianMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::SeaLambertianMaterial*&);
}

namespace rtrt {

class SeaLambertianMaterial : public Material, public Object {
  Color R;
  TimeVaryingCheapCaustics *caustics;
  double currentTime;
public:
  SeaLambertianMaterial(const Color& R, TimeVaryingCheapCaustics *caustics);
  virtual ~SeaLambertianMaterial();

  SeaLambertianMaterial() : Material(), Object(this) {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, SeaLambertianMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);

  // Object.  None of these do anything, except for animate...
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void animate(double t, bool& changed);
  virtual void compute_bounds(BBox& bbox, double offset);
};

} // end namespace rtrt

#endif
