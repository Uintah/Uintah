
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

class SeaLambertianMaterial : public Material {
  Color R;
  TimeVaryingCheapCaustics *caustics;
public:
  SeaLambertianMaterial(const Color& R, TimeVaryingCheapCaustics *caustics);
  virtual ~SeaLambertianMaterial();

  SeaLambertianMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, SeaLambertianMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
