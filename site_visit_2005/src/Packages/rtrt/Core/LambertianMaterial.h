
#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
class LambertianMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::LambertianMaterial*&);
}

namespace rtrt {

class LambertianMaterial : public Material {
  Color R;
public:
  LambertianMaterial(const Color& R);
  virtual ~LambertianMaterial();
  LambertianMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, LambertianMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
