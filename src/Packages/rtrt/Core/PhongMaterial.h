#ifndef PHONGMATERIAL_H
#define PHONGMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
  class PhongMaterial;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::PhongMaterial*&);
}

namespace rtrt {

class PhongMaterial : public Material {
  Color Rd;   // diffuse reflectance
  double opacity;  // transparancy = 1 - opacity
  double Rphong;  // phong reflectance
  double phong_exponent;
  bool reflects;
public:
  PhongMaterial(const Color& Rd, double opacity, double Rphong = 0.0, 
		double phong_exponent = 100, bool refl=false);
  virtual ~PhongMaterial();

  PhongMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, PhongMaterial*&);

  inline double get_opacity() { return opacity; }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
