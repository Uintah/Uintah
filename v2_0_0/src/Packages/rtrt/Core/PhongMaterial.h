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
public:
  PhongMaterial(const Color& Rd, double opacity, double Rphong = 0.0, 
		double phong_exponent = 100);
  inline Color get_diffuse() { return Rd; }
  inline void set_diffuse(const Color &d) { Rd = d; }
  inline double get_opacity() { return opacity; }
  inline void set_opacity(double o) { opacity = o; }
  inline double get_reflectance() { return Rphong; }
  inline void set_reflectance(double r) { Rphong = r; }
  inline double get_shininess() { return phong_exponent; }
  inline void set_shininess(double s) { phong_exponent = s; }
  virtual ~PhongMaterial();

  PhongMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, PhongMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
