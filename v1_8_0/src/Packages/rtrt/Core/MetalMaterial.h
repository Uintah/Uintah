
#ifndef METAL_H
#define METAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
class MetalMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::MetalMaterial*&);
}

namespace rtrt {

class MetalMaterial : public Material {
  Color specular_reflectance;
  double phong_exponent;
public:
  MetalMaterial(const Color& specular_reflectance);
  MetalMaterial(const Color& specular_reflectance, double phong_exponent);
  virtual ~MetalMaterial();
    
  MetalMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, MetalMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
