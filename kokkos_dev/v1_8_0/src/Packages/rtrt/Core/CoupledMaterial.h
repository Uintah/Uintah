#ifndef COUPLED_H
#define COUPLED_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {

class CoupledMaterial : public Material {
  Color Rd;   // diffuse reflectance
  double R0;  // normal reflectance of polish
  double phong_exponent;
public:
  CoupledMaterial(const Color& Rd, double R0 = 0.05, 
		  double phong_exponent = 100);
  virtual ~CoupledMaterial();
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
