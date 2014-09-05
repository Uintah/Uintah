
#ifndef DIELECTRIC_H
#define DIELECTRIC_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
  class DielectricMaterial;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::DielectricMaterial*&);
}

namespace rtrt {

class DielectricMaterial : public Material {
  double R0;            // reflectance at normal incidence
  double n_in;          // refractive index of media normal points away from
  double n_out;         // refractive index of media normal points to
  int phong_exponent;
  Color extinction_in;  // transmittance through one unit distance of material:
  Color extinction_out;
  bool nothing_inside;	/* True if this object is empty - 
			 * optimize the recursive hits...*/
  double extinction_scale;    // Allow for a scale of t

  Color bg_out;         // exctinction_in to the infinite power
  Color bg_in;          // exctinction_out to the infinite power

public:
  DielectricMaterial(double n_in, double n_out, bool nothing_inside=false);
  DielectricMaterial(double n_in, double n_out, double R0, 
		     int phong_exponent, const Color& extinction_in,  
		     const Color& extinction_out,
		     bool nothing_inside=false, double extinction_scale=1);
  virtual ~DielectricMaterial();

  DielectricMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, DielectricMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
