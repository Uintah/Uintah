//========================================================================
//
// Filename: SHLambertianMaterial.cc
//
//
// Material used for rendering irradiance environment maps with 
// spherical harmonic coefficients.
//
//
//
//
// Reference: This is an implementation of the method described by
//            Ravi Ramamoorthi and Pat Hanrahan in their SIGGRAPH 2001 
//            paper, "An Efficient Representation for Irradiance
//            Environment Maps".
//
//========================================================================

#ifndef SHLAMBERTIANMATERIAL_H
#define SHLAMBERTIANMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {

class SHLambertianMaterial : public Material {

public:

  SHLambertianMaterial( const Color& R, char* envmap, float scale = 10.0, 
			int type = 1 );
  virtual ~SHLambertianMaterial( void );
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade( Color& result, const Ray& ray,
		      const HitInfo& hit, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx );
    
private:

  Color irradCoeffs( const Vector& N ) const;

  Color albedo;
  Color L00, L1_1, L10, L11, L2_2, L2_1, L20, L21, L22;
  float fudgeFactor;

};

} // end namespace rtrt

#endif
