
#ifndef DIELECTRIC_H
#define DIELECTRIC_H 1

#include "Material.h"
#include "Color.h"

namespace rtrt {

class DielectricMaterial : public Material {
    double R0;                 // reflectance at normal incidence
    double n_in;               // refractive index of media normal points away from
    double n_out;              // refractive index of media normal points to
    double phong_exponent;
    Color extinction_in;       // transmittance through one unit distance of material:
                               // newcolor = extinction * oldcolor
    Color extinction_out;
    Color extinction_constant_in;  // what actually gets sent to the exp function
    Color extinction_constant_out;  // what actually gets sent to the exp function
    bool nothing_inside;	// True if this object is empty - optimize the recursive hits...
public:
    DielectricMaterial(double n_in, double n_out, bool nothing_inside=false);
    DielectricMaterial(double n_in, double n_out, double R0, double phong_exponent,
                       const Color& extinction_in,  const Color& extinction_out,
		       bool nothing_inside=false);
    virtual ~DielectricMaterial();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
