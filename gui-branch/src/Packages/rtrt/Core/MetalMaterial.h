
#ifndef METAL_H
#define METAL_H 1

#include "Material.h"
#include "Color.h"

namespace rtrt {

class MetalMaterial : public Material {
    Color specular_reflectance;
    double phong_exponent;
public:
    MetalMaterial(const Color& specular_reflectance);
    MetalMaterial(const Color& specular_reflectance, double phong_exponent);
    virtual ~MetalMaterial();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth, 
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
