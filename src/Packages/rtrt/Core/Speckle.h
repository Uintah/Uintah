
#ifndef SPECKLE_H
#define SPECKLE_H 1

#include "Material.h"
#include "Color.h"
#include "Vector.h"
#include "FastTurbulence.h"

namespace rtrt {

class Speckle : public Material {
    double scale;
    Color c1, c2;
    FastTurbulence turbulence;
public:
    Speckle(double scale,
               const Color&  c1,
               const Color&  c2);
    virtual ~Speckle();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth, 
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
