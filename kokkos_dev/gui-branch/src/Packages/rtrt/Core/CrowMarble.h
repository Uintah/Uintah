
#ifndef CROWMARBLE_H
#define CROWMARBLE_H 1

#include "Material.h"
#include "Color.h"
#include "Vector.h"
#include "CatmullRomSpline.h"
#include "FastTurbulence.h"

namespace rtrt {

class CrowMarble : public Material {
    double scale;
    Color c1, c2, c3;
    Vector direction;
    CatmullRomSpline<Color> spline;
    FastTurbulence turbulence;
    double phong_exponent;
    double R0;
public:
    CrowMarble(double scale,
               const Vector& direction,
               const Color&  c1,
               const Color&  c2,
               const Color&  c3,
               double R0 = 0.04,
               double phong_exponent=100);
    virtual ~CrowMarble();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth, 
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
