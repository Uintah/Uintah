
#ifndef WOOD_H
#define WOOD_H 1

#include "Material.h"
#include "Color.h"
#include "Vector.h"
#include "FastTurbulence.h"

namespace rtrt {

class Wood : public Material {
    double ringscale;
    Color lightwood, darkwood;
    FastNoise noise;
public:
  Wood(const Color&, const Color&, const double=10);
  virtual ~Wood();
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
