
#ifndef PHONG_H
#define PHONG_H 1

#include "Material.h"
#include "Color.h"

namespace rtrt {

class Phong : public Material {
    Color ambient;
    Color diffuse;
    Color specular;
    double specpow;
    double refl;
    double transp;
public:
    Phong(const Color& ambient, const Color& diffuse,
	  const Color& specular, double specpow, double refl=0);
    virtual ~Phong();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
