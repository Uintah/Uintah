
#ifndef CHECKER_H
#define CHECKER_H 1

#include "Material.h"
#include "Color.h"
#include "Vector.h"

namespace rtrt {

class Checker : public Material {
    Material* matl0;
    Material* matl1;
    Vector u,v;
public:
    Checker(Material* matl0, Material* matl1, const Vector& u, const Vector& v);
    virtual ~Checker();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth, 
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
