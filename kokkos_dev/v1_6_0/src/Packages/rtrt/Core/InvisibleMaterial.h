
#ifndef Invisible_H
#define Invisible_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {

class InvisibleMaterial : public Material {
public:
    InvisibleMaterial();
    virtual ~InvisibleMaterial();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
