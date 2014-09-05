
#ifndef LIGHTMATERIAL_H
#define LIGHTMATERIAL_H

#include <Packages/rtrt/Core/Material.h>

namespace rtrt { 

// LightMaterials are used as the material for the sphere that is/
// rendered for the light so that it shows up brightly all the time
// instead of being colored based on the amount of light.

class LightMaterial : public Material {

public:

  LightMaterial( const Color & color );

  virtual ~LightMaterial();

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);

private:
  Color color_;

};

} // end namespace rtrt

#endif
