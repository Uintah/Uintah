#ifndef PERLINBUMPMATERIAL_H
#define PERLINBUMPMATERIAL_H

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/SolidNoise3.h>
#include <Packages/rtrt/Core/BubbleMap.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Context.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>

namespace rtrt { 

class PerlinBumpMaterial : public Material
{
public:

  SolidNoise3 noise;
  BubbleMap bubble;
  Material* m;
    
  PerlinBumpMaterial(Material* m): m(m) {}
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
