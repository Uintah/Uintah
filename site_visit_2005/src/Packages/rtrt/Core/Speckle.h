
#ifndef SPECKLE_H
#define SPECKLE_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/FastTurbulence.h>

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
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
