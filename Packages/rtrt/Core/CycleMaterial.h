#ifndef CYCLE_H
#define CYCLE_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

class CycleMaterial : public Material {
protected:
  int current;
public:
  Array1<Material *> members;
public:
  CycleMaterial();
  virtual ~CycleMaterial();
  void next();
  void prev();
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
