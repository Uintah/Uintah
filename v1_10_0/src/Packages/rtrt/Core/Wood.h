
#ifndef WOOD_H
#define WOOD_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/FastTurbulence.h>

namespace rtrt {
class Wood;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Wood*&);
}

namespace rtrt {

class Wood : public Material {
    double ringscale;
    Color lightwood, darkwood;
    FastNoise noise;
public:
  Wood(const Color&, const Color&, const double=10);
  virtual ~Wood();

  Wood() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Wood*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
