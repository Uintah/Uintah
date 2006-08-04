
#ifndef CHECKER_H
#define CHECKER_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class Checker;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Checker*&);
}

namespace rtrt {

class Checker : public Material {
  Material* matl0;
  Material* matl1;
  Vector u,v;
public:
  Checker(Material* matl0, Material* matl1, const Vector& u, const Vector& v);
  virtual ~Checker();

  Checker() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Checker*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
