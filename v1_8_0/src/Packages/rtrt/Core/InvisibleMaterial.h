
#ifndef Invisible_H
#define Invisible_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
class InvisibleMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::InvisibleMaterial*&);
}

namespace rtrt {

class InvisibleMaterial : public Material {
public:
  InvisibleMaterial();
  virtual ~InvisibleMaterial();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, InvisibleMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
