#ifndef CYCLE_MATERIAL_H
#define CYCLE_MATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {
  class CycleMaterial;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::CycleMaterial*&);
}

namespace rtrt {

class CycleMaterial : public Material {
protected:
  int current;
public:
  Array1<Material *> members;
public:
  CycleMaterial();
  virtual ~CycleMaterial();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, CycleMaterial*&);

  void next();
  void prev();

  inline Material *curr() { return members[current]; }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
