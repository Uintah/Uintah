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

/*
  Note: There is no check to make sure that members contains 1 or more
  materials.  This is a silly waste of time, because it should
  dump core the first time it tries to access members[0].  Besides
  it doesn't make sense to have less than 1 material anyway.
  -- bigler
*/
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
