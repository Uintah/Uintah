
#ifndef PHONG_H
#define PHONG_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
class Phong;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Phong*&);
}

namespace rtrt {

class Phong : public Material {
  Color diffuse;
  Color specular;
  double refl;
  int specpow;
public:
  Phong(const Color& diffuse,
	const Color& specular, int specpow, double refl=0);
  virtual ~Phong();

  Phong() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Phong*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
