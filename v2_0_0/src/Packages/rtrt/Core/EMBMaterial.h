
#ifndef EMBMATERIAL_H
#define EMBMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Background.h>
#include <Core/Persistent/Persistent.h>

namespace rtrt {
class EMBMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::EMBMaterial*&);
}

namespace rtrt {

class EMBMaterial : public Material, public EnvironmentMapBackground
{

protected:

public:

  EMBMaterial(const string& filename) 
    : Material(), EnvironmentMapBackground((char*)filename.c_str()) {}
  virtual ~EMBMaterial() {}

  EMBMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, EMBMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
                     const HitInfo& hit, int depth, 
                     double atten, const Color& accumcolor,
                     Context* cx)
  {
    color_in_direction(ray.direction(),result);
  }    
};

} // end namespace

#endif






