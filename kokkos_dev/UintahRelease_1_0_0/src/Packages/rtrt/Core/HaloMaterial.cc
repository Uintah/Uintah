#include <Packages/rtrt/Core/HaloMaterial.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* haloMaterial_maker() {
  return new HaloMaterial();
}

// initialize the static member type_id
PersistentTypeID HaloMaterial::type_id("HaloMaterial", "Material", 
				       haloMaterial_maker);

const int HALOMATERIAL_VERSION = 1;

void 
HaloMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("HaloMaterial", HALOMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, transparent_);
  SCIRun::Pio(str, fg_);
  SCIRun::Pio(str, pow_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::HaloMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::HaloMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::HaloMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
