#include <Packages/rtrt/Core/EMBMaterial.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* embmaterial_maker() {
  return new EMBMaterial();
}

// initialize the static member type_id
PersistentTypeID EMBMaterial::type_id("EMBMaterial", "Material", 
					   embmaterial_maker);

const int EMBMATERIAL_VERSION = 1;

void 
EMBMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("EMBMaterial", EMBMATERIAL_VERSION);
  Material::io(str);
  EnvironmentMapBackground::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::EMBMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::EMBMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::EMBMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
