#include <Packages/rtrt/Core/SubMaterial.h>
#include <Core/Persistent/PersistentSTL.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* subMaterial_maker() {
  return new SubMaterial();
}

// initialize the static member type_id
PersistentTypeID SubMaterial::type_id("SubMaterial", "Material", 
				      subMaterial_maker);

const int SUBMATERIAL_VERSION = 1;

void 
SubMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("SubMaterial", SUBMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, materials_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::SubMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::SubMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::SubMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
