#include <Packages/rtrt/Core/MapBlendMaterial.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* mapBlendMaterial_maker() {
  return new MapBlendMaterial();
}

// initialize the static member type_id
PersistentTypeID MapBlendMaterial::type_id("MapBlendMaterial", "Material", 
					   mapBlendMaterial_maker);

const int MAPBLENDMATERIAL_VERSION = 1;

void 
MapBlendMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("MapBlendMaterial", MAPBLENDMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, mat1_);
  SCIRun::Pio(str, mat2_);
  SCIRun::Pio(str, map_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::MapBlendMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::MapBlendMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MapBlendMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
