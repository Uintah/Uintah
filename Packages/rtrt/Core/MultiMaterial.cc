#include <Packages/rtrt/Core/MultiMaterial.h>


using namespace rtrt;
using namespace SCIRun;

Persistent* multiMaterial_maker() {
  return new MultiMaterial();
}

// initialize the static member type_id
PersistentTypeID MultiMaterial::type_id("MultiMaterial", "Material", multiMaterial_maker);

const int MULTIMATERIAL_VERSION = 1;
const int MATPERCENT_VERSION = 1;

void 
MultiMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("MultiMaterial", MULTIMATERIAL_VERSION);
  Material::io(str);

  unsigned int size = material_stack_.size();
  SCIRun::Pio(str, size);
  if (str.reading()) {
    material_stack_.resize(size);
  }
  
  for (int i = 0; i < size; i++) {
    MatPercent* m = 0;
    if (str.reading()) {
      m = new MatPercent(0, 0.0);
      material_stack_[i] = m;
    } else {
      m = material_stack_[i];
    }
    SCIRun::Pio(str, *m);
  }
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& str, rtrt::MatPercent& obj)
{
  str.begin_class("MatPercent", MATPERCENT_VERSION);
  SCIRun::Pio(str, obj.material);
  SCIRun::Pio(str, obj.percent);
  str.end_class();
}

void Pio(SCIRun::Piostream& stream, rtrt::MultiMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::MultiMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MultiMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
