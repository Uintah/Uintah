#include <Packages/rtrt/Core/Instance.h>

SCIRun::Persistent* obj_maker() {
  return new Instance();
}

// initialize the static member type_id
SCIRun::PersistentTypeID Instance::type_id("Instance", "Object", obj_maker);

const int UVSPHERE_VERSION = 1;

void 
Instance::io(SCIRun::Piostream &str)
{
  str.begin_class("Instance", UVSPHERE_VERSION);
  Object::io(str);
  Material::io(str);
  Pio(str, o);
  Pio(str, currentTransform);
  Pio(str, bbox);
  str.end_class();
}

namespace SCIRun {
void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::Instance*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Instance::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Instance*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
