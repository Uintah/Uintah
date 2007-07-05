#include <Packages/rtrt/Core/Instance.h>

using namespace rtrt;
using namespace SCIRun;

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
  UVMapping::io(str);
  SCIRun::Pio(str, o);
  SCIRun::Pio(str, currentTransform);
  SCIRun::Pio(str, bbox);
  str.end_class();
}

namespace SCIRun {

void Pio( Piostream& stream, rtrt::Instance*& obj )
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::Instance::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Instance*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun








