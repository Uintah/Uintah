#include <Packages/rtrt/Core/InstanceWrapperObject.h>

SCIRun::Persistent* instanceWrapperObject_maker() {
  return new InstanceWrapperObject();
}

// initialize the static member type_id
SCIRun::PersistentTypeID 
InstanceWrapperObject::type_id("InstanceWrapperObject", 
			       "Persistent", 
			       instanceWrapperObject_maker);

const int INSTANCEWRAPPEROBJECT_VERSION = 1;

void 
InstanceWrapperObject::io(SCIRun::Piostream &str)
{
  str.begin_class("InstanceWrapperObject", INSTANCEWRAPPEROBJECT_VERSION);
  Pio(str, obj);
  Pio(str, bb);
  Pio(str, was_processed);
  Pio(str, computed_bbox);
  str.end_class();
}

namespace SCIRun {
void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::InstanceWrapperObject*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::InstanceWrapperObject::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::InstanceWrapperObject*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
