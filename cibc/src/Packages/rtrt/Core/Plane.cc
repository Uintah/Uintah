#include <Packages/rtrt/Core/Plane.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* plane_maker() {
  return new Plane();
}

// initialize the static member type_id
PersistentTypeID Plane::type_id("Plane", "Persistent", plane_maker);

Plane::~Plane() 
{
}

const int PLANE_VERSION = 1;

void 
Plane::io(SCIRun::Piostream &str)
{
  str.begin_class("Plane", PLANE_VERSION);
  SCIRun::Pio(str, inplane);
  SCIRun::Pio(str, normal);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Plane*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Plane::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Plane*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
