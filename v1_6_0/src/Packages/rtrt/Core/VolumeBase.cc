
#include <Packages/rtrt/Core/VolumeBase.h>
#include <Packages/rtrt/Core/VolumeDpy.h>

using namespace rtrt;

// initialize the static member type_id
SCIRun::PersistentTypeID VolumeBase::type_id("VolumeBase", "Object", 0);


VolumeBase::VolumeBase(Material* matl, VolumeDpy* dpy) : 
  Object(matl), 
  dpy(dpy)
{
    dpy->attach(this);
}

VolumeBase::~VolumeBase()
{
}

void VolumeBase::animate(double, bool& changed)
{
    dpy->animate(changed);
}


const int VOLUMEBASE_VERSION = 1;

void 
VolumeBase::io(SCIRun::Piostream &str)
{
  str.begin_class("VolumeBase", VOLUMEBASE_VERSION);
  Object::io(str);
  //Pio(str, dpy);
  if (str.reading()) {
    dpy->attach(this);
  }
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::VolumeBase*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::VolumeBase::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::VolumeBase*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
