
#include <Packages/rtrt/Core/VolumeVisBase.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>

using namespace rtrt;

// initialize the static member type_id
SCIRun::PersistentTypeID VolumeVisBase::type_id("VolumeVisBase", "Object", 0);


VolumeVisBase::VolumeVisBase(VolumeVisDpy* dpy) : 
  Object(this), 
  dpy(dpy)
{
}

VolumeVisBase::~VolumeVisBase()
{
}

void VolumeVisBase::animate(double, bool& changed)
{
  dpy->animate(changed);
}


const int VOLUMEVISBASE_VERSION = 1;

void 
VolumeVisBase::io(SCIRun::Piostream &str)
{
  str.begin_class("VolumeVisBase", VOLUMEVISBASE_VERSION);
  Object::io(str);
  //Pio(str, dpy);
  if (str.reading()) {
    dpy->attach(this);
  }
  str.end_class();
}

namespace SCIRun {
  void Pio(SCIRun::Piostream& stream, rtrt::VolumeVisBase*& obj)
  {
    SCIRun::Persistent* pobj=obj;
    stream.io(pobj, rtrt::VolumeVisBase::type_id);
    if(stream.reading()) {
      obj=dynamic_cast<rtrt::VolumeVisBase*>(pobj);
      //ASSERT(obj != 0)
    }
  }
} // end namespace SCIRun
