
#include <Packages/rtrt/Core/VolumeVGBase.h>
#include <Packages/rtrt/Core/Hist2DDpy.h>

using namespace rtrt;

VolumeVGBase::VolumeVGBase(Material* matl, Hist2DDpy* dpy)
  : Object(matl), dpy(dpy)
{
  dpy->attach(this);
}

VolumeVGBase::~VolumeVGBase()
{
}

void VolumeVGBase::animate(double, bool& changed)
{
  dpy->animate(changed);
}


