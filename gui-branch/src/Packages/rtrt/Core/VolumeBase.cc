
#include "VolumeBase.h"
#include "VolumeDpy.h"

using namespace rtrt;

VolumeBase::VolumeBase(Material* matl, VolumeDpy* dpy)
    : Object(matl), dpy(dpy)
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


