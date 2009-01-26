/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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
