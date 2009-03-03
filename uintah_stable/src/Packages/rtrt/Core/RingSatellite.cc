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



#include <Packages/rtrt/Core/RingSatellite.h>

namespace rtrt {
using namespace SCIRun;

Persistent* ring_satellite_maker() {
  return new RingSatellite();
}

// initialize the static member type_id
PersistentTypeID RingSatellite::type_id("RingSatellite", "Object", ring_satellite_maker);

void RingSatellite::animate(double /*t*/, bool& changed)
{
  cen = parent_->get_center();
  d=Dot(this->n, cen);
  changed = true;
}
  
void RingSatellite::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
  // radial mapping of a 1D texture
  double hitdist = (hitpos-cen).length();
  double edge1 = hitdist-radius;
  double edge2 = thickness;
  double u = edge1/edge2;
  uv.set(u,0);
}

const int RING_SATELLITE_VERSION = 1;

void 
RingSatellite::io(SCIRun::Piostream &str)
{
  str.begin_class("RingSatellite", RING_SATELLITE_VERSION);
  Ring::io(str);
  SCIRun::Pio(str, parent_);
  str.end_class();
}
} // end namespace

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::RingSatellite*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::RingSatellite::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::RingSatellite*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
