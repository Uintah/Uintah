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



#include <Packages/rtrt/Core/Shadows/MultiSampleSoftShadows.h>
#include <Packages/rtrt/Core/Array1.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* multiSampleSoftShadows_maker() {
  return new MultiSampleSoftShadows();
}

// initialize the static member type_id
PersistentTypeID MultiSampleSoftShadows::type_id("MultiSampleSoftShadows", 
						 "ShadowBase", 
						 multiSampleSoftShadows_maker);

MultiSampleSoftShadows::MultiSampleSoftShadows()
{
}

MultiSampleSoftShadows::~MultiSampleSoftShadows()
{
}


bool MultiSampleSoftShadows::lit(const Point& hitpos, Light* light,
		    const Vector& light_dir, double dist, Color& atten,
		    int depth, Context* cx)
{
  // These are busted.  Do not use Array1....
#if 0
  Array1<Vector>& beamdirs=light->get_beamdirs();
  int n=beamdirs.size();
  for(int i=0;i<n;i++)
    attens[i]=Color(1,1,1);
  obj->multi_light_intersect(light, hitpos, beamdirs, attens,
			     dist, &cx->stats->ds[depth], ppc);
  atten = Color(0,0,0);
  for(int i=0;i<n;i++)
    atten+=attens[i];
  atten = atten*(1.0/n);
#endif
  return true;
}

const int MULTISAMPLESHADOWS_VERSION = 1;

void 
MultiSampleSoftShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("MultiSampleSoftShadows", MULTISAMPLESHADOWS_VERSION);
  ShadowBase::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::MultiSampleSoftShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::MultiSampleSoftShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MultiSampleSoftShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
