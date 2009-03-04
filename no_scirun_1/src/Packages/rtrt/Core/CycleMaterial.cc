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


#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <cmath>

using namespace rtrt;
using namespace SCIRun;

Persistent* cycleMaterial_maker() {
  return new CycleMaterial();
}

// initialize the static member type_id
PersistentTypeID CycleMaterial::type_id("CycleMaterial", "Material", 
					cycleMaterial_maker);


CycleMaterial::CycleMaterial()
    : current(0)
{
}

CycleMaterial::~CycleMaterial()
{
}

void CycleMaterial::next() {
  if (members.size() == 0) ASSERTFAIL("Cycle material has no members");
  if (current+1 == members.size())
    current=0;
  else
    current++;
}

void CycleMaterial::prev() {
  if (members.size() == 0) ASSERTFAIL("Cycle material has no members");
  if (current-1 < 0)
    current=members.size()-1;
  else
    current--;
}

void CycleMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx)
{
  members[current]->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

const int CYCLEMATERIAL_VERSION = 1;

void 
CycleMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("CycleMaterial", CYCLEMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, current);
  SCIRun::Pio(str, members);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::CycleMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::CycleMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::CycleMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
