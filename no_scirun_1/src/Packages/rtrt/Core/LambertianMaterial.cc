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


#include <Packages/rtrt/Core/LambertianMaterial.h>
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

Persistent* lambertianMaterial_maker() {
  return new LambertianMaterial();
}

// initialize the static member type_id
PersistentTypeID LambertianMaterial::type_id("LambertianMaterial", "Material", 
					     lambertianMaterial_maker);

LambertianMaterial::LambertianMaterial(const Color& R)
    : R(R)
{
}

LambertianMaterial::~LambertianMaterial()
{
}

void LambertianMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double , const Color& ,
		  Context* cx)
{
  lambertianshade(result, R, ray, hit, depth, cx);
}


const int LAMBERTIANMATERIAL_VERSION = 1;

void 
LambertianMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("LambertianMaterial", LAMBERTIANMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, R);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::LambertianMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::LambertianMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::LambertianMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
