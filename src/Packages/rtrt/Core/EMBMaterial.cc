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


#include <Packages/rtrt/Core/EMBMaterial.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* embmaterial_maker() {
  return new EMBMaterial();
}

// initialize the static member type_id
PersistentTypeID EMBMaterial::type_id("EMBMaterial", "Material", 
					   embmaterial_maker);

const int EMBMATERIAL_VERSION = 1;

void 
EMBMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("EMBMaterial", EMBMATERIAL_VERSION);
  Material::io(str);
  EnvironmentMapBackground::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::EMBMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::EMBMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::EMBMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
