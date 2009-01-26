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


#include <Packages/rtrt/Core/MultiMaterial.h>


using namespace rtrt;
using namespace SCIRun;

Persistent* multiMaterial_maker() {
  return new MultiMaterial();
}

// initialize the static member type_id
PersistentTypeID MultiMaterial::type_id("MultiMaterial", "Material", multiMaterial_maker);

const int MULTIMATERIAL_VERSION = 1;
const int MATPERCENT_VERSION = 1;

void 
MultiMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("MultiMaterial", MULTIMATERIAL_VERSION);
  Material::io(str);

  size_t size = material_stack_.size();
  SCIRun::Pio(str, size);
  if (str.reading()) {
    material_stack_.resize(size);
  }
  
  for (size_t i = 0; i < size; i++) {
    MatPercent* m = 0;
    if (str.reading()) {
      m = new MatPercent(0, 0.0);
      material_stack_[i] = m;
    } else {
      m = material_stack_[i];
    }
    SCIRun::Pio(str, *m);
  }
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& str, rtrt::MatPercent& obj)
{
  str.begin_class("MatPercent", MATPERCENT_VERSION);
  SCIRun::Pio(str, obj.material);
  SCIRun::Pio(str, obj.percent);
  str.end_class();
}

void Pio(SCIRun::Piostream& stream, rtrt::MultiMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::MultiMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MultiMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
