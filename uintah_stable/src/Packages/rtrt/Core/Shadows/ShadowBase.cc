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



#include <Packages/rtrt/Core/Shadows/ShadowBase.h>
#include <Core/Persistent/Persistent.h>

using namespace rtrt;

// initialize the static member type_id
SCIRun::PersistentTypeID ShadowBase::type_id("ShadowBase", "Persistent", 0);

char * ShadowBase::shadowTypeNames[] = { "No Shadows",
					 "Single Soft Shadow",
					 "Hard Shadows",
					 "Glass Shadows",
					 "Soft Shadows",
					 "Uncached Shadows" };

ShadowBase::ShadowBase()
  : name("unknown")
{
}

ShadowBase::~ShadowBase()
{
}

void ShadowBase::preprocess(Scene*, int&, int&)
{
}

int ShadowBase::increment_shadow_type(int shadow_type) {
  if( shadow_type == Uncached_Shadows )
    return No_Shadows;
  else
    return shadow_type+1;
}

int ShadowBase::decrement_shadow_type(int shadow_type) {
  if( shadow_type == No_Shadows )
    return Uncached_Shadows;
  else
    return shadow_type-1;
}

const int SHADOWBASE_VERSION = 1;
void 
ShadowBase::io(SCIRun::Piostream &str)
{
  str.begin_class("ShadowBase", SHADOWBASE_VERSION);
  //  Pio(str, matl);
  //Pio(str, uv);
  str.end_class();
}

namespace SCIRun {
void Pio(Piostream& stream, rtrt::ShadowBase*& obj)
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::ShadowBase::type_id);
  if(stream.reading())
    obj=(rtrt::ShadowBase*)pobj;
}
} // end namespace SCIRun
