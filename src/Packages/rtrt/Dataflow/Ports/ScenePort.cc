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


#include <Core/Persistent/Persistent.h>
#include <Packages/rtrt/Dataflow/Ports/ScenePort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

using namespace rtrt;
using namespace SCIRun;

PersistentTypeID SceneContainer::type_id("SceneContainer","Datatype", 0);

void SceneContainer::io(Piostream&) {
  NOT_FINISHED("SceneContainer::io(Piostream&)");
}

extern "C" {
  IPort* make_SceneIPort(Module* module, const string& name) {
    return scinew SimpleIPort<SceneHandle>(module,name);
  }
  OPort* make_SceneOPort(Module* module, const string& name) {
    return scinew SimpleOPort<SceneHandle>(module,name);
  }
}

namespace SCIRun {
  using namespace rtrt;
  
  template<> string SimpleIPort<SceneHandle>::port_type_("Scene");
  template<> string SimpleIPort<SceneHandle>::port_color_("lightsteelblue4");
}

