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

