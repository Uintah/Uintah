
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
  IPort*
  make_ScalarParticlesIPort(Module* module, const string& name) {
    return scinew SimpleIPort<ScalarParticlesHandle>(module,name);
  }
  OPort*
  make_ScalarParticlesOPort(Module* module, const string& name) {
    return scinew SimpleOPort<ScalarParticlesHandle>(module,name);
  }
}

template<>
string
SimpleIPort<ScalarParticlesHandle>::port_type_("ScalarParticles");

template<>
string
SimpleIPort<ScalarParticlesHandle>::port_color_("chartreuse");

