
#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>
#include <Packages/Uintah/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
  UINTAHSHARE IPort*
  make_ScalarParticlesIPort(Module* module, const string& name) {
    return scinew SimpleIPort<ScalarParticlesHandle>(module,name);
  }
  UINTAHSHARE OPort*
  make_ScalarParticlesOPort(Module* module, const string& name) {
    return scinew SimpleOPort<ScalarParticlesHandle>(module,name);
  }
}

template<>
string
SimpleIPort<ScalarParticlesHandle>::port_type("ScalarParticles");

template<>
string
SimpleIPort<ScalarParticlesHandle>::port_color("chartreuse");

