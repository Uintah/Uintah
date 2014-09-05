
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

#undef SCISHARE
#ifdef _WIN32
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE
#endif

extern "C" {
  SCISHARE IPort*
  make_ScalarParticlesIPort(Module* module, const string& name) {
    return scinew SimpleIPort<ScalarParticlesHandle>(module,name);
  }
  SCISHARE OPort*
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

