
#include <Dataflow/Ports/ScalarParticlesPort.h>
#include <SCIRun/Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllexport)
#else
#define UINTAHSHARE
#endif

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
SimpleIPort<ScalarParticlesHandle>::port_type_("ScalarParticles");

template<>
string
SimpleIPort<ScalarParticlesHandle>::port_color_("chartreuse");

