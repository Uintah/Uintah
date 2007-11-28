
#include <Dataflow/Ports/TensorParticlesPort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllexport)
#else
#define UINTAHSHARE
#endif

extern "C" {
UINTAHSHARE IPort* make_TensorParticlesIPort(Module* module,
					     const string& name) {
  return scinew SimpleIPort<TensorParticlesHandle>(module,name);
}
UINTAHSHARE OPort* make_TensorParticlesOPort(Module* module,
					     const string& name) {
  return scinew SimpleOPort<TensorParticlesHandle>(module,name);
}
}

template<>
string
SimpleIPort<TensorParticlesHandle>::port_type_("TensorParticles");

template<>
string
SimpleIPort<TensorParticlesHandle>::port_color_("chartreuse4");



