
#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
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
SCISHARE IPort* make_TensorParticlesIPort(Module* module,
					     const string& name) {
  return scinew SimpleIPort<TensorParticlesHandle>(module,name);
}
SCISHARE OPort* make_TensorParticlesOPort(Module* module,
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



