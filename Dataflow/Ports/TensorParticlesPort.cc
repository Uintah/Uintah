
#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
IPort* make_TensorParticlesIPort(Module* module,
					     const string& name) {
  return scinew SimpleIPort<TensorParticlesHandle>(module,name);
}
OPort* make_TensorParticlesOPort(Module* module,
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



