
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Packages/Uintah/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
UINTAHSHARE IPort* make_TensorParticlesIPort(Module* module,
					     const clString& name) {
  return scinew SimpleIPort<TensorParticlesHandle>(module,name);
}
UINTAHSHARE OPort* make_TensorParticlesOPort(Module* module,
					     const clString& name) {
  return scinew SimpleOPort<TensorParticlesHandle>(module,name);
}
}

template<>
clString
SimpleIPort<TensorParticlesHandle>::port_type("TensorParticles");

template<>
clString
SimpleIPort<TensorParticlesHandle>::port_color("chartreuse4");



