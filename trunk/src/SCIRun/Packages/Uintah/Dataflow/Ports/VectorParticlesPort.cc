
#include <Packages/Uintah/Dataflow/Ports/VectorParticlesPort.h>
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
  SCISHARE IPort* make_VectorParticlesIPort(Module* module,
					       const string& name) {
    return scinew SimpleIPort<VectorParticlesHandle>(module,name);
  }
  SCISHARE OPort* make_VectorParticlesOPort(Module* module,
					       const string& name) {
    return scinew SimpleOPort<VectorParticlesHandle>(module,name);
  }
}

template<> string SimpleIPort<VectorParticlesHandle>::port_type_("VectorParticles");
template<> string SimpleIPort<VectorParticlesHandle>::port_color_("chartreuse3");


