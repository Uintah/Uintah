
#include <Uintah/Datatypes/VectorParticlesPort.h>
#include <Uintah/share/share.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


extern "C" {
UINTAHSHARE IPort* make_VectorParticlesIPort(Module* module,
					     const clString& name) {
  return scinew SimpleIPort<VectorParticlesHandle>(module,name);
}
UINTAHSHARE OPort* make_VectorParticlesOPort(Module* module,
					     const clString& name) {
  return scinew SimpleOPort<VectorParticlesHandle>(module,name);
}
}

template<> clString SimpleIPort<VectorParticlesHandle>::port_type("VectorParticles");
template<> clString SimpleIPort<VectorParticlesHandle>::port_color("chartreuse3");


} // End namespace Datatypes
} // End namespace PSECore

