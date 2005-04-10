
#include <Uintah/Datatypes/ScalarParticlesPort.h>
#include <Uintah/share/share.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


extern "C" {
UINTAHSHARE IPort* make_ScalarParticlesIPort(Module* module,
					     const clString& name) {
  return scinew SimpleIPort<ScalarParticlesHandle>(module,name);
}
UINTAHSHARE OPort* make_ScalarParticlesOPort(Module* module,
					     const clString& name) {
  return scinew SimpleOPort<ScalarParticlesHandle>(module,name);
}
}

template<> clString SimpleIPort<ScalarParticlesHandle>::port_type("ScalarParticles");
template<> clString SimpleIPort<ScalarParticlesHandle>::port_color("chartreuse");


} // End namespace Datatypes
} // End namespace PSECore

