
#include <Uintah/Datatypes/TensorParticlesPort.h>
#include <Uintah/share/share.h>

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


extern "C" {
UINTAHSHARE IPort* make_TensorParticlesIPort(Module* module,
					     const clString& name) {
  return new SimpleIPort<TensorParticlesHandle>(module,name);
}
UINTAHSHARE OPort* make_TensorParticlesOPort(Module* module,
					     const clString& name) {
  return new SimpleOPort<TensorParticlesHandle>(module,name);
}
}

template<> clString SimpleIPort<TensorParticlesHandle>::port_type("TensorParticles");
template<> clString SimpleIPort<TensorParticlesHandle>::port_color("chartreuse4");


} // End namespace Datatypes
} // End namespace PSECore

