
#include <Kurt/Datatypes/VisParticleSetPort.h>
#include <Kurt/share/share.h>

namespace PSECore {
namespace Datatypes {

using namespace Kurt::Datatypes;


extern "C" {
KurtSHARE IPort* make_VisParticleSetIPort(Module* module,
					 const clString& name) {
  return new SimpleIPort<VisParticleSetHandle>(module,name);
}
KurtSHARE OPort* make_VisParticleSetOPort(Module* module,
					 const clString& name) {
  return new SimpleOPort<VisParticleSetHandle>(module,name);
}
}

template<> clString SimpleIPort<VisParticleSetHandle>::port_type("VisParticleSet");
template<> clString SimpleIPort<VisParticleSetHandle>::port_color("chartreuse2");


} // End namespace Datatypes
} // End namespace PSECore

