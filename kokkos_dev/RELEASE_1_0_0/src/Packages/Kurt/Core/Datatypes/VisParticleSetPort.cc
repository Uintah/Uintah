
#include <Packages/Kurt/Core/Datatypes/VisParticleSetPort.h>
#include <Packages/Kurt/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace Kurt {
using namespace Kurt::Datatypes;


extern "C" {
Packages/KurtSHARE IPort* make_VisParticleSetIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<VisParticleSetHandle>(module,name);
}
Packages/KurtSHARE OPort* make_VisParticleSetOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<VisParticleSetHandle>(module,name);
}
}

template<> clString SimpleIPort<VisParticleSetHandle>::port_type("VisParticleSet");
template<> clString SimpleIPort<VisParticleSetHandle>::port_color("chartreuse2");

} // End namespace Kurt


