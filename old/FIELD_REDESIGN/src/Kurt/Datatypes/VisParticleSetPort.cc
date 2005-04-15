
#include "VisParticleSetPort.h"

namespace PSECore {
namespace Datatypes {

using namespace Kurt::Datatypes;


template<> clString SimpleIPort<VisParticleSetHandle>::port_type("VisParticleSet");
template<> clString SimpleIPort<VisParticleSetHandle>::port_color("chartreuse2");


} // End namespace Datatypes
} // End namespace PSECore

