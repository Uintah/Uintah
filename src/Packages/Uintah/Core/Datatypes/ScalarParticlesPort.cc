
#include "ScalarParticlesPort.h"

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


template<> clString SimpleIPort<ScalarParticlesHandle>::port_type("ScalarParticles");
template<> clString SimpleIPort<ScalarParticlesHandle>::port_color("chartreuse");


} // End namespace Datatypes
} // End namespace PSECore

