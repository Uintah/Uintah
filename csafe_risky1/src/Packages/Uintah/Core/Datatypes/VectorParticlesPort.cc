
#include "VectorParticlesPort.h"

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


template<> clString SimpleIPort<VectorParticlesHandle>::port_type("VectorParticles");
template<> clString SimpleIPort<VectorParticlesHandle>::port_color("chartreuse3");


} // End namespace Datatypes
} // End namespace PSECore

