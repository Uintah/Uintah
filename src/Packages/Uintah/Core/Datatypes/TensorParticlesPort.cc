
#include "TensorParticlesPort.h"

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


template<> clString SimpleIPort<TensorParticlesHandle>::port_type("TensorParticles");
template<> clString SimpleIPort<TensorParticlesHandle>::port_color("chartreuse4");


} // End namespace Datatypes
} // End namespace PSECore

