
#include "ArchivePort.h"

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;


template<> clString SimpleIPort<ArchiveHandle>::port_type("Archive");
template<> clString SimpleIPort<ArchiveHandle>::port_color("lightsteelblue4");


} // End namespace Datatypes
} // End namespace PSECore

