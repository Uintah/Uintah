
#include "TensorFieldPort.h"

namespace PSECore {
namespace Datatypes {


using SCICore::Datatypes::TensorFieldHandle;

template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
template<> clString SimpleIPort<TensorFieldHandle>::port_color("yellow4");


} // End namespace Datatypes
} // End namespace PSECore

