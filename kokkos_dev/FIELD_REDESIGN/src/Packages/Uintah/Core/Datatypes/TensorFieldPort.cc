
#include <Uintah/Datatypes/TensorFieldPort.h>
#include <Uintah/share/share.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {


using SCICore::Datatypes::TensorFieldHandle;

extern "C" {
UINTAHSHARE IPort* make_TensorFieldIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<TensorFieldHandle>(module,name);
}
UINTAHSHARE OPort* make_TensorFieldOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<TensorFieldHandle>(module,name);
}
}

template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
template<> clString SimpleIPort<TensorFieldHandle>::port_color("yellow4");


} // End namespace Datatypes
} // End namespace PSECore

