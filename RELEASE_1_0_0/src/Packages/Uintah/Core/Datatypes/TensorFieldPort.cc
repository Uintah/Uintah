
#include <Packages/Uintah/Core/Datatypes/TensorFieldPort.h>
#include <Packages/Uintah/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

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

