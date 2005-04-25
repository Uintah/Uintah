
#include <Packages/Uintah/Core/Datatypes/TensorFieldPort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
IPort* make_TensorFieldIPort(Module* module,
					 const string& name) {
  return scinew SimpleIPort<TensorFieldHandle>(module,name);
}
OPort* make_TensorFieldOPort(Module* module,
					 const string& name) {
  return scinew SimpleOPort<TensorFieldHandle>(module,name);
}
}

template<> string SimpleIPort<TensorFieldHandle>::port_type("TensorField");
template<> string SimpleIPort<TensorFieldHandle>::port_color("yellow4");

