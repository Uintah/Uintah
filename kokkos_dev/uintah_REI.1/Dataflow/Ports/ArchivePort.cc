
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

#undef SCISHARE
#ifdef _WIN32
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE
#endif

extern "C" {
  SCISHARE IPort* make_ArchiveIPort(Module* module, const string& name) {
    return scinew SimpleIPort<ArchiveHandle>(module,name);
  }
  SCISHARE OPort* make_ArchiveOPort(Module* module, const string& name) {
    return scinew SimpleOPort<ArchiveHandle>(module,name);
  }
}

template<> string SimpleIPort<ArchiveHandle>::port_type_("Archive");
template<> string SimpleIPort<ArchiveHandle>::port_color_("lightsteelblue4");

