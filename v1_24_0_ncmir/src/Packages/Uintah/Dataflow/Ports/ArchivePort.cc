
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
  UINTAHSHARE IPort* make_ArchiveIPort(Module* module, const string& name) {
    return scinew SimpleIPort<ArchiveHandle>(module,name);
  }
  UINTAHSHARE OPort* make_ArchiveOPort(Module* module, const string& name) {
    return scinew SimpleOPort<ArchiveHandle>(module,name);
  }
}

template<> string SimpleIPort<ArchiveHandle>::port_type_("Archive");
template<> string SimpleIPort<ArchiveHandle>::port_color_("lightsteelblue4");

