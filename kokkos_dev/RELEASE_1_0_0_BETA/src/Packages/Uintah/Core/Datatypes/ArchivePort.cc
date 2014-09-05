
#include <Packages/Uintah/Core/Datatypes/ArchivePort.h>
#include <Packages/Uintah/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

extern "C" {
  UINTAHSHARE IPort* make_ArchiveIPort(Module* module, const clString& name) {
    return scinew SimpleIPort<ArchiveHandle>(module,name);
  }
  UINTAHSHARE OPort* make_ArchiveOPort(Module* module, const clString& name) {
    return scinew SimpleOPort<ArchiveHandle>(module,name);
  }
}

template<> clString SimpleIPort<ArchiveHandle>::port_type("Archive");
template<> clString SimpleIPort<ArchiveHandle>::port_color("lightsteelblue4");

