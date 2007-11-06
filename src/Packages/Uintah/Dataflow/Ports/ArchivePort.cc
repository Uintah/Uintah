
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllexport)
#else
#define UINTAHSHARE
#endif

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

