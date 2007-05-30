#include <Core/Datatypes/Archive.h>
#include <SCIRun/Core/Util/NotFinished.h>
#include <SCIRun/Core/Malloc/Allocator.h>

namespace Uintah {
using namespace SCIRun;

static Persistent* maker()
{
    return scinew Archive;
}

PersistentTypeID Archive::type_id("Data Archive", "Archive", maker);
#define Archive_VERSION 3
void Archive::io(Piostream&)
{
    NOT_FINISHED("Archive::io(Piostream&)");
}
Archive::Archive() :
  archive(0)
{
}

Archive::Archive(const DataArchiveHandle& ar) :
  archive(ar)
{
}

Archive::~Archive()
{
}
} // End namespace Uintah

