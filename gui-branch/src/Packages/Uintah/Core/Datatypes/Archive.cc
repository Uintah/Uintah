#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>

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

Archive::Archive(DataArchive *ar) :
  archive(ar)
{
}

Archive::~Archive()
{
}
} // End namespace Uintah

