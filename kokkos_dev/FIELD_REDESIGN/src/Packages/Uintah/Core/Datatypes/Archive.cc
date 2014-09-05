#include "Archive.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>

namespace Uintah {
namespace Datatypes {

using SCICore::Datatypes::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;


static Persistent* maker()
{
    return scinew Archive;
}

PersistentTypeID Archive::type_id("Data Archive", "Archive", maker);
#define Archive_VERSION 3
void Archive::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
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

} // end namespace Datatypes
} // end namespace Uintah
