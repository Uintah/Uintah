
#include <Packages/Uintah/Core/Grid/PerPatchBase.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;

PerPatchBase::~PerPatchBase()
{
}

PerPatchBase::PerPatchBase()
{
}


RefCounted*
PerPatchBase::getRefCounted()
{
  throw InternalError("getRefCounted not implemented for PerPatch");
}
