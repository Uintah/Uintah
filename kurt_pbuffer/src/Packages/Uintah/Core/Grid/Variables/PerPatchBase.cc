
#include <Packages/Uintah/Core/Grid/Variables/PerPatchBase.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

PerPatchBase::~PerPatchBase()
{
}

PerPatchBase::PerPatchBase()
{
}


RefCounted*
PerPatchBase::getRefCounted()
{
  SCI_THROW(InternalError("getRefCounted not implemented for PerPatch"));
}

const Uintah::TypeDescription* PerPatchBase::virtualGetTypeDescription() const
{
  SCI_THROW(InternalError("virtualGetTypeDescription not implemented for PerPatch"));
}

void PerPatchBase::offsetGrid(const IntVector&)
{
}

void PerPatchBase::emitNormal(ostream&, const IntVector&,
                              const IntVector&, ProblemSpecP, bool)
{
  SCI_THROW(InternalError("emitNormal not implemented for PerPatch"));

}
void PerPatchBase::readNormal(istream&, bool)
{
  SCI_THROW(InternalError("readNormal not implemented for PerPatch"));

}

void PerPatchBase::allocate(const Patch*, const IntVector&)
{
  SCI_THROW(InternalError("Should not call allocate for PerPatch"));

}
