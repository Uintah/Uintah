
#include <Packages/Uintah/Core/Grid/Variables/PerPatchBase.h>
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
  SCI_THROW(InternalError("getRefCounted not implemented for PerPatch"));
}

const Uintah::TypeDescription* PerPatchBase::virtualGetTypeDescription() const
{
  SCI_THROW(InternalError("virtualGetTypeDescription not implemented for PerPatch"));
}

void PerPatchBase::offsetGrid(const IntVector& /*offset*/)
{
}

void PerPatchBase::emitNormal(ostream& out, const IntVector& l,
   			      const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat )
{
  SCI_THROW(InternalError("emitNormal not implemented for PerPatch"));

}
void PerPatchBase::readNormal(istream& in, bool swapbytes)
{
  SCI_THROW(InternalError("readNormal not implemented for PerPatch"));

}

void PerPatchBase::allocate(const Patch* patch, const IntVector& boundary)
{
  SCI_THROW(InternalError("Should not call allocate for PerPatch"));

}
