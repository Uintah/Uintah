
#include <Packages/Uintah/Core/Grid/ReductionVariableBase.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;

ReductionVariableBase::~ReductionVariableBase()
{
}

ReductionVariableBase::ReductionVariableBase()
{
}   


RefCounted*
ReductionVariableBase::getRefCounted()
{
  SCI_THROW(InternalError("getRefCounted not implemented for ReductionVariable"));
}
    
