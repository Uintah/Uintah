
#include <Packages/Uintah/Core/Grid/ReductionVariableBase.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>

using namespace Uintah;

ReductionVariableBase::~ReductionVariableBase()
{
}

ReductionVariableBase::ReductionVariableBase()
{
}   


RefCounted*
ReductionVariableBase::getRefCounted()
{
  throw InternalError("getRefCounted not implemented for ReductionVariable");
}
    
