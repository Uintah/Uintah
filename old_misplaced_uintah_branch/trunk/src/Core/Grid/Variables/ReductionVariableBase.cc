
#include <Core/Grid/Variables/ReductionVariableBase.h>
#include <Core/Disclosure/TypeDescription.h>
#include <SCIRun/Core/Exceptions/InternalError.h>

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
  SCI_THROW(InternalError("getRefCounted not implemented for ReductionVariable", __FILE__, __LINE__));
}
    
