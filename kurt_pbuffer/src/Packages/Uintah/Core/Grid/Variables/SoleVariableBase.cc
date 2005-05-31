
#include <Packages/Uintah/Core/Grid/Variables/SoleVariableBase.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;

SoleVariableBase::~SoleVariableBase()
{
}

SoleVariableBase::SoleVariableBase()
{
}   


RefCounted*
SoleVariableBase::getRefCounted()
{
  SCI_THROW(InternalError("getRefCounted not implemented for SoleVariable"));
}
    
