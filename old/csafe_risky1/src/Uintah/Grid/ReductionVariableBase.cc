
#include <Uintah/Grid/ReductionVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>

using namespace Uintah;

ReductionVariableBase::~ReductionVariableBase()
{
}

ReductionVariableBase::ReductionVariableBase()
{
}   

//
// $Log$
// Revision 1.3  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.2  2000/04/26 06:48:53  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

