
#include <Uintah/Grid/NCVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCICore::Exceptions;

NCVariableBase::~NCVariableBase()
{
}

NCVariableBase::NCVariableBase()
{
}

void NCVariableBase::getMPIBuffer(void*& buf, int& count,
				  MPI_Datatype& datatype)
{
   buf = getBasePointer();
   const TypeDescription* td = virtualGetTypeDescription()->getSubType();
   datatype=td->getMPIType();
   IntVector low, high, size;
   getSizes(low, high, size);
   IntVector d = high-low;
   if(d != size)
      throw InternalError("getMPIBuffer needs to be smarter to send windowed arrays");
   count = d.x()*d.y()*d.z();
}

//
// $Log$
// Revision 1.3  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.2  2000/04/26 06:48:50  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/20 22:58:19  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
//

