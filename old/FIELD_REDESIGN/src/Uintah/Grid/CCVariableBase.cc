#include <Uintah/Grid/CCVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCICore::Exceptions;


CCVariableBase::~CCVariableBase()
{
}

CCVariableBase::CCVariableBase()
{
}

void CCVariableBase::getMPIBuffer(void*& buf, int& count,
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
// Revision 1.2  2000/09/25 14:41:31  rawat
// added mpi support for cell centered and staggered cell variables
//
// Revision 1.1  2000/05/11 20:12:34  dav
// Added CCVaraibleBase
//
//

