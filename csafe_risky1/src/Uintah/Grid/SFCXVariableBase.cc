#include <Uintah/Grid/SFCXVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCICore::Exceptions;


SFCXVariableBase::~SFCXVariableBase()
{
}

SFCXVariableBase::SFCXVariableBase()
{
}


void SFCXVariableBase::getMPIBuffer(void*& buf, int& count,
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
// Revision 1.2  2000/09/25 14:41:32  rawat
// added mpi support for cell centered and staggered cell variables
//
// Revision 1.1  2000/06/27 23:18:17  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//


