
#include <Uintah/Grid/NCVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Exceptions/InternalError.h>
#include <iostream>

using namespace Uintah;
using namespace SCICore::Exceptions;

NCVariableBase::~NCVariableBase()
{
}

NCVariableBase::NCVariableBase()
{
}

void NCVariableBase::getMPIBuffer(void*& buf, int& count,
				  MPI_Datatype& datatype, bool& free_datatype,
				  const IntVector& low, const IntVector& high)
{
   const TypeDescription* td = virtualGetTypeDescription()->getSubType();
   MPI_Datatype basetype=td->getMPIType();
   IntVector l, h, s, strides;
   getSizes(l, h, s, strides);

   IntVector off = low-l;
   char* startbuf = (char*)getBasePointer();
   startbuf += strides.x()*off.x()+strides.y()*off.y()+strides.z()*off.z();
   buf = startbuf;
   IntVector d = high-low;
   MPI_Datatype type1d;
   MPI_Type_hvector(d.x(), 1, strides.x(), basetype, &type1d);
   using namespace std;
   MPI_Datatype type2d;
   MPI_Type_hvector(d.y(), 1, strides.y(), type1d, &type2d);
   MPI_Type_free(&type1d);
   MPI_Type_hvector(d.z(), 1, strides.z(), type2d, &datatype);
   MPI_Type_free(&type2d);
   MPI_Type_commit(&datatype);
   free_datatype=true;
   count=1;
}

//
// $Log$
// Revision 1.4  2000/12/10 09:06:16  sparker
// Merge from csafe_risky1
//
// Revision 1.3.4.2  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.3.4.1  2000/09/29 06:12:29  sparker
// Added support for sending data along patch edges
//
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

