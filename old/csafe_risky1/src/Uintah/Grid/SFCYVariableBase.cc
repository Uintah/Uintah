#include <Uintah/Grid/SFCYVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCICore::Exceptions;


SFCYVariableBase::~SFCYVariableBase()
{
}

SFCYVariableBase::SFCYVariableBase()
{
}

void SFCYVariableBase::getMPIBuffer(void*& buf, int& count,
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


void SFCYVariableBase::getMPIBuffer(void*& buf, int& count,
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
// Revision 1.2.4.1  2000/10/20 02:06:37  rawat
// modified cell centered and staggered variables to optimize communication
//
// Revision 1.2  2000/09/25 14:41:32  rawat
// added mpi support for cell centered and staggered cell variables
//
// Revision 1.1  2000/06/27 23:18:18  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//


