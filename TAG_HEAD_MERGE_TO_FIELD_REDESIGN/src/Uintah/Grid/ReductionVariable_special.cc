
#include <Uintah/Grid/ReductionVariable.h>
#include <SCICore/Geometry/Vector.h>

using namespace Uintah;
using namespace SCICore::Geometry;

template<>
void
ReductionVariable<double, Reductions::Min<double> >
   ::getMPIBuffer(void*& buf, int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   buf = &value;
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_MIN;
}

template<>
void
ReductionVariable<double, Reductions::Sum<double> >
   ::getMPIBuffer(void*& buf, int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   buf = &value;
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_SUM;
}

template<>
void
ReductionVariable<Vector, Reductions::Sum<Vector> >
   ::getMPIBuffer(void*& buf, int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   buf = &value;
   datatype = MPI_DOUBLE;
   count = 3;
   op = MPI_SUM;
}

//
// $Log$
// Revision 1.1  2000/07/28 00:41:47  sparker
// Specializations for ReductionVariables (for MPI)
//
//
