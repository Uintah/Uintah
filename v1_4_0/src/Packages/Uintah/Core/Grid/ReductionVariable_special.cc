#include<Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

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
ReductionVariable<long64, Reductions::Sum<long64> >
   ::getMPIBuffer(void*& buf, int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   buf = &value;
   datatype = MPI_LONG;
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

