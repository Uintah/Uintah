#include<Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

#ifdef __sgi
template<>
#endif
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

#ifdef __sgi
template<>
#endif
void
ReductionVariable<double, Reductions::Max<double> >
   ::getMPIBuffer(void*& buf, int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   buf = &value;
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_MAX;
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

#ifdef __sgi
template<>
#endif
void
ReductionVariable<bool, Reductions::And<bool> >
   ::getMPIBuffer(void*& buf, int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   buf = &value;
   datatype = MPI_CHAR;
   count = 1;
   op = MPI_LAND;
}

#ifdef __sgi
template<>
#endif
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

#ifdef __sgi
template<>
#endif
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

