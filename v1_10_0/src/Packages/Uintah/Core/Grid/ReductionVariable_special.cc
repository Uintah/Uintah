#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

namespace Uintah { // <- This is necessary for IBM SP AIX xlC Compiler

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Min<double> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_MIN;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Min<double> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Min<double> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Max<double> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_MAX;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Max<double> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Max<double> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Sum<double> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_SUM;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Sum<double> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<double, Reductions::Sum<double> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<bool, Reductions::And<bool> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_CHAR;
   count = 1;
   op = MPI_LAND;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<bool, Reductions::And<bool> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(char), 0, data.size()+1);
  char* ptr = reinterpret_cast<char*>(&data[index]);
  *ptr = value;
  index += sizeof(char);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<bool, Reductions::And<bool> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(char), 0, data.size()+1);
  char* ptr = reinterpret_cast<char*>(&data[index]);
  value = *ptr;
  index += sizeof(char);
}

// We reduce a "long", not a long64 because on 2/24/03, LAM-MPI did not
// support MPI_Reduce for LONG_LONG_INT.  We could use MPI_Create_op instead?
#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<long64, Reductions::Sum<long64> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_LONG;
   count = 1;
   op = MPI_SUM;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<long64, Reductions::Sum<long64> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(long), 0, data.size()+1);
  long* ptr = reinterpret_cast<long*>(&data[index]);
  *ptr = value;
  index += sizeof(long);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<long64, Reductions::Sum<long64> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+sizeof(long), 0, data.size()+1);
  long* ptr = reinterpret_cast<long*>(&data[index]);
  value = *ptr;
  index += sizeof(long);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Sum<Vector> >
   ::getMPIInfo(int& count,
		  MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 3;
   op = MPI_SUM;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Sum<Vector> >
   ::getMPIData(vector<char>& data, int& index)
{	
  ASSERTRANGE(index+3*sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.x();
  *ptr++ = value.y();
  *ptr++ = value.z();
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Sum<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index+3*sizeof(double), 0, data.size()+1);
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.x(*ptr++);
  value.y(*ptr++);
  value.z(*ptr++);
}

} // end namespace Uintah
