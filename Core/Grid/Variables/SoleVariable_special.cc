#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

namespace Uintah { // <- This is necessary for IBM SP AIX xlC Compiler

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
SoleVariable<double>::getMPIInfo(int& count, MPI_Datatype& datatype)
{
   datatype = MPI_DOUBLE;
   count = 1;
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
SoleVariable<double>::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if defined(__sgi) || defined(_AIX)
template<>
#endif
void
SoleVariable<double>::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}


} // end namespace Uintah
