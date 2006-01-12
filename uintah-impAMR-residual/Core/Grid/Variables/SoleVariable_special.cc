#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

namespace Uintah { // <- This is necessary for IBM SP AIX xlC Compiler

template<>
void
SoleVariable<double>::getMPIInfo(int& count, MPI_Datatype& datatype)
{
   datatype = MPI_DOUBLE;
   count = 1;
}

template<>
void
SoleVariable<double>::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

template<>
void
SoleVariable<double>::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}


template<>
void
SoleVariable<int>::getMPIInfo(int& count, MPI_Datatype& datatype)
{
   datatype = MPI_INT;
   count = 1;
}

template<>
void
SoleVariable<int>::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(int)));
  int* ptr = reinterpret_cast<int*>(&data[index]);
  *ptr = value;
  index += sizeof(int);
}

template<>
void
SoleVariable<int>::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(int)));
  int* ptr = reinterpret_cast<int*>(&data[index]);
  value = *ptr;
  index += sizeof(int);
}

template<>
void
SoleVariable<bool>::getMPIInfo(int& count, MPI_Datatype& datatype)
{
   datatype = MPI_CHAR;
   count = 1;
}

template<>
void
SoleVariable<bool>::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  *ptr = value;
  index += sizeof(char);
}

template<>
void
SoleVariable<bool>::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  value = *ptr;
  index += sizeof(char);
}


} // end namespace Uintah
