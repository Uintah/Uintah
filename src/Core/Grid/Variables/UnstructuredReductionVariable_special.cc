/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/Variables/UnstructuredReductionVariable.h>

#include <Core/Geometry/Vector.h>
#include <Core/Util/FancyAssert.h>

#include <sci_defs/bits_defs.h> // for SCI_32BITS
#include <sci_defs/osx_defs.h>  // for OSX_SNOW_LEOPARD_OR_LATER

#include <Core/Disclosure/UnstructuredTypeUtils.h>

using namespace Uintah;
using namespace std;

namespace Uintah { // <- This is necessary for IBM SP AIX xlC Compiler

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<double, UnstructuredReductions::Min<double> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_MIN;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<double, UnstructuredReductions::Min<double> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if !defined( SCI_32BITS )
#  if !defined(__digital__) || defined(__GNUC__)
template<>
#  endif
 void
UnstructuredReductionVariable<long long, UnstructuredReductions::Min<long long> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long* ptr = reinterpret_cast<long long*>(&data[index]);
  *ptr = value;
  index += sizeof(long long);
}

#  if !defined(__digital__) || defined(__GNUC__)
template<>
#  endif
 void
UnstructuredReductionVariable<long long, UnstructuredReductions::Sum<long long> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long* ptr = reinterpret_cast<long long*>(&data[index]);
  *ptr = value;
  index += sizeof(long long);
}
#endif

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<double, UnstructuredReductions::Min<double> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<double, UnstructuredReductions::Max<double> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_MAX;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<double, UnstructuredReductions::Max<double> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<double, UnstructuredReductions::Max<double> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<double, UnstructuredReductions::Sum<double> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 1;
   op = MPI_SUM;
}

#if !defined( SCI_32BITS )
#  if !defined(__digital__) || defined(__GNUC__)
template<>
#  endif
 void
UnstructuredReductionVariable<long long, UnstructuredReductions::Sum<long long> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_LONG_LONG;
   count = 1;
   op = MPI_SUM;
}
#endif

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<double, UnstructuredReductions::Sum<double> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = value;
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<double, UnstructuredReductions::Sum<double> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = *ptr;
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<bool, UnstructuredReductions::And<bool> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_CHAR;
   count = 1;
   op = MPI_LAND;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<bool, UnstructuredReductions::And<bool> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  *ptr = value;
  index += sizeof(char);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<bool, UnstructuredReductions::And<bool> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  value = *ptr;
  index += sizeof(char);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

void
UnstructuredReductionVariable<bool, UnstructuredReductions::Or<bool> >
   ::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_CHAR;
   count = 1;
   op = MPI_LOR;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<bool, UnstructuredReductions::Or<bool> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  *ptr = value;
  index += sizeof(char);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

void
UnstructuredReductionVariable<bool, UnstructuredReductions::Or<bool> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  value = *ptr;
  index += sizeof(char);
}

#if !defined( __PGI ) && !defined( OSX_SNOW_LEOPARD_OR_LATER )
// We reduce a "long", not a long64 because on 2/24/03, LAM-MPI did not
// support Uintah::MPI::Reduce for LONG_LONG_INT.  We could use Uintah::MPI::Create_op instead?
  #if !defined(__digital__) || defined(__GNUC__)
template<>
  #endif
 void
UnstructuredReductionVariable<long64, UnstructuredReductions::Sum<long64> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_LONG;
   count = 1;
   op = MPI_SUM;
}

#  if !defined(__digital__) || defined(__GNUC__)
template<>
#  endif
 void
UnstructuredReductionVariable<long64, UnstructuredReductions::Sum<long64> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long)));
  long* ptr = reinterpret_cast<long*>(&data[index]);
  *ptr = value;
  index += sizeof(long);
}
#endif

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<long64, UnstructuredReductions::Sum<long64> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long)));
  long* ptr = reinterpret_cast<long*>(&data[index]);
  value = *ptr;
  index += sizeof(long);
}

#if !defined( OSX_SNOW_LEOPARD_OR_LATER ) && !defined( __PGI )
#  if !defined( SCI_32BITS )
#    if !defined(__digital__) || defined(__GNUC__)
template<>
#    endif
 void
UnstructuredReductionVariable<long long, UnstructuredReductions::Sum<long long> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long* ptr = reinterpret_cast<long long*>(&data[index]);
  value = *ptr;
  index += sizeof(long long);
}
#  endif
#endif

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Sum<Vector> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 3;
   op = MPI_SUM;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Sum<Vector> >
::getMPIData(vector<char>& data, int& index)
{       
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.x();
  *ptr++ = value.y();
  *ptr++ = value.z();
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Sum<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.x(*ptr++);
  value.y(*ptr++);
  value.z(*ptr++);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Min<Vector> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 3;
   op = MPI_MIN;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Min<Vector> >
::getMPIData(vector<char>& data, int& index)
{       
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.x();
  *ptr++ = value.y();
  *ptr++ = value.z();
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Min<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.x(*ptr++);
  value.y(*ptr++);
  value.z(*ptr++);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Max<Vector> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_DOUBLE;
   count = 3;
   op = MPI_MAX;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Max<Vector> >
::getMPIData(vector<char>& data, int& index)
{       
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.x();
  *ptr++ = value.y();
  *ptr++ = value.z();
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

 void
UnstructuredReductionVariable<Vector, UnstructuredReductions::Max<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.x(*ptr++);
  value.y(*ptr++);
  value.z(*ptr++);
}

} // end namespace Uintah
