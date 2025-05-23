/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Util/FancyAssert.h>

#include <sci_defs/bits_defs.h> // for SCI_32BITS
#include <sci_defs/osx_defs.h>  // for OSX_SNOW_LEOPARD_OR_LATER


using namespace Uintah;
using namespace std;

namespace Uintah { // <- This is necessary for IBM SP AIX xlC Compiler

// Bool AND
#if !defined(__digital__) || defined(__GNUC__)
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

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<bool, Reductions::And<bool> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(char);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<bool, Reductions::And<bool> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  value = std::make_shared<bool>(*ptr);
  index += sizeof(char);
}

// Bool OR
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<bool, Reductions::Or<bool> >
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
ReductionVariable<bool, Reductions::Or<bool> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(char);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<bool, Reductions::Or<bool> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(char)));
  char* ptr = reinterpret_cast<char*>(&data[index]);
  value = std::make_shared<bool>(*ptr);
  index += sizeof(char);
}

//  Double Min
#if !defined(__digital__) || defined(__GNUC__)
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

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<double, Reductions::Min<double> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<double, Reductions::Min<double> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = std::make_shared<double>(*ptr);
  index += sizeof(double);
}

// Double Max
#if !defined(__digital__) || defined(__GNUC__)
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

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<double, Reductions::Max<double> >
   ::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<double, Reductions::Max<double> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = std::make_shared<double>(*ptr);
  index += sizeof(double);
}

// Double Sum
#if !defined(__digital__) || defined(__GNUC__)
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

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<double, Reductions::Sum<double> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<double, Reductions::Sum<double> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value = std::make_shared<double>(*ptr);
  index += sizeof(double);
}

// ARS - This check is not well understood
#if !defined( OSX_SNOW_LEOPARD_OR_LATER ) && !defined( SCI_32BITS )

// Long long min
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long long, Reductions::Min<long long> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_LONG_LONG;
   count = 1;
   op = MPI_MIN;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long long, Reductions::Min<long long> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long * ptr = reinterpret_cast<long long*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(long long);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long long, Reductions::Min<long long> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long * ptr = reinterpret_cast<long long*>(&data[index]);
  value = std::make_shared<long long>(*ptr);
  index += sizeof(long long);
}

// Long long sum
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long long, Reductions::Sum<long long> >
::getMPIInfo(int& count, MPI_Datatype& datatype, MPI_Op& op)
{
   datatype = MPI_LONG_LONG;
   count = 1;
   op = MPI_SUM;
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long long, Reductions::Sum<long long> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long* ptr = reinterpret_cast<long long*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(long long);
}

// ARS - these if defines make no sense as they are around just the
// putMPIData but not the getMPIInfo and getMPIData. They have been
// changed to be around all three functions for both min and sum.

//#if !defined( __PGI ) && !defined( OSX_SNOW_LEOPARD_OR_LATER )
//#  if !defined( SCI_32BITS )
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long long, Reductions::Sum<long long> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long long)));
  long long* ptr = reinterpret_cast<long long*>(&data[index]);
  value = std::make_shared<long long>(*ptr);
  index += sizeof(long long);
}
//#  endif // !defined( SCI_32BITS )
//#endif // !defined( __PGI ) && !defined( OSX_SNOW_LEOPARD_OR_LATER )

#endif // !defined( OSX_SNOW_LEOPARD_OR_LATER ) && !defined( SCI_32BITS )

// Long64 Sum

// ARS - these if defines make no sense as they are around just the
// getMPIInfo and getMPIData but not the putMPIData. It has been
// removed.

// #if !defined( __PGI ) && !defined( OSX_SNOW_LEOPARD_OR_LATER )
#if !defined(__digital__) || defined(__GNUC__)
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

// We reduce a "long", not a long64 because on 2/24/03, LAM-MPI did
// not support Uintah::MPI::Reduce for LONG_LONG_INT.  We could use
// Uintah::MPI::Create_op instead?
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long64, Reductions::Sum<long64> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long)));
  long* ptr = reinterpret_cast<long*>(&data[index]);
  *ptr = *(value.get());
  index += sizeof(long);
}
// #endif // !defined( __PGI ) && !defined( OSX_SNOW_LEOPARD_OR_LATER )

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<long64, Reductions::Sum<long64> >
::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-sizeof(long)));
  long* ptr = reinterpret_cast<long*>(&data[index]);
  value = std::make_shared<long64>(*ptr);
  index += sizeof(long);
}

// Vector Min
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Min<Vector> >
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
ReductionVariable<Vector, Reductions::Min<Vector> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.get()->x();
  *ptr++ = value.get()->y();
  *ptr++ = value.get()->z();
  index += 3*sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Min<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.get()->x(*ptr++);
  value.get()->y(*ptr++);
  value.get()->z(*ptr++);
  index += 3*sizeof(double);
}

// Vector Max
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Max<Vector> >
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
ReductionVariable<Vector, Reductions::Max<Vector> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.get()->x();
  *ptr++ = value.get()->y();
  *ptr++ = value.get()->z();
  index += 3*sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Max<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.get()->x(*ptr++);
  value.get()->y(*ptr++);
  value.get()->z(*ptr++);
  index += 3*sizeof(double);
}

// Vector Sum
#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif
void
ReductionVariable<Vector, Reductions::Sum<Vector> >
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
ReductionVariable<Vector, Reductions::Sum<Vector> >
::getMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  *ptr++ = value.get()->x();
  *ptr++ = value.get()->y();
  *ptr++ = value.get()->z();
  index += 3*sizeof(double);
}

#if !defined(__digital__) || defined(__GNUC__)
template<>
#endif

void
ReductionVariable<Vector, Reductions::Sum<Vector> >
   ::putMPIData(vector<char>& data, int& index)
{
  ASSERTRANGE(index, 0, static_cast<int>(data.size()+1-3*sizeof(double)));
  double* ptr = reinterpret_cast<double*>(&data[index]);
  value.get()->x(*ptr++);
  value.get()->y(*ptr++);
  value.get()->z(*ptr++);
  index += 3*sizeof(double);
}

} // end namespace Uintah
