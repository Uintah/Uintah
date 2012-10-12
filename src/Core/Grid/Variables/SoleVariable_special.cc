/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Disclosure/TypeUtils.h>
#include <Core/Grid/Variables/SoleVariable.h>
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
