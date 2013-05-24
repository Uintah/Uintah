/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <Core/Containers/LinearArray3.h>
#include <CCA/Components/MD/SPME.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/Matrix3.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h> // for other swapbytes() functions.

#include <complex>
#include <string>

using namespace Uintah;

template<class T>
LinearArray3<T>::LinearArray3()
{
  objs = 0;
  dm1 = dm2 = dm3 = 0;
}

template<class T>
LinearArray3<T>::LinearArray3(int dm1,
                              int dm2,
                              int dm3) :
    dm1(dm1), dm2(dm2), dm3(dm3)
{
  allocate();
}

template<class T>
LinearArray3<T>::LinearArray3(const LinearArray3& copy)
{
  resize(copy.dim1(), copy.dim2(), copy.dim3());
  for (int i = 0; i < dm1; i++) {
    for (int j = 0; j < dm2; j++) {
      for (int k = 0; k < dm3; k++) {
        int idx = (i) + ((j) * dm1) + ((k) * dm1 * dm2);
        objs[idx] = copy.objs[idx];
      }
    }
  }
}

template<class T>
LinearArray3<T>::~LinearArray3()
{
  if (objs) {
    delete objs;
  }
}

template<class T>
void LinearArray3<T>::allocate()
{
  if (dm1 && dm2 && dm3) {
    objs = new T[(dm1 * dm2 * dm3 * sizeof(T))];
  } else {
    objs = 0;
  }
}

template<class T>
void LinearArray3<T>::resize(int d1,
                             int d2,
                             int d3)
{
  if (objs && dm1 == d1 && dm2 == d2 && dm3 == d3) {
    return;
  }
  dm1 = d1;
  dm2 = d2;
  dm3 = d3;
  if (objs) {
    delete objs;
  }
  allocate();
}

template<class T>
void LinearArray3<T>::initialize(const T& t)
{
  ASSERT(objs != 0);
  for (int i = 0; i < dm1; i++) {
    for (int j = 0; j < dm2; j++) {
      for (int k = 0; k < dm3; k++) {
        int idx = (i) + ((j) * dm1) + ((k) * dm1 * dm2);
        objs[idx] = t;
      }
    }
  }
}

template<class T>
const std::string& LinearArray3<T>::get_h_file_path()
{
  static const std::string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

namespace Uintah {

template<class T>
MPI_Datatype makeMPI_LinearArray3()
{
  IntVector elements = SPME::numKElements();

  int x = elements.x();
  int y = elements.y();
  int z = elements.z();

  int count = y * z;      // number of blocks
  int blockLength = x;    // number of elements in each block
  int stride = x;         // number of elements between start of each block

  ASSERTEQ(sizeof(LinearArray3<T>), sizeof(T) * (x*y*z));
  const TypeDescription* td = fun_getTypeDescription((T*)0);
  MPI_Datatype mpitype;
  MPI_Type_vector(count, blockLength, stride, td->getMPIType(), &mpitype);
  MPI_Type_commit(&mpitype);

  return mpitype;
}

template<class T>
const TypeDescription* fun_getTypeDescription(LinearArray3<T>*)
{
  static TypeDescription* td = 0;
  if (!td) {
    //some compilers don't like passing templated function pointers directly across function calls
    MPI_Datatype (*func)() = makeMPI_LinearArray3<T>;
    td = scinew TypeDescription(TypeDescription::LinearArray3, "LinearArray3", true, func);
  }
  return td;
}

}  // namespace Uintah

namespace SCIRun {

template<class T>
const TypeDescription* get_type_description(LinearArray3<T>*)
{
  static TypeDescription* td = 0;
  if (!td) {
    td = scinew TypeDescription("LinearArray3", Point::get_h_file_path(), "SCIRun", TypeDescription::DATA_E);
  }
  return td;
}

template<class T>
void swapbytes(LinearArray3<T>& array)
{
  long int size = array.get_datasize();
  for (int i = 0; i < size; i++) {
    swapbytes(array.objs(i));
  }
}
}  // namespace SCIRun

// Explicit template instantiations:
template class LinearArray3<std::complex<double> > ;
template class LinearArray3<Uintah::Matrix3> ;
template class LinearArray3<double> ;
template class LinearArray3<SCIRun::Vector> ;

