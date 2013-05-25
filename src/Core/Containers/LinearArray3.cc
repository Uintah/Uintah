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
#include <ostream>
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
  long int size = get_datasize();
  for (long int idx = 0; idx < size; idx++) {
    objs[idx] = t;
  }
}

namespace Uintah {

  MPI_Datatype makeMPI_LinearArray3()
  {

    int x = 32;
    int y = 32;
    int z = 32;
    int count = y * z;      // number of blocks
    int blockLength = x;    // number of elements in each block
    int stride = x;         // number of elements between start of each block

    ASSERTEQ(sizeof(LinearArray3<dblcomplex>), sizeof(dblcomplex) * (x*y*z));

    MPI_Datatype mpitype;
    MPI_Type_vector(count, blockLength, stride, MPI_C_DOUBLE_COMPLEX, &mpitype);
    MPI_Type_commit(&mpitype);

    return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(LinearArray3<std::complex<double> >*)
  {
    static TypeDescription* td = 0;
    if (!td) {
      td = scinew TypeDescription(TypeDescription::LinearArray3, "LinearArray3", true, &makeMPI_LinearArray3);
    }
    return td;
  }

}  // namespace Uintah

namespace SCIRun {

  void swapbytes(LinearArray3<std::complex<double> >& la3)
  {
    int dm1 = la3.dim1();
    int dm2 = la3.dim2();
    int dm3 = la3.dim3();

    for (int i = 0; i < dm1; i++) {
      for (int j = 0; j < dm2; j++) {
        for (int k = 0; k < dm3; k++) {
          swapbytes(la3(i, j, k));
        }
      }
    }
  }

  std::ostream& operator<<(ostream& out_file,
                           const LinearArray3<std::complex<double> > &la3)
  {
    // Overload the output stream << operator
    int dm1 = la3.dim1();
    int dm2 = la3.dim2();
    int dm3 = la3.dim3();
    for (int i = 0; i < dm1; i++) {
      for (int j = 0; j < dm2; j++) {
        for (int k = 0; k < dm3; k++) {
          out_file << "objs[" << i << "][" << j << "][" << k << " =\t" << la3(i, j, k) << ' ' << std::endl;
        }
      }
    }
    return out_file;
  }

}  // namespace SCIRun

// Explicit template instantiations:
template class LinearArray3<std::complex<double> >;
template class LinearArray3<Uintah::Matrix3>;
template class LinearArray3<double>;
template class LinearArray3<SCIRun::Vector>;

