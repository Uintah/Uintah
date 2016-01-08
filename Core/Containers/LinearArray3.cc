/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <Core/Geometry/Vector.h>
#include <Core/Math/Matrix3.h>
#include <Core/Util/Assert.h>

#include <complex>

using namespace SCIRun;

template<class T>
LinearArray3<T>::LinearArray3()
{
  objs = 0;
  dm1 = dm2 = dm3 = 0;
}

template<class T>
LinearArray3<T>::LinearArray3( int dm1,
                               int dm2,
                               int dm3 ) :
    dm1(dm1), dm2(dm2), dm3(dm3)
{
  allocate();
}

template<class T>
LinearArray3<T>::LinearArray3( int dim1,
                               int dim2,
                               int dim3,
                               T   value ) :
    dm1(dim1), dm2(dim2), dm3(dim3)
{
  allocate();
  initialize(value);
}

template<class T>
LinearArray3<T>::LinearArray3( const LinearArray3& copy )
{
  resize(copy.dim1(), copy.dim2(), copy.dim3());

  long int size = getSize();
  for (long int idx = 0; idx < size; idx++) {
    objs[idx] = copy.objs[idx];
  }
}

template<class T>
LinearArray3<T>::~LinearArray3()
{
  if (objs) {
    delete[] objs;
  }
}

template<class T>
void LinearArray3<T>::allocate()
{
  if (dm1 && dm2 && dm3) {
    objs = new T[(dm1 * dm2 * dm3)];
  } else {
    objs = 0;
  }
}

template<class T>
void LinearArray3<T>::resize( int d1,
                              int d2,
                              int d3 )
{
  if (objs && (dm1 == d1) && (dm2 == d2) && (dm3 == d3)) {
    return;
  }

  dm1 = d1;
  dm2 = d2;
  dm3 = d3;

  if (objs) {
    delete[] objs;
  }

  allocate();
}

template<class T>
void LinearArray3<T>::copyData( const LinearArray3<T>& copy )
{
  long int size = getSize();
  ASSERTEQ(size, copy.getSize());
  ASSERTL3(dm1==copy.dm1 && dm2==copy.dm2 && dm3==copy.dm3);
  for (long int idx = 0; idx < size; idx++) {
    objs[idx] = copy.objs[idx];
  }
}

template<class T>
void LinearArray3<T>::initialize( const T& t )
{
  long int size = getSize();
  for (long int idx = 0; idx < size; idx++) {
    objs[idx] = t;
  }
}

// Explicit template instantiations:
template class LinearArray3<double> ;
template class LinearArray3<std::complex<double> > ;
template class LinearArray3<Uintah::Matrix3> ;
template class LinearArray3<SCIRun::Vector> ;

