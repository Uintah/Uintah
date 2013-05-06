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

#include <CCA/Components/MD/SPMEArray3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/Matrix3.h>
#include <Core/Util/Assert.h>

#include <sci_values.h>
#include <sci_defs/fftw_defs.h>

using namespace SCIRun;

namespace Uintah {

template<class T>
SPMEArray3<T>::SPMEArray3()
{
  objs = 0;
  dm1 = dm2 = dm3 = 0;
}

template<class T>
SPMEArray3<T>::SPMEArray3(int dm1,
                          int dm2,
                          int dm3) :
    dm1(dm1), dm2(dm2), dm3(dm3)
{
  allocate();
}

template<class T>
SPMEArray3<T>::~SPMEArray3()
{
  if (objs) {
    delete objs;
  }
}

template<class T>
void SPMEArray3<T>::allocate()
{
  if (dm1 && dm2 && dm3) {
    objs = new T[(dm1 * dm2 * dm3 * sizeof(T))];
  } else {
    objs = 0;
  }
}

template<class T>
void SPMEArray3<T>::resize(int d1,
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
void SPMEArray3<T>::initialize(const T& t)
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
void SPMEArray3<T>::copy(const SPMEArray3<T> &other)
{
  resize(other.dim1(), other.dim2(), other.dim3());
  for (int i = 0; i < dm1; i++) {
    for (int j = 0; j < dm2; j++) {
      for (int k = 0; k < dm3; k++) {
        int idx = (i) + ((j) * dm1) + ((k) * dm1 * dm2);
        objs[idx] = other.objs[idx];
      }
    }
  }
}

//// Explicit template instantiations:
template class SPMEArray3<dblcomplex> ;
template class SPMEArray3<Uintah::Matrix3> ;
template class SPMEArray3<double> ;
template class SPMEArray3<SCIRun::Vector> ;

}  // End namespace Uintah

