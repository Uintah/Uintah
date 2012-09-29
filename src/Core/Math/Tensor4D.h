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

#ifndef UINTAH_TENSOR_4D_H
#define UINTAH_TENSOR_4D_H

// Headers
#include <Core/Util/Assert.h>
#include <iostream>

// Base class
#include <Core/Math/Tensor.h>

namespace Uintah {

  /*! \class Tensor4D
   *  \brief Templated derived class for 4th order tensors.
   *  \author  Biswajit Banerjee, 
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group
   */

  template <class T>
    class Tensor4D : public Tensor<T>
    {
    public:

      /** Constructor for tensor */
      Tensor4D(int dim0, int dim1, int dim2, int dim3);

      /** Copy constructor */
      Tensor4D(const Tensor4D<T>& tensor);

      /** Destructor */
      virtual ~Tensor4D();

      /** Assignment operator */
      Tensor4D<T>& operator=(const Tensor4D<T>& tensor); 

      /** Element modification operator */
      T& operator()(int dim0, int dim1, int dim2, int dim3);

      /** Element access operator */
      T operator()(int dim0, int dim1, int dim2, int dim3) const;

    protected:
      
      // Offset
      int d_offset0;
      int d_offset1;
      int d_offset2;
    };

  // Implementation of Tensor4D

  // Standard constructor : Create tensor of given size 
  template <class T>
    Tensor4D<T>::Tensor4D(int dim0, int dim1, int dim2, int dim3):
    Tensor<T>(dim0*dim1*dim2*dim3,4)
    {
      this->d_dim[0] = dim0; this->d_dim[1] = dim1; this->d_dim[2] = dim2; this->d_dim[3] = dim3;
      d_offset0 = dim1*dim2*dim3;
      d_offset1 = dim2*dim3;
      d_offset2 = dim3;
    }

  // Copy constructor : Create tensor from given tensor
  template <class T>
    Tensor4D<T>::Tensor4D(const Tensor4D<T>& tensor):Tensor<T>(tensor),
    d_offset0(tensor.d_offset0), 
    d_offset1(tensor.d_offset1), 
    d_offset2(tensor.d_offset2) 
    {
    }

  // Destructor
  template <class T>
    Tensor4D<T>::~Tensor4D()
    {
    }

  // Assignment operator
  template <class T>
    inline Tensor4D<T>& 
    Tensor4D<T>::operator=(const Tensor4D<T>& tensor)
    {
      Tensor<T>::operator=(tensor);
      d_offset0 = tensor.d_offset0;
      d_offset1 = tensor.d_offset1;
      d_offset2 = tensor.d_offset2;
      return *this;   
    }

  // Operator()
  template <class T>
    inline T&
    Tensor4D<T>::operator()(int dim0, int dim1, int dim2, int dim3) 
    {
      ASSERT(!(dim0 < 0 || dim0 >= this->d_dim[0] || dim1 < 0 || dim1 >= this->d_dim[1] ||
	       dim2 < 0 || dim2 >= this->d_dim[2] || dim3 < 0 || dim3 >= this->d_dim[3]));
      return this->d_data[dim0*d_offset0 + dim1*d_offset1 + dim2*d_offset2 + dim3];
    }

  // Operator()
  template <class T>
    inline T
    Tensor4D<T>::operator()(int dim0, int dim1, int dim2, int dim3) const
    {
      ASSERT(!(dim0 < 0 || dim0 >= this->d_dim[0] || dim1 < 0 || dim1 >= this->d_dim[1] ||
	       dim2 < 0 || dim2 >= this->d_dim[2] || dim3 < 0 || dim3 >= this->d_dim[3]));
      return this->d_data[dim0*d_offset0 + dim1*d_offset1 + dim2*d_offset2 + dim3];
    }

} // namespace Uintah

#endif // UINTAH_TENSOR_4D_H
