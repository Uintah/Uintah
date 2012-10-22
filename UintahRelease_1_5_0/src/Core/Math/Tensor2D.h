/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_TENSOR_2D_H
#define UINTAH_TENSOR_2D_H

// Headers
#include <Core/Util/Assert.h>

// Base class
#include <Core/Math/Tensor.h>

namespace Uintah {

  /*! \class Tensor2D
   *  \brief Templated derived class for 2nd order tensors.
   *  \author  Biswajit Banerjee, 
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group
  */

  template <class T>
    class Tensor2D : public Tensor<T>
    {
    public:

      /** Constructor for tensor */
      Tensor2D(int dim0, int dim1);

      /** Copy constructor */
      Tensor2D(const Tensor2D<T>& tensor);

      /** Destructor */
      virtual ~Tensor2D();

      /** Assignment operator */
      Tensor2D<T>& operator=(const Tensor2D<T>& tensor); 

      /** Element access operator */
      T& operator()(int dim0, int dim1) const;

    protected:
      
      // Offset
      int d_offset;
    };

// Implementation of Tensor2D

// Standard constructor : Create tensor of given size 
template <class T>
Tensor2D<T>::Tensor2D(int dim0, int dim1):Tensor<T>(dim0*dim1,2)
{
  d_dim[0] = dim0; d_dim[1] = dim1;
  d_offset = dim1;
}

// Copy constructor : Create tensor from given tensor
template <class T>
Tensor2D<T>::Tensor2D(const Tensor2D<T>& tensor):Tensor<T>(tensor),
                                                 d_offset(tensor.d_offset)
{
}

// Destructor
template <class T>
Tensor2D<T>::~Tensor2D()
{
}

// Assignment operator
template <class T>
inline Tensor2D<T>& 
Tensor2D<T>::operator=(const Tensor2D<T>& tensor)
{
  Tensor<T>::operator=(tensor);
  d_offset = tensor.d_offset;
  return *this;   
}

// Operator()
template <class T>
inline T&
Tensor2D<T>::operator()(int dim0, int dim1)
{
  ASSERT(!(dim0 < 0 || dim0 >= d_dim[0] || dim1 < 0 || dim1 >= d_dim[1]));
  return d_data[dim0*d_offset + dim1];
}

} // namespace Uintah

#endif // UINTAH_TENSOR_2D_H
