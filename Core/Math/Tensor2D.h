#ifndef UINTAH_TENSOR_2D_H
#define UINTAH_TENSOR_2D_H

// Headers
#include <Core/Util/Assert.h>

// Base class
#include <Packages/Uintah/Core/Math/Tensor.h>

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
