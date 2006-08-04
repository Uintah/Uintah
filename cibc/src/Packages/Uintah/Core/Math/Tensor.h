#ifndef UINTAH_TENSOR_H
#define UINTAH_TENSOR_H

// Headers
#include <Core/Util/Assert.h>
#include <Packages/Uintah/Core/Math/StaticNumberArray.h>

namespace Uintah {

  /*! \class Tensor
   *  \brief Templated base class for tensors.
   *  \author  Biswajit Banerjee, 
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group

   Based on Tahoe implementation of TensorT.h .
  */

  template <class T>
    class Tensor 
    {
    public:

      /** Constructor for tensor of given size and rank */
      Tensor(int size, int rank);

      /** Copy constructor */
      Tensor(const Tensor<T>& tensor);

      /** Destructor */
      virtual ~Tensor();

      /** Get rank of tensor */
      int rank() const;
     
      /** Get the size of the tensor in the "dim" dimension
          starting from 0 */
      int dimension(int dim) const;
     
      /** Assignment operator */
      Tensor<T>& operator=(const Tensor<T>& tensor); 

    protected:
      
      StaticNumberArray<T> d_data;
      StaticNumberArray<int> d_dim;

    };

// Implementation of Tensor

// Standard constructor : Create tensor of given size 
template <class T>
Tensor<T>::Tensor(int size, int rank):d_data(size), d_dim(rank) 
{
}

// Copy constructor : Create tensor from given tensor
template <class T>
Tensor<T>::Tensor(const Tensor<T>& tens):d_data(tens.d_data),d_dim(tens.d_dim)
{
}

// Destructor
template <class T>
Tensor<T>::~Tensor()
{
}

// Get rank of tensor
template <class T>
inline int
Tensor<T>::rank() const
{
  return d_dim.size();
}

// Get size of tensor in the dim-th dimension
template <class T>
inline int
Tensor<T>::dimension(int dim) const
{
  return d_dim[dim];
}

// Assignment operator
template <class T>
inline Tensor<T>& 
Tensor<T>::operator=(const Tensor<T>& tensor)
{
  d_data = tensor.d_data;
  d_dim = tensor.d_dim;
  return *this;   
}


} // namespace Uintah

#endif
