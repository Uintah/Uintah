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

#ifndef __UINTAH_TANGENT_MODULUS_4D_H__
#define __UINTAH_TANGENT_MODULUS_4D_H__

// Headers
#include <Core/Util/Assert.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/Matrix3.h>

// Base class
#include <Core/Math/Tensor4D.h>

namespace Uintah {

  /*! \class TangentModulusTensor
   *  \brief Derived class for 4th order tensors that represent the tangent
   *         modulus \f$ C_{ijkl} \f$.
   *  \author  Biswajit Banerjee, 
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group
   */

  class TangentModulusTensor : public Tensor4D<double>
    {
    public:

      /** Constructor for tensor : Creates a 3x3x3x3 Tensor4D object */
      TangentModulusTensor();

      /** Convert a 6x6 matrix form of the tangent modulus to a 
          3x3x3x3 form */
      TangentModulusTensor(const FastMatrix& C_6x6);

      /** Copy constructor */
      TangentModulusTensor(const TangentModulusTensor& tensor);

      /** Destructor */
      virtual ~TangentModulusTensor();

      /** Assignment operator */
      TangentModulusTensor& operator=(const TangentModulusTensor& tensor); 
  
      /** Convert the 3x3x3x3 tangent modulus to a 6x6 matrix */
      void convertToVoigtForm(FastMatrix& C_6x6) const;

      /** Convert a 6x6 tangent modulus to a 3x3x3x3 matrix */
      void convertToTensorForm(const FastMatrix& C_6x6); 

      /*! \brief Contract a fourth order tensor with a second order tensor */ 
      void contract(const Matrix3& D, Matrix3& sigrate) const;

      /* Convert a 6x6 tangent modulus to a 3x3x3x3 matrix */
      void transformBy2ndOrderTensor(const Matrix3& F, double J);
    };

} // namespace Uintah

std::ostream& operator<<(std::ostream& out, 
                         const Uintah::TangentModulusTensor& C);

#endif //__UINTAH_TANGENT_MODULUS_4D_H__
