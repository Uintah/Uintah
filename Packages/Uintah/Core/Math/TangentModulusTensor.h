#ifndef __UINTAH_TANGENT_MODULUS_4D_H__
#define __UINTAH_TANGENT_MODULUS_4D_H__

// Headers
#include <Core/Util/Assert.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>

// Base class
#include <Packages/Uintah/Core/Math/Tensor4D.h>

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
      void convertToVoigtForm(FastMatrix& C_6x6);

      /** Convert a 6x6 tangent modulus to a 3x3x3x3 matrix */
      void convertToTensorForm(const FastMatrix& C_6x6); 
    };

} // namespace Uintah

#endif //__UINTAH_TANGENT_MODULUS_4D_H__
