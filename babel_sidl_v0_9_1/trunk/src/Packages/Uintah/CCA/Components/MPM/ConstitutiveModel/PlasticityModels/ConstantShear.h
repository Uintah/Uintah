#ifndef __CONSTANT_SHEAR_MODEL_H__
#define __CONSTANT_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class ConstantShear
   *  \brief The shear modulus does not vary with density and temperature
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class ConstantShear : public ShearModulusModel {

  private:
    ConstantShear& operator=(const ConstantShear &smm);

  public:
	 
    /*! Construct a constant shear modulus model. */
    ConstantShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    ConstantShear(const ConstantShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~ConstantShear();
	 
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __CONSTANT_SHEAR_MODEL_H__

