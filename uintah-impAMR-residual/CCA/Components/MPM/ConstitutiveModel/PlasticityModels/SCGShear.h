#ifndef __SCG_SHEAR_MODEL_H__
#define __SCG_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class SCGShear
   *  \brief The shear modulus model used by Steinberg,Cochran,Guinan in
   *         the SCG plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class SCGShear : public ShearModulusModel {

  private:

    double d_mu0; // Material constant (also in SCG model)
    double d_A;   // Material constant (also in SCG model)
    double d_B;   // Material constant (also in SCG model)

    SCGShear& operator=(const SCGShear &smm);

  public:
	 
    /*! Construct a constant shear modulus model. */
    SCGShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    SCGShear(const SCGShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~SCGShear();
	 
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __SCG_SHEAR_MODEL_H__

