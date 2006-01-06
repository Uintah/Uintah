#ifndef __NP_SHEAR_MODEL_H__
#define __NP_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class NPShear
   *  \brief The shear modulus model given by Nadal and LePoac
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class NPShear : public ShearModulusModel {

  private:

    double d_mu0;    // Material constant 
    double d_zeta;   // Material constant 
    double d_slope_mu_p_over_mu0; // Material constant
    double d_C;      // Material constant
    double d_m;      // atomic mass

    NPShear& operator=(const NPShear &smm);

  public:
	 
    /*! Construct a constant shear modulus model. */
    NPShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    NPShear(const NPShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~NPShear();
	 
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __NP_SHEAR_MODEL_H__

