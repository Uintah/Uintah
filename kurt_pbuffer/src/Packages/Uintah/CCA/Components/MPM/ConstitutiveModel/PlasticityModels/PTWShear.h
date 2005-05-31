#ifndef __PTW_SHEAR_MODEL_H__
#define __PTW_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class PTWShear
   *  \brief The shear modulus model used by Preston,Tonks,Wallace
   *         the PTW plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class PTWShear : public ShearModulusModel {

  private:

    double d_mu0;     // Material constant 
    double d_alpha;   // Material constant 
    double d_alphap;  // Material constant 
    double d_slope_mu_p_over_mu0; // Material constant (constant A in SCG model)

    PTWShear& operator=(const PTWShear &smm);

  public:
	 
    /*! Construct a constant shear modulus model. */
    PTWShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    PTWShear(const PTWShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~PTWShear();
	 
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __PTW_SHEAR_MODEL_H__

