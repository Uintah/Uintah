#ifndef __MTS_SHEAR_MODEL_H__
#define __MTS_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class MTSShear
   *  \brief The shear modulus model used by Folansbee and Kocks in 
   *         the MTS plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class MTSShear : public ShearModulusModel {

  private:

    double d_mu0; // Material constant (also in MTS model)
    double d_D;   // Material constant (also in MTS model)
    double d_T0;  // Material constant (also in MTS model)

    MTSShear& operator=(const MTSShear &smm);

  public:
	 
    /*! Construct a constant shear modulus model. */
    MTSShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    MTSShear(const MTSShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~MTSShear();
	 
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __MTS_SHEAR_MODEL_H__

