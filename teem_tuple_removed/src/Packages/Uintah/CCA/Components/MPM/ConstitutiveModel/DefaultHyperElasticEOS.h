#ifndef __DEFAULT_HYPERELASTIC_EOS_MODEL_H__
#define __DEFAULT_HYPERELASTIC_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class DefaultHyperElasticEOS
    \brief Not really an equation of state but just an isotropic
    hyperelastic pressure calculator based on bulk modulus
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2002-2003 University of Utah

    The equation of state is given by
    \f[
    p = 0.5 K (J - \frac{1}{J})
    \f]
    where,\n
    \f$ K \f$ is the bulk modulus \n
    \f$ J \f$ is the Jacobian
  */
  ////////////////////////////////////////////////////////////////////////////

  class DefaultHyperElasticEOS : public MPMEquationOfState {

    // Create datatype for storing model parameters
  public:

  private:

    // Prevent copying of this class
    // copy constructor
    DefaultHyperElasticEOS(const DefaultHyperElasticEOS &cm);
    DefaultHyperElasticEOS& operator=(const DefaultHyperElasticEOS &cm);

  public:
    // constructors
    DefaultHyperElasticEOS(ProblemSpecP& ps); 
	 
    // destructor 
    virtual ~DefaultHyperElasticEOS();
	 
    //////////
    // Calculate the pressure using a equation of state
    double computePressure(const MPMMaterial* matl,
			   const PlasticityState* state,
			   const Matrix3& deformGrad,
			   const Matrix3& rateOfDeformation,
			   const double& delT);
  
  };

} // End namespace Uintah

#endif  // __DEFAULT_HYPERELASTIC_EOS_MODEL_H__ 
