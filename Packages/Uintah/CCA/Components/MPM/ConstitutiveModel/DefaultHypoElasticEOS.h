#ifndef __DEFAULT_HYPOELASTIC_EOS_MODEL_H__
#define __DEFAULT_HYPOELASTIC_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class DefaultHypoElasticEOS
    \brief Not really an equation of state but just an isotropic
    hypoelastic pressure calculator based on bulk modulus.
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2002-2003 University of Utah

    The equation of state is given by
    \f[
    p = Tr(D) K \Delta T
    \f]
    where \n
    \f$p\f$ = pressure\n
    \f$D\f$ = rate of deformation tensor\n
    \f$K\f$ = bulk modulus\n
    \f$\Delta T\f$ = time increment
  */
  ////////////////////////////////////////////////////////////////////////////

  class DefaultHypoElasticEOS : public MPMEquationOfState {

  private:

    // Prevent copying of this class
    // copy constructor
    DefaultHypoElasticEOS(const DefaultHypoElasticEOS &cm);
    DefaultHypoElasticEOS& operator=(const DefaultHypoElasticEOS &cm);

  public:
    // constructors
    DefaultHypoElasticEOS(ProblemSpecP& ps); 
	 
    // destructor 
    virtual ~DefaultHypoElasticEOS();
	 
    //////////
    // Calculate the pressure using a equation of state
    double computePressure(const MPMMaterial* matl,
			   const PlasticityState* state,
			   const Matrix3& deformGrad,
			   const Matrix3& rateOfDeformation,
			   const double& delT);
  
  };

} // End namespace Uintah

#endif  // __DEFAULT_HYPOELASTIC_EOS_MODEL_H__ 
