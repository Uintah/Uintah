#ifndef __HYPERELASTIC_EOS_MODEL_H__
#define __HYPERELASTIC_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class HyperElasticEOS
    \brief Hyperelastic relation for pressure from Simo and Hughes, 1998.

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

  class HyperElasticEOS : public MPMEquationOfState {

  private:

    // Prevent copying of this class
    // copy constructor
    //HyperElasticEOS(const HyperElasticEOS &cm);
    HyperElasticEOS& operator=(const HyperElasticEOS &cm);

  public:
    // constructors
    HyperElasticEOS(); // This constructor is used when there is
                             // no equation_of_state tag in the input
                             // file  ** WARNING **
    HyperElasticEOS(ProblemSpecP& ps); 
    HyperElasticEOS(const HyperElasticEOS* cm);
	 
    // destructor 
    virtual ~HyperElasticEOS();

    virtual void outputProblemSpec(ProblemSpecP& ps);
	 
    //////////
    // Calculate the pressure using a equation of state
    double computePressure(const MPMMaterial* matl,
			   const PlasticityState* state,
			   const Matrix3& deformGrad,
			   const Matrix3& rateOfDeformation,
			   const double& delT);
  
    double eval_dp_dJ(const MPMMaterial* matl,
                      const double& detF,
                      const PlasticityState* state);
  };

} // End namespace Uintah

#endif  // __HYPERELASTIC_EOS_MODEL_H__ 
