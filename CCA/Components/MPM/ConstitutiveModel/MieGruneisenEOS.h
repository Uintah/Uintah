#ifndef __MIE_GRUNEISEN_EOS_MODEL_H__
#define __MIE_GRUNEISEN_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*!
    \class MieGruneisenEOS
   
    \brief A Mie-Gruneisen type equation of state model
   
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2003 University of Utah \n

    Reference:
    
    Zocher, Maudlin, Chen, Flower-Maudlin, 2000,
    European Congress on Computational Methods in Applied Science 
    and Engineering, ECOMAS 2000, Barcelona)


    The equation of state is given by
    \f[
    p = \frac{\rho_0 C_0^2 \zeta 
              \left[1 + \left(1-\frac{\Gamma_0}{2}\right)\zeta\right]}
             {\left[1 - (S_{\alpha} - 1) \zeta\right]^2 + \Gamma_0 C_p T}
    \f]
    where 
    \f$ p\f$ = pressure \n
    \f$ C_0 \f$= bulk speed of sound \n
    \f$ \zeta = (\rho/\rho_0 - 1)\f$ \n
    where \f$\rho\f$ = current density \n
    \f$\rho_0\f$ = initial density \n
    \f$ E\f$ = internal energy = \f$C_p T\f$ \n
    where \f$C_p\f$ = specfic heat at constant pressure \n
    \f$T\f$ = temperature \n
    \f$\Gamma_0\f$ = Gruneisen's gamma at reference state \n
    \f$S_{\alpha}\f$ = linear Hugoniot slope coefficient 
  */
  ////////////////////////////////////////////////////////////////////////////

  class MieGruneisenEOS : public MPMEquationOfState {

    // Create datatype for storing model parameters
  public:
    struct CMData {
      double C_0;
      double Gamma_0;
      double S_alpha;
    };	 

  private:

    CMData d_const;
	 
    // Prevent copying of this class
    // copy constructor
    MieGruneisenEOS(const MieGruneisenEOS &cm);
    MieGruneisenEOS& operator=(const MieGruneisenEOS &cm);

  public:
    // constructors
    MieGruneisenEOS(ProblemSpecP& ps); 
	 
    // destructor 
    virtual ~MieGruneisenEOS();
	 
    /////////////////////////////////////////////////////////////////////////
    /*! Calculate the pressure using a equation of state */
    /////////////////////////////////////////////////////////////////////////
    virtual double computePressure(const MPMMaterial* matl,
				   const PlasticityState* state,
				   const Matrix3& deformGrad,
				   const Matrix3& rateOfDeformation,
				   const double& delT);
  
  };

} // End namespace Uintah

#endif  // __MIE_GRUNEISEN_EOS_MODEL_H__ 
