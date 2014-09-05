#ifndef __JOHNSONCOOK_DAMAGE_MODEL_H__
#define __JOHNSONCOOK_DAMAGE_MODEL_H__


#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class JohnsonCookDamage
    \brief Johnson-Cook Damage Model
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2002 University of Utah

    References:
    1) Johnson and Cook, 1985, Int. J. Eng. Fracture Mech., 21, 31-48.

    The damage evolution rule is given by \n
    \f$
    \dot{D} = \dot{\epsilon_p}/\epsilon_p^f
    \f$ \n
    where \n
    \f$ D \f$ = damage variable \n
    where \f$ D \f$ = 0 for virgin material, 
    \f$ D \f$ = 1 for fracture \n
    \f$ \epsislon_p^f\f$  = value of fracture strain given by \n
    \f$ 
    \epsilon_p^f = (D1 + D2 \exp (D3 \sigma*)][1+\dot{p}^*]^(D4)[1+D5 T^*
    \f$ \n 
    where \f$ \sigma^*= 1/3*trace(\sigma)/\sigma_{eq} \f$ \n
    \f$  D1, D2, D3, D4, D5\f$  are constants \n
    \f$  T^* = (T-T_{room})/(T_{melt}-T_{room}) \f$
  */
  /////////////////////////////////////////////////////////////////////////////

  class JohnsonCookDamage : public DamageModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double D0; /*< Initial damage */
      double Dc; /*< Critical damage */
      double D1;
      double D2;
      double D3;
      double D4;
      double D5;
    };	 

  private:

    CMData d_initialData;
	 
    // Prevent copying of this class
    // copy constructor
    JohnsonCookDamage(const JohnsonCookDamage &cm);
    JohnsonCookDamage& operator=(const JohnsonCookDamage &cm);

  public:
    // constructors
    JohnsonCookDamage(ProblemSpecP& ps); 
	 
    // destructor 
    virtual ~JohnsonCookDamage();
	 
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Initialize the damage parameter in the calling function
    */
    //////////////////////////////////////////////////////////////////////////
    double initialize();

    //////////////////////////////////////////////////////////////////////////
    /*! 
      Determine if damage has crossed cut off
    */
    //////////////////////////////////////////////////////////////////////////
    bool hasFailed(double damage);
    
    //////////
    // Calculate the scalar damage parameter 
    virtual double computeScalarDamage(const Matrix3& rateOfDeformation,
				       const Matrix3& stress,
				       const double& temperature,
				       const double& delT,
				       const MPMMaterial* matl,
				       const double& tolerance,
				       const double& damage_old);
  
  protected:

    double calcStrainAtFracture(const Matrix3& sig, 
				const double& epdot,
				const double& T,
				const MPMMaterial* matl,
				const double& tolerance);
  };

} // End namespace Uintah

#endif  // __JOHNSONCOOK_DAMAGE_MODEL_H__ 
