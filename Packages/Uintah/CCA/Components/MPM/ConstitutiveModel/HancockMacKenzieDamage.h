#ifndef __HANCOCKMACKENZIE_DAMAGE_MODEL_H__
#define __HANCOCKMACKENZIE_DAMAGE_MODEL_H__


#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class HancockMacKenzieDamage
    \brief HancockMacKenzie Damage Model
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2002 University of Utah

    References:
    1) Hancock and MacKenzie, 1976, Int. J. Mech. Phys. Solids, 24, 147-169.

    The damage evolution rule is given by \n
    \f$
    \dot{D} = (1/1.65)\dot{\epsilon_p}\exp(3\sigma_h/2\sigma_{eq})
    \f$ \n
    where \n
    \f$ D \f$ = damage variable \n
    where \f$ D = 0 \f$ for virgin material, \n
    \f$ \epsilon_p \f$ is the plastic strain, \n
    \f$ \sigma_h = (1/3) Tr(\sigma) \n
    \f$ \sigma_{eq} = \sqrt{(3/2) \sigma_{dev}:\sigma_{dev}}
  */
  /////////////////////////////////////////////////////////////////////////////

  class HancockMacKenzieDamage : public DamageModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double D0; /*< Initial damage */
      double Dc; /*< Critical damage */
    };	 

  private:

    CMData d_initialData;
	 
    // Prevent copying of this class
    // copy constructor
    HancockMacKenzieDamage(const HancockMacKenzieDamage &cm);
    HancockMacKenzieDamage& operator=(const HancockMacKenzieDamage &cm);

  public:
    // constructors
    HancockMacKenzieDamage(ProblemSpecP& ps); 
	 
    // destructor 
    virtual ~HancockMacKenzieDamage();
	 
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
  };

} // End namespace Uintah

#endif  // __HANCOCKMACKENZIE_DAMAGE_MODEL_H__ 
