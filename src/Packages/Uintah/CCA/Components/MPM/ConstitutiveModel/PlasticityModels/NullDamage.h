#ifndef __NULL_DAMAGE_MODEL_H__
#define __NULL_DAMAGE_MODEL_H__


#include "DamageModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class NullDamage
    \brief Default Damage Model (no damage)
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2007 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class NullDamage : public DamageModel {

  public:

  private:

    // Prevent copying of this class
    // copy constructor
    //NullDamage(const NullDamage &cm);
    NullDamage& operator=(const NullDamage &cm);

  public:
    // constructors
    NullDamage(); 
    NullDamage(ProblemSpecP& ps); 
    NullDamage(const NullDamage* cm);
	 
    // destructor 
    virtual ~NullDamage();

    virtual void outputProblemSpec(ProblemSpecP& ps);
	 
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
    virtual double computeScalarDamage(const double& plasticStrainRate,
				       const Matrix3& stress,
				       const double& temperature,
				       const double& delT,
				       const MPMMaterial* matl,
				       const double& tolerance,
				       const double& damage_old);
  
  };

} // End namespace Uintah

#endif  // __NULL_DAMAGE_MODEL_H__ 
