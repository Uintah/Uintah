#ifndef __DAMAGE_MODEL_H__
#define __DAMAGE_MODEL_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>


namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class DamageModel
    \brief Abstract base class for damage models   
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2002 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class DamageModel {
  public:
	 
    DamageModel();
    virtual ~DamageModel();
	 
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Initialize the damage parameter in the calling function
    */
    //////////////////////////////////////////////////////////////////////////
    virtual double initialize() = 0;

    //////////////////////////////////////////////////////////////////////////
    /*! 
      Determine if damage has crossed cut off
    */
    //////////////////////////////////////////////////////////////////////////
    virtual bool hasFailed(double damage) = 0;
    
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Calculate the scalar damage parameter 
    */
    //////////////////////////////////////////////////////////////////////////
    virtual double computeScalarDamage(const double& plasticStrainRate,
				       const Matrix3& stress,
				       const double& temperature,
				       const double& delT,
				       const MPMMaterial* matl,
				       const double& tolerance,
				       const double& damage_old) = 0;

  };
} // End namespace Uintah
      


#endif  // __DAMAGE_MODEL_H__

