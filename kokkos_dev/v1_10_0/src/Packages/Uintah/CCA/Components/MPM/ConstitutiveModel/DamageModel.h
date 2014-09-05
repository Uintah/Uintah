#ifndef __DAMAGE_MODEL_H__
#define __DAMAGE_MODEL_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>


namespace Uintah {

/**************************************

CLASS
   DamageModel
   
   Short description...

GENERAL INFORMATION

   DamageModel.h

   Biswajit Banerjee
   Department of Mechanical Enegineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 McMurtry Container Dynamics Group

KEYWORDS
   Damage_Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class DamageModel {
      public:
	 
	 DamageModel();
	 virtual ~DamageModel();
	 
	 //////////
	 // Calculate the scalar damage parameter 
	 virtual double computeScalarDamage(const Matrix3& rateOfDeformation,
                                            const Matrix3& stress,
                                            const double& temperature,
                                            const double& delT,
                                            const MPMMaterial* matl,
                                            const double& tolerance,
                                            const double& damage_old) = 0;

      };
} // End namespace Uintah
      


#endif  // __DAMAGE_MODEL_H__

