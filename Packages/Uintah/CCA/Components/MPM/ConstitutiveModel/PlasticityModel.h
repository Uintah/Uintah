#ifndef __PLASTICITY_MODEL_H__
#define __PLASTICITY_MODEL_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>


namespace Uintah {

/**************************************

CLASS
   PlasticityModel
   
   Short description...

GENERAL INFORMATION

   PlasticityModel.h

   Biswajit Banerjee
   Department of Mechanical Enegineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 McMurtry Container Dynamics Group

KEYWORDS
   Plasticity_Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class PlasticityModel {
      public:
	 
	 PlasticityModel();
	 virtual ~PlasticityModel();
	 
	 //////////
	 // Calculate the flow stress
         virtual double computeFlowStress(const Matrix3& rateOfDeformation,
                                          const Matrix3& stress,
                                          const double& temperature,
                                          const double& delT,
                                          const MPMMaterial* matl,
                                          const double& tolerance,
                                          double& plasticStrain) = 0;

      };
} // End namespace Uintah
      


#endif  // __PLASTICITY_MODEL_H__

