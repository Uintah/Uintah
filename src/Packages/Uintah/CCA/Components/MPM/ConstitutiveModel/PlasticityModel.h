#ifndef __PLASTICITY_MODEL_H__
#define __PLASTICITY_MODEL_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>


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

      private:

      public:
	 
	 PlasticityModel();
	 virtual ~PlasticityModel();
	 
         // Computes and requires for internal evolution variables
	 virtual void addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* patches) const = 0;

	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const PatchSet* patches) const = 0;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to) = 0;

         virtual void initializeInternalVars(ParticleSubset* pset,
				             DataWarehouse* new_dw) = 0;

         virtual void getInternalVars(ParticleSubset* pset,
                                      DataWarehouse* old_dw) = 0;

         virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                                 DataWarehouse* new_dw) = 0; 

         virtual void updateElastic(const particleIndex idx) = 0;

         virtual void updatePlastic(const particleIndex idx, const double& delGamma) = 0;

	 //////////
	 // Calculate the flow stress
         virtual double computeFlowStress(const Matrix3& rateOfDeformation,
                                          const Matrix3& stress,
                                          const double& temperature,
                                          const double& delT,
                                          const double& tolerance,
                                          const MPMMaterial* matl,
                                          const particleIndex idx) = 0;
      };
} // End namespace Uintah
      


#endif  // __PLASTICITY_MODEL_H__

