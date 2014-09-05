#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <vector>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

#include <Packages/Uintah/CCA/Components/MPM/Fracture/Lattice.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/ParticlesNeighbor.h>

namespace Uintah {

  class Task;
  class Patch;
  class VarLabel;
  class MPMMaterial;

/**************************************

CLASS
   ConstitutiveModel
   
   Short description...

GENERAL INFORMATION

   ConstitutiveModel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Constitutive_Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class ConstitutiveModel {
      public:
	 
	 ConstitutiveModel();
	 virtual ~ConstitutiveModel();
	 
	 //////////
	 // Basic constitutive model calculations
	 virtual void computeStressTensor(const PatchSubset* patches,
					  const MPMMaterial* matl,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw) = 0;
	 
	 //////////
	 // Create space in data warehouse for CM data
	 virtual void initializeCMData(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw) = 0;

	 virtual void addComputesAndRequires(Task* task,
					   const MPMMaterial* matl,
					   const PatchSet* patches) const = 0;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to) = 0;

/*`==========TESTING==========*/ 
         double computeRhoMicroCM(double pressure, const MPMMaterial* matl);

         void computePressEOSCM(double rho_m, double& press_eos,
                                double& dp_drho, double& ss_new,
			        const MPMMaterial* matl);

	 double computeRhoMicro(double& press,double& gamma,
				        double& cv, double& Temp);
	 
	 void computePressEOS(double& rhoM, double& gamma,
				      double& cv, double& Temp,
				      double& press, double& dp_drho,
				      double& dp_de);
 /*==========TESTING==========`*/

        protected:

	 MPMLabel* lb;
      };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__

