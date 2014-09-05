// Approach.h

#ifndef __APPROACH_H__
#define __APPROACH_H__

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>


namespace Uintah {
/**************************************

CLASS
   ApproachContact
   
   Short description...

GENERAL INFORMATION

   ApproachContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Approach

DESCRIPTION
  One of the derived Contact classes.  This particular
  version is used to apply Coulombic frictional contact.
  
WARNING
  
****************************************/

      class ApproachContact : public Contact {
      private:
	 
	 // Prevent copying of this class
	 // copy constructor
	 ApproachContact(const ApproachContact &con);
	 ApproachContact& operator=(const ApproachContact &con);
	 
	 SimulationStateP d_sharedState;

         // Coefficient of friction
         double d_mu;
         // Nodal volume fraction that must occur before contact is applied
         double d_vol_const;
         int d_8or27;
         int NGP;
         int NGN;

      public:
	 // Constructor
	 ApproachContact(ProblemSpecP& ps, SimulationStateP& d_sS,MPMLabel* lb,
                                                                   int n8or27);
	 
	 // Destructor
	 virtual ~ApproachContact();

	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw);
	 
	 virtual void exMomIntegrated(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw);
	 
         virtual void addComputesAndRequiresInterpolated(Task* task,
					     const PatchSet* patches,
					     const MaterialSet* matls) const;

         virtual void addComputesAndRequiresIntegrated(Task* task,
					     const PatchSet* patches,
					     const MaterialSet* matls) const;
      };
} // End namespace Uintah
      

#endif /* __APPROACH_H__ */

