// Friction.h

#ifndef __FRICTION_H__
#define __FRICTION_H__

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
   FrictionContact
   
   Short description...

GENERAL INFORMATION

   FrictionContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Friction

DESCRIPTION
  One of the derived Contact classes.  This particular
  version is used to apply Coulombic frictional contact.
  
WARNING
  
****************************************/

      class FrictionContact : public Contact {
      private:
	 
	 // Prevent copying of this class
	 // copy constructor
	 FrictionContact(const FrictionContact &con);
	 FrictionContact& operator=(const FrictionContact &con);
	 
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
	 FrictionContact(ProblemSpecP& ps, SimulationStateP& d_sS,MPMLabel* lb,
                                                                   int n8or27);
	 
	 // Destructor
	 virtual ~FrictionContact();

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
      

#endif /* __FRICTION_H__ */

