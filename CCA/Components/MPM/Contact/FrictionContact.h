// Friction.h

#ifndef __FRICTION_H__
#define __FRICTION_H__

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>


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

      public:
	 // Constructor
	 FrictionContact(ProblemSpecP& ps, SimulationStateP& d_sS);
	 
	 // Destructor
	 virtual ~FrictionContact();

         // Initialiation function create storage for traction and surf. norm.
         virtual void initializeContact(const Patch* patch,
                                        int vfindex,
                                        DataWarehouseP& new_dw);
	 
	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw);
	 
	 virtual void exMomIntegrated(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);
	 
         virtual void addComputesAndRequiresInterpolated(Task* task,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const;

         virtual void addComputesAndRequiresIntegrated(Task* task,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const;
      };
} // End namespace Uintah
      

#endif /* __FRICTION_H__ */

