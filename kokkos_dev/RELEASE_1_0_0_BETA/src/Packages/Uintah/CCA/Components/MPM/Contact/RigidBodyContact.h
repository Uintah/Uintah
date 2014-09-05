// RigidBody.h

#ifndef __RIGID_BODY_H_
#define __RIGID_BODY_H_

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Task.h>

namespace Uintah {

class VarLabel;

/**************************************

CLASS
   RigidBodyContact
   
   Short description...

GENERAL INFORMATION

   RigidBodyContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Rigid_Body

DESCRIPTION
  One of the derived Contact classes.  This particular
  class contains methods for recapturing single velocity
  field behavior from objects belonging to multiple velocity
  fields.  The main purpose of this type of contact is to
  ensure that one can get the same answer using prescribed
  contact as can be gotten using "automatic" contact.
  
WARNING
  
****************************************/

      class RigidBodyContact : public Contact {
      private:
	 
	 // Prevent copying of this class
	 // copy constructor
	 RigidBodyContact(const RigidBodyContact &con);
	 RigidBodyContact& operator=(const RigidBodyContact &con);
	 
	 SimulationStateP d_sharedState;
	 double d_stop_time;
	 
      public:
	 // Constructor
	 RigidBodyContact(ProblemSpecP& ps, SimulationStateP& d_sS);
	 
	 // Destructor
	 virtual ~RigidBodyContact();

	 // Initialiation function, empty for RigidBodyContact Velocity contact
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
      
} // end namespace Uintah

#endif /* __RIGID_BODY_H_ */
