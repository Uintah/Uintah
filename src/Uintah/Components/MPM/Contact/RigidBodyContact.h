// RigidBody.h

#ifndef __RIGID_BODY_H_
#define __RIGID_BODY_H_

#include <Uintah/Components/MPM/Contact/Contact.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Task.h>



namespace Uintah {
   class VarLabel;
   namespace MPM {

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
      
   } // end namespace MPM
} // end namespace Uintah

#endif /* __RIGID_BODY_H_ */

// $Log$
// Revision 1.1  2001/01/11 03:31:31  guilkey
// Created new contact model for rigid bodies.
//
