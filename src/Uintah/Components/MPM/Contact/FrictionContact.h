// Friction.h

#ifndef __FRICTION_H__
#define __FRICTION_H__

#include <Uintah/Components/MPM/Contact/Contact.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>


namespace Uintah {
   class VarLabel;
   namespace MPM {

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

         // VarLabels specific to Frictional contact
	 const VarLabel* gNormTractionLabel;
	 const VarLabel* gSurfNormLabel; 
	 const VarLabel* gStressLabel; 
	 // const VarLabel* pStressLabel; 
	 //const VarLabel* pXLabel; 
	
	 
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
      
   } // end namespace MPM
} // end namespace Uintah

#endif /* __FRICTION_H__ */

// $Log$
// Revision 1.11  2000/06/17 07:06:37  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.10  2000/05/30 20:19:09  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.9  2000/05/26 22:05:40  jas
// Using Singleton class MPMLabel for label management.
//
// Revision 1.8  2000/05/26 21:37:35  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.7  2000/05/25 23:05:09  guilkey
// Created addComputesAndRequiresInterpolated and addComputesAndRequiresIntegrated
// for each of the three derived Contact classes.  Also, got the NullContact
// class working.  It doesn't do anything besides carry forward the data
// into the "MomExed" variable labels.
//
// Revision 1.6  2000/05/11 20:10:16  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.5  2000/05/08 18:42:46  guilkey
// Added an initializeContact function to all contact classes.  This is
// a null function for all but the FrictionContact.
//
// Revision 1.4  2000/05/05 22:37:27  bard
// Added frictional contact logic.  Compiles but doesn't yet work.
//
// Revision 1.3  2000/05/05 02:24:35  guilkey
// Added more stuff to FrictionContact, most of which is currently
// commented out until a compilation issue is resolved.
//
// Revision 1.2  2000/04/27 21:28:58  jas
// Contact is now created using a factory.
//
// Revision 1.1  2000/04/27 20:00:26  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
