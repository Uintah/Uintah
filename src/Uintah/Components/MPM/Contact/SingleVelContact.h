// SingleVel.h

#ifndef __SINGLE_VEL_H__
#define __SINGLE_VEL_H__

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
   SingleVelContact
   
   Short description...

GENERAL INFORMATION

   SingleVelContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Single_Velocity

DESCRIPTION
  One of the derived Contact classes.  This particular
  class contains methods for recapturing single velocity
  field behavior from objects belonging to multiple velocity
  fields.  The main purpose of this type of contact is to
  ensure that one can get the same answer using prescribed
  contact as can be gotten using "automatic" contact.
  
WARNING
  
****************************************/

      class SingleVelContact : public Contact {
      private:
	 
	 // Prevent copying of this class
	 // copy constructor
	 SingleVelContact(const SingleVelContact &con);
	 SingleVelContact& operator=(const SingleVelContact &con);
	 
	 SimulationStateP d_sharedState;
	 
      public:
	 // Constructor
	 SingleVelContact(ProblemSpecP& ps, SimulationStateP& d_sS);
	 
	 // Destructor
	 virtual ~SingleVelContact();

	 // Initialiation function, empty for Single Velocity contact
	 virtual void initializeContact(const Region* region,
					int vfindex,
					DataWarehouseP& new_dw);
	 
	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorContext*,
					const Region* region,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw);
	 
	 virtual void exMomIntegrated(const ProcessorContext*,
				      const Region* region,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

         virtual void addComputesAndRequiresInterpolated(Task* task,
                                             const MPMMaterial* matl,
                                             const Region* region,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const;	 
         virtual void addComputesAndRequiresIntegrated(Task* task,
                                             const MPMMaterial* matl,
                                             const Region* region,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const;	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif /* __SINGLE_VEL_H__ */

// $Log$
// Revision 1.10  2000/05/26 21:37:35  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.9  2000/05/25 23:05:11  guilkey
// Created addComputesAndRequiresInterpolated and addComputesAndRequiresIntegrated
// for each of the three derived Contact classes.  Also, got the NullContact
// class working.  It doesn't do anything besides carry forward the data
// into the "MomExed" variable labels.
//
// Revision 1.8  2000/05/11 20:10:17  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.7  2000/05/08 18:42:47  guilkey
// Added an initializeContact function to all contact classes.  This is
// a null function for all but the FrictionContact.
//
// Revision 1.6  2000/04/27 21:28:58  jas
// Contact is now created using a factory.
//
// Revision 1.5  2000/04/27 20:00:26  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
// Revision 1.4  2000/04/26 06:48:21  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/04/25 22:57:31  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.2  2000/04/20 23:21:02  dav
// updated to match Contact.h
//
// Revision 1.1  2000/03/20 23:50:44  dav
// renames SingleVel to SingleVelContact
//
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:14  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
