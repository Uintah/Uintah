#ifndef __CONTACT_H__
#define __CONTACT_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/MPMLabel.h>

#include <math.h>


namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;
   namespace MPM {
     class MPMMaterial;

/**************************************

CLASS
   Contact
   
   Short description...

GENERAL INFORMATION

   Contact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

      class Contact {
      public:
         // Constructor
	 Contact();

	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw) = 0;
	 
	 virtual void exMomIntegrated(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) = 0;

	 virtual void initializeContact(const Patch* patch,
					int vfindex,
					DataWarehouseP& new_dw) = 0;

         virtual void addComputesAndRequiresInterpolated(Task* task,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const = 0;
	 
         virtual void addComputesAndRequiresIntegrated(Task* task,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const = 0;


         // VarLabels common to all contact models go here
	 /*
         const VarLabel* deltLabel;
         const VarLabel* gMassLabel;
         const VarLabel* gAccelerationLabel;
         const VarLabel* gMomExedAccelerationLabel;
         const VarLabel* gVelocityLabel;
         const VarLabel* gMomExedVelocityLabel;
         const VarLabel* gVelocityStarLabel;
         const VarLabel* gMomExedVelocityStarLabel;
	 */

      protected:
	 MPMLabel* lb;
      };
      
      inline bool compare(double num1, double num2)
	 {
	    double EPSILON=1.e-8;
	    
	    return (fabs(num1-num2) <= EPSILON);
	 }
      
      
   } // end namespace MPM
} // end namespace Uintah
   
// $Log$
// Revision 1.19  2000/07/05 23:43:35  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.18  2000/06/17 07:06:37  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.17  2000/05/30 20:19:08  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.16  2000/05/26 21:37:34  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.15  2000/05/25 23:05:06  guilkey
// Created addComputesAndRequiresInterpolated and addComputesAndRequiresIntegrated
// for each of the three derived Contact classes.  Also, got the NullContact
// class working.  It doesn't do anything besides carry forward the data
// into the "MomExed" variable labels.
//
// Revision 1.14  2000/05/15 19:39:41  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.13  2000/05/11 20:10:16  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.12  2000/05/08 18:42:46  guilkey
// Added an initializeContact function to all contact classes.  This is
// a null function for all but the FrictionContact.
//
// Revision 1.11  2000/05/02 18:41:18  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.10  2000/05/02 17:54:27  sparker
// Implemented more of SerialMPM
//
// Revision 1.9  2000/05/02 06:07:14  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.8  2000/04/27 20:00:25  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
// Revision 1.7  2000/04/26 06:48:20  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/25 22:57:30  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.5  2000/04/12 16:57:27  guilkey
// Converted the SerialMPM.cc to have multimaterial/multivelocity field
// capabilities.  Tried to guard all the functions against breaking the
// compilation, but then who really cares?  It's not like sus has compiled
// for more than 5 minutes in a row for two months.
//
// Revision 1.4  2000/03/21 01:29:41  dav
// working to make MPM stuff compile successfully
//
// Revision 1.3  2000/03/20 23:50:44  dav
// renames SingleVel to SingleVelContact
//
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//

#endif // __CONTACT_H__

