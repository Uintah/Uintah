#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <vector>
#include <Uintah/Components/MPM/MPMLabel.h>

namespace Uintah {
   class Task;
   class Patch;
   class VarLabel;
   namespace MPM {
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
	 virtual void computeStressTensor(const Patch* patch,
					  const MPMMaterial* matl,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw) = 0;
	 
	 //////////
	 // Computation of strain energy.  Useful for tracking energy balance.
	 virtual double computeStrainEnergy(const Patch* patch,
					    const MPMMaterial* matl,
					    DataWarehouseP& new_dw) = 0;
	 
	 //////////
	 // Create space in data warehouse for CM data
	 virtual void initializeCMData(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouseP& new_dw) = 0;

	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const = 0;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to) = 0;
	 /*
         const VarLabel* deltLabel;

         const VarLabel* pDeformationMeasureLabel;
         const VarLabel* pStressLabel;
         const VarLabel* pVolumeLabel;
         const VarLabel* pMassLabel;
         const VarLabel* pXLabel;

         const VarLabel* gMomExedVelocityLabel;
	 */

        protected:

	 MPMLabel* lb;
      };
      
   } // end namespace MPM
} // end namespace Uintah

// $Log$
// Revision 1.20  2000/07/05 23:43:33  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.19  2000/06/16 05:03:05  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.18  2000/06/15 21:57:05  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.17  2000/06/09 21:02:39  jas
// Added code to get the fudge factor directly into the constitutive model
// inititialization.
//
// Revision 1.16  2000/05/30 20:19:03  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.15  2000/05/26 21:37:34  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.14  2000/05/11 20:10:14  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.13  2000/05/07 06:02:04  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.12  2000/05/02 18:41:17  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.11  2000/05/02 17:54:25  sparker
// Implemented more of SerialMPM
//
// Revision 1.10  2000/05/01 17:25:00  jas
// Changed the var labels to be consistent with SerialMPM.
//
// Revision 1.9  2000/04/26 06:48:15  sparker
// Streamlined namespaces
//
// Revision 1.8  2000/04/21 01:22:56  guilkey
// Put the VarLabels which are common to all constitutive models in the
// base class.  The only one which isn't common is the one for the CMData.
//
// Revision 1.7  2000/04/14 02:19:41  jas
// Now using the ProblemSpec for input.
//
// Revision 1.6  2000/03/20 17:17:08  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.5  2000/03/17 09:29:34  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  2000/03/17 02:57:02  dav
// more namespace, cocoon, etc
//
// Revision 1.3  2000/03/15 20:05:56  guilkey
// Worked over the ConstitutiveModel base class, and the CompMooneyRivlin
// class to operate on all particles in a patch of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//

#endif  // __CONSTITUTIVE_MODEL_H__

