#ifndef __COMPMOONRIV_CONSTITUTIVE_MODEL_H__
#define __COMPMOONRIV_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {
   namespace MPM {
/**************************************

CLASS
   CompMooneyRivlin
   
   Short description...

GENERAL INFORMATION

   CompMooneyRivlin.h

   Author?
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Comp_Mooney_Rivlin

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class CompMooneyRivlin : public ConstitutiveModel {
	 // Create datatype for storing model parameters
      public:
	 struct CMData {
	    double C1;
	    double C2;
	    double PR;
	 };
      private:
	 CMData d_initialData;
	 
	 // Prevent copying of this class
	 // copy constructor
	 CompMooneyRivlin(const CompMooneyRivlin &cm);
	 CompMooneyRivlin& operator=(const CompMooneyRivlin &cm);
	 
      public:
	 // constructor
	 CompMooneyRivlin(ProblemSpecP& ps);
	 
	 // destructor 
	 virtual ~CompMooneyRivlin();
	 
	 // compute stable timestep for this patch
	 virtual void computeStableTimestep(const Patch* patch,
					    const MPMMaterial* matl,
					    DataWarehouseP& new_dw);
	 
	 // compute stress at each particle in the patch
	 virtual void computeStressTensor(const Patch* patch,
					  const MPMMaterial* matl,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw);
	 
	 // compute total strain energy for all particles in the patch
	 virtual double computeStrainEnergy(const Patch* patch,
					    const MPMMaterial* matl,
					    DataWarehouseP& new_dw);
	 
	 // initialize  each particle's constitutive model data
	 virtual void initializeCMData(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouseP& new_dw);
	 
	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const;

         //for fracture
         virtual void computeCrackSurfaceContactForce(const Patch* patch,
                                           const MPMMaterial* mpm_matl,
                                           DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw);

	 virtual void addComputesAndRequiresForCrackSurfaceContact(
	                                     Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);
      };

   }
}

#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

// $Log$
// Revision 1.27  2000/10/11 01:30:28  guilkey
// Made CMData no longer a per particle variable for these models.
// None of them currently have anything worthy of being called StateData,
// so no such struct was created.
//
// Revision 1.26  2000/09/12 16:52:10  tan
// Reorganized crack surface contact force algorithm.
//
// Revision 1.25  2000/07/05 23:43:33  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.24  2000/06/21 00:35:17  bard
// Added timestep control.  Changed constitutive constant number (only 3 are
// independent) and format.
//
// Revision 1.23  2000/06/15 21:57:04  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.22  2000/05/30 20:19:02  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.21  2000/05/26 18:15:11  guilkey
// Brought the CompNeoHook constitutive model up to functionality
// with the UCF.  Also, cleaned up all of the working models to
// rid them of the SAMRAI crap.
//
// Revision 1.20  2000/05/20 08:09:06  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.19  2000/05/15 19:39:39  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.18  2000/05/11 20:10:13  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.17  2000/05/07 06:02:03  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.16  2000/05/04 16:37:30  guilkey
// Got the CompNeoHookPlas constitutive model up to speed.  It seems
// to work but hasn't had a rigorous test yet.
//
// Revision 1.15  2000/05/02 19:31:23  guilkey
// Added a put for cmdata.
//
// Revision 1.14  2000/05/02 06:07:11  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.13  2000/04/27 23:18:43  sparker
// Added problem initialization for MPM
//
// Revision 1.12  2000/04/26 06:48:14  sparker
// Streamlined namespaces
//
// Revision 1.11  2000/04/25 18:42:33  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.10  2000/04/21 01:22:55  guilkey
// Put the VarLabels which are common to all constitutive models in the
// base class.  The only one which isn't common is the one for the CMData.
//
// Revision 1.9  2000/04/20 18:56:18  sparker
// Updates to MPM
//
// Revision 1.8  2000/04/19 05:26:03  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.7  2000/04/14 17:34:41  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.6  2000/03/21 01:29:40  dav
// working to make MPM stuff compile successfully
//
// Revision 1.5  2000/03/20 17:17:07  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.4  2000/03/17 09:29:34  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 00:49:32  guilkey
// Fixed the parameter lists in the .cc files
//
// Revision 1.2  2000/03/15 20:05:56  guilkey
// Worked over the ConstitutiveModel base class, and the CompMooneyRivlin
// class to operate on all particles in a patch of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
