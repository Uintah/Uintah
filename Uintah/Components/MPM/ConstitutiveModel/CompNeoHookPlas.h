#ifndef __NEOHOOKPLAS_CONSTITUTIVE_MODEL_H__
#define __NEOHOOKPLAS_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>

#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {
   namespace MPM {
/**************************************

CLASS
   CompNeoHookPlas
   
   Short description...

GENERAL INFORMATION

   CompNeoHookPlas.h

   Author?
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Comp_Neo_Hookean

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class CompNeoHookPlas : public ConstitutiveModel {
	 // Create datatype for storing model parameters
      public:
	 struct CMData {
	    double Bulk;
	    double Shear;
	    double FlowStress;
	    double K;
            double Alpha;
	 };	 
      private:
	 friend const TypeDescription* fun_getTypeDescription(CMData*);

	 CMData d_initialData;
	 
	 // Prevent copying of this class
	 // copy constructor
	 CompNeoHookPlas(const CompNeoHookPlas &cm);
	 CompNeoHookPlas& operator=(const CompNeoHookPlas &cm);

      public:
	 // constructors
	 CompNeoHookPlas(ProblemSpecP& ps);
	 
	 // destructor 
	 virtual ~CompNeoHookPlas();
	 
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

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);
 	 const VarLabel* p_cmdata_label;
         const VarLabel* bElBarLabel;
 	 const VarLabel* p_cmdata_label_preReloc;
         const VarLabel* bElBarLabel_preReloc;
      };

   }
}

#endif  // __NEOHOOK_CONSTITUTIVE_MODEL_H__ 

// $Log$
// Revision 1.16  2000/07/27 22:39:44  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.15  2000/06/15 21:57:04  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.14  2000/05/30 20:19:03  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.13  2000/05/26 18:15:12  guilkey
// Brought the CompNeoHook constitutive model up to functionality
// with the UCF.  Also, cleaned up all of the working models to
// rid them of the SAMRAI crap.
//
// Revision 1.12  2000/05/20 08:09:06  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.11  2000/05/15 22:24:34  dav
// Added isFlat declaration to CompNeoHookPlas.h
//
// Revision 1.10  2000/05/11 20:10:14  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.9  2000/05/07 06:02:03  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.8  2000/05/04 16:37:30  guilkey
// Got the CompNeoHookPlas constitutive model up to speed.  It seems
// to work but hasn't had a rigorous test yet.
//
// Revision 1.7  2000/04/26 06:48:15  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/25 18:42:34  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.5  2000/04/19 21:15:55  jas
// Changed BoundedArray to vector<double>.  More stuff to compile.  Critical
// functions that need access to data warehouse still have WONT_COMPILE_YET
// around the methods.
//
// Revision 1.4  2000/04/19 05:26:04  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/04/14 17:34:42  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.2  2000/03/20 17:17:08  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:11:48  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.

