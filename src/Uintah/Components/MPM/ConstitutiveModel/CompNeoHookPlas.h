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
      private:
	 // Create datatype for storing model parameters
	 struct CMData {
	    double Bulk;
	    double Shear;
	    double FlowStress;
	    double K;
            double Alpha;
	 };	 
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
	 
	 // compute stable timestep for this region
	 virtual void computeStableTimestep(const Region* region,
					    const MPMMaterial* matl,
					    DataWarehouseP& new_dw);

	 // compute stress at each particle in the region
	 virtual void computeStressTensor(const Region* region,
					  const MPMMaterial* matl,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw);

	 // compute total strain energy for all particles in the region
	 virtual double computeStrainEnergy(const Region* region,
					    const MPMMaterial* matl,
					    const DataWarehouseP& new_dw);

         // initialize  each particle's constitutive model data
         virtual void initializeCMData(const Region* region,
				       const MPMMaterial* matl,
				       DataWarehouseP& new_dw);

	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const Region* region,
					     const DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const;

	 // class function to read correct number of parameters
	 // from the input file
	 static void readParameters(ProblemSpecP ps, double *p_array);

	 // class function to write correct number of parameters
	 // from the input file, and create a new object
	 static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps);
 
	 // member function to read correct number of parameters
	 // from the input file, and any other particle information
	 // need to restart the model for this particle 
	 // and create a new object
	 static ConstitutiveModel* readRestartParametersAndCreate(ProblemSpecP ps);
	 
	 // class function to create a new object from parameters
	 static ConstitutiveModel* create(double *p_array);

 	 const VarLabel* p_cmdata_label;
         const VarLabel* bElBarLabel;
      };

   }
}

#endif  // __NEOHOOK_CONSTITUTIVE_MODEL_H__ 

// $Log$
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

