#ifndef __COMPMOONRIV_CONSTITUTIVE_MODEL_H__
#define __COMPMOONRIV_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>

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
      private:
	 
	 // Create datatype for storing model parameters
	 struct CMData {
	    double C1;
	    double C2;
	    double C3;
	    double C4;
	 };
	 
	 // Prevent copying of this class
	 // copy constructor
	 CompMooneyRivlin(const CompMooneyRivlin &cm);
	 CompMooneyRivlin& operator=(const CompMooneyRivlin &cm);
	 
      public:
	 // constructor
	 CompMooneyRivlin(const Region* region,
			  const MPMMaterial* matl);
	 
	 // constructor
	 CompMooneyRivlin(ProblemSpecP& ps);
	 
	 // destructor 
	 virtual ~CompMooneyRivlin();
	 
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
	 
	 // class function to read correct number of parameters
	 // from the input file
	 static void readParameters(ProblemSpecP ps, double *p_array);
	 
	 
	 // class function to read correct number of parameters
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
      };
      
   }
}

#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

// $Log$
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
// class to operate on all particles in a region of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
