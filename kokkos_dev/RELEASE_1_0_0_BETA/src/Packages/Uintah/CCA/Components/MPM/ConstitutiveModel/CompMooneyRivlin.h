#ifndef __COMPMOONRIV_CONSTITUTIVE_MODEL_H__
#define __COMPMOONRIV_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {

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
      };
} // End namespace Uintah


#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

