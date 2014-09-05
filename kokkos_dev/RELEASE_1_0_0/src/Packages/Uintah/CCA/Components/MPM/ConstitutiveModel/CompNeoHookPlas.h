#ifndef __NEOHOOKPLAS_CONSTITUTIVE_MODEL_H__
#define __NEOHOOKPLAS_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {

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
	 struct StateData {
            double Alpha;
	 };	 
      private:
	 friend const TypeDescription* fun_getTypeDescription(StateData*);

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
					    DataWarehouse* new_dw);

	 // compute stress at each particle in the patch
	 virtual void computeStressTensor(const PatchSubset* patches,
					  const MPMMaterial* matl,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw);

         // initialize  each particle's constitutive model data
         virtual void initializeCMData(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const PatchSet* patches) const;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);

 	 const VarLabel* p_statedata_label;
 	 const VarLabel* p_statedata_label_preReloc;
         const VarLabel* bElBarLabel;
         const VarLabel* bElBarLabel_preReloc;
      };
} // End namespace Uintah

#endif  // __NEOHOOK_CONSTITUTIVE_MODEL_H__ 
