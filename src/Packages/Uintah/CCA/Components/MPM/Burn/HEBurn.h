#ifndef __HEBURN_H__
#define __HEBURN_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

#include <math.h>

namespace Uintah {

using namespace SCIRun;

   class Patch;
   class VarLabel;
   class Task;

/**************************************

CLASS
   HEBurn 
   
   Short description...

GENERAL INFORMATION

   HEBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HEBurn_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

      class HEBurn {
      public:
         // Constructor
	 HEBurn();
	 virtual ~HEBurn();

	 // Basic burn methods

	  bool isBurnable();

          virtual void computeMassRate(const PatchSubset* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw) = 0;


         //////////
         // Create space in data warehouse for burn model data

          virtual bool getBurns() const = 0;

          virtual void initializeBurnModelData(const Patch* patch,
                                               const MPMMaterial* matl,
                                               DataWarehouse* new_dw) = 0;

          virtual void addComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const PatchSet* patch) const = 0;

       protected:
	  bool d_burnable;
	  MPMLabel* lb;
    
      };
      
} // End namespace Uintah

#endif // __HEBURN_H__

