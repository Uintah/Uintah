#ifndef __SIMPLE_HEBURN_H__
#define __SIMPLE_HEBURN_H__

#include <Packages/Uintah/CCA/Components/MPM/Burn/HEBurn.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>


namespace Uintah {
/**************************************

CLASS
   SimpleHEBurn
   
   Short description...

GENERAL INFORMATION

   SimpleHEBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HEBurn_Model_Simple

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class SimpleHEBurn : public HEBurn { 
    private:
      double a,b;
      
      // Prevent copying of this class
      // copy constructor
      SimpleHEBurn(const SimpleHEBurn &burn);
      SimpleHEBurn & operator=(const SimpleHEBurn &burn);
      
    public:
      // Constructor
      SimpleHEBurn(ProblemSpecP& ps);
      
      // Destructor
      ~SimpleHEBurn();

      void computeMassRate(const PatchSubset* patch,
			   const MPMMaterial* matl,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

      virtual bool getBurns() const;

      // initialize and allocate the burn model data
      void initializeBurnModelData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw);

      virtual void addComputesAndRequires(Task* task,
					  const MPMMaterial* matl,
					  const PatchSet* patch) const;
    };
} // End namespace Uintah
    


#endif /* __SIMPLE_HEBURN_H__*/

