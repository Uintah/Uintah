#ifndef __NULL_HEBURN_H__
#define __NULL_HEBURN_H__

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
   NullHEBurn
   
   Short description...

GENERAL INFORMATION

   NullHEBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HEBurn_Model_Null

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class NullHEBurn : public HEBurn { 
    private:
      
      // Prevent copying of this class
      // copy constructor
      NullHEBurn(const NullHEBurn &burn);
      NullHEBurn & operator=(const NullHEBurn &burn);
      
    public:
      // Constructor
      NullHEBurn(ProblemSpecP& ps);
      
      // Destructor
      ~NullHEBurn();

      void computeMassRate(const Patch* patch,
			   const MPMMaterial* matl,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);

      virtual bool getBurns() const;

      // initialize and allocate the burn model data
      void initializeBurnModelData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouseP& new_dw);

      void addComputesAndRequires(Task* task,
				  const MPMMaterial* matl,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw) const;

    };
} // End namespace Uintah
    


#endif /* __NULL_HEBURN_H__*/

