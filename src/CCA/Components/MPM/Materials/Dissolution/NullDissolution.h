/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// NullDissolution.h

#ifndef __NULL_DISSOLUTION_H__
#define __NULL_DISSOLUTION_H__

#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>

namespace Uintah {
/**************************************

CLASS
   NullDissolution
   
   Short description...

GENERAL INFORMATION

   NullDissolution.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_Null

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class NullDissolution : public Dissolution {
    private:
      
      // Prevent copying of this class
      // copy constructor
      NullDissolution(const NullDissolution &con);
      NullDissolution& operator=(const NullDissolution &con);

      MaterialManagerP d_materialManager;
      
    public:
      // Constructor
      NullDissolution(const ProcessorGroup* myworld,
                      MaterialManagerP& ss, MPMLabel* lb);

      // Destructor
      virtual ~NullDissolution();

      virtual void outputProblemSpec(ProblemSpecP& ps);

      // Basic dissolution methods
      virtual void computeMassBurnFraction(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);
      
      virtual void addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* matls);
    };
} // End namespace Uintah
    
#endif /* __NULL_DISSOLUTION_H__ */
