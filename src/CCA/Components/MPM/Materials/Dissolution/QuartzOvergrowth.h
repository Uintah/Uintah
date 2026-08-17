/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// QuartzOvergrowth.h

#ifndef __QUARTZ_OVERGROWTH_MODEL
#define __QUARTZ_OVERGROWTH_MODEL

#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>
#include <CCA/Components/MPM/Materials/Dissolution/DissolutionMaterialSpec.h> 
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>

namespace Uintah {
/**************************************

CLASS
   QuartzOvergrowth
   
   Short description...

GENERAL INFORMATION

   QuartzOvergrowth.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_QuartzOvergrowth

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/

      class QuartzOvergrowth : public Dissolution {
      private:
         
        // Prevent copying of this class
        // copy constructor
        QuartzOvergrowth(const QuartzOvergrowth &con);
        QuartzOvergrowth& operator=(const QuartzOvergrowth &con);

        MaterialManagerP    d_materialManager;

        // Growth rate
        double d_growthRate;
        double d_growthRateClay;
        // master material
        int    d_masterModalID;

      public:
         // Constructor
         QuartzOvergrowth(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb);

         // Destructor
         virtual ~QuartzOvergrowth();

         virtual void outputProblemSpec(ProblemSpecP& ps);

         // Dissolution methods
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

#endif /* __QUARTZ_OVERGROWTH_MODEL */
