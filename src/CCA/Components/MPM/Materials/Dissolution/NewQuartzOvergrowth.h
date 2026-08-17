/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// NewQuartzOvergrowth.h

#ifndef __NEW_QUARTZ_OVERGROWTH_MODEL
#define __NEW_QUARTZ_OVERGROWTH_MODEL

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
   NewQuartzOvergrowth
   
   Short description...

GENERAL INFORMATION

   NewQuartzOvergrowth.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_NewQuartzOvergrowth

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/
      class NewQuartzOvergrowth : public Dissolution {
      private:
         
        // Prevent copying of this class
        // copy constructor
        NewQuartzOvergrowth(const NewQuartzOvergrowth &con);
        NewQuartzOvergrowth& operator=(const NewQuartzOvergrowth &con);

        MaterialManagerP    d_materialManager;

        // Crystal pressure (pressure above which overgrowth won't take place)
        double d_crystalPressure;
        // master material
        int    d_masterModalID;

      public:
         // Constructor
         NewQuartzOvergrowth(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb);

         // Destructor
         virtual ~NewQuartzOvergrowth();

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
