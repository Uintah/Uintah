/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// ContactStressIndependent.h

#ifndef __CONTACT_STRESS_INDEPENDENT
#define __CONTACT_STRESS_INDEPENDENT

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
   ContactStressIndependent
   
   Short description...

GENERAL INFORMATION

   ContactStressIndependent.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_ContactStressIndependent

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/

      class ContactStressIndependent : public Dissolution {
      private:

        // Prevent copying of this class
        // copy constructor
        ContactStressIndependent(const ContactStressIndependent &ci);
        ContactStressIndependent& operator=(const ContactStressIndependent &ci);

        MaterialManagerP    d_materialManager;

        // Dissolution rate
        double d_Vm;
        double d_R;
        double d_StressThresh;
        double d_maxCemThickness;  // Diss doesn't occur for thicker overgrowth
        double d_Ao;
        double d_Ea;
        double d_Ao_clay; // Modified value in the presence of clay
        double d_Ea_clay; // Modified value in the presence of clay
        // master material
        int    d_masterModalID;
        int    d_inContactWithModalID;

      public:
         // Constructor
         ContactStressIndependent(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb,
                          MPMFlags* flag);

         // Destructor
         virtual ~ContactStressIndependent();

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

#endif /* __CONTACT_STRESS_INDEPENDENT */
