/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// SaltPrecipitationModel.h

#ifndef __SALT_PRECIPITATION_MODEL
#define __SALT_PRECIPITATION_MODEL

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
   SaltPrecipitationModel
   
   Short description...

GENERAL INFORMATION

   SaltPrecipitationModel.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_SaltPrecipitationModel

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/

      class SaltPrecipitationModel : public Dissolution {
      private:
         
        // Prevent copying of this class
        // copy constructor
        SaltPrecipitationModel(const SaltPrecipitationModel &con);
        SaltPrecipitationModel& operator=(const SaltPrecipitationModel &con);

        MaterialManagerP    d_materialManager;

        // Dissolution rate
        double d_Vm;
        double d_R;
        double d_StressThresh;
        double d_Ao;
        double d_Ea;
        double d_Ao_clay; // Modified value in the presence of clay
        double d_Ea_clay; // Modified value in the presence of clay
        // master material
        int    d_masterModalID;
        int    d_inContactWithModalID;

      public:
         // Constructor
         SaltPrecipitationModel(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb);

         // Destructor
         virtual ~SaltPrecipitationModel();

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

#endif /* __SALT_PRECIPITATION_MODEL */
