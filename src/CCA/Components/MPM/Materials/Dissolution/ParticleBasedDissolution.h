/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// ContactStressIndependent.h

#ifndef __PARTICLE_BASED
#define __PARTICLE_BASED

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

   ParticleBasedDissolution.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_Particle_Based

DESCRIPTION
  Constant rate of dissolution prescribed by dLdt
WARNING
  
****************************************/

      class ParticleBasedDissolution : public Dissolution {
      private:

        // Prevent copying of this class
        // copy constructor
        ParticleBasedDissolution(const ParticleBasedDissolution &ci);
        ParticleBasedDissolution& operator=(const ParticleBasedDissolution &ci);

        MaterialManagerP    d_materialManager;

        // Dissolution rate
        double d_dLdt;

      public:
         // Constructor
         ParticleBasedDissolution(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb,
                          MPMFlags* flag);

         // Destructor
         virtual ~ParticleBasedDissolution();

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

#endif /* __PARTICLE_BASED */
