/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// ParticleBasedPyrolysis.h

#ifndef __PARTICLE_BASED_PYROLYSIS
#define __PARTICLE_BASED_PYROLYSIS

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

   ParticleBasedPyrolysis.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_Particle_Based

DESCRIPTION
  Constant rate of dissolution prescribed by rate
WARNING
  
****************************************/

      class ParticleBasedPyrolysis : public Dissolution {
      private:

        // Prevent copying of this class
        // copy constructor
        ParticleBasedPyrolysis(const ParticleBasedPyrolysis &ci);
        ParticleBasedPyrolysis& operator=(const ParticleBasedPyrolysis &ci);

        MaterialManagerP    d_materialManager;

        // Pyrolysis rate
        double d_rate;  // dM/dt

      public:
         // Constructor
         ParticleBasedPyrolysis(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb,
                          MPMFlags* flag);

         // Destructor
         virtual ~ParticleBasedPyrolysis();

         virtual void outputProblemSpec(ProblemSpecP& ps);

         // Pyrolysis methods
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

#endif /* __PARTICLE_BASED_PYROLYSIS */
