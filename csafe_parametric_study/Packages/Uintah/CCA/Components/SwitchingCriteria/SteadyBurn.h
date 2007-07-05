#ifndef Packages_Uintah_CCA_Components_Switching_SteadyBurn_h
#define Packages_Uintah_CCA_Components_Switching_SteadyBurn_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SwitchingCriteria.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;

  class SteadyBurnCriteria : public SwitchingCriteria
    {
    public:
      SteadyBurnCriteria(ProblemSpecP& ps);
      virtual ~SteadyBurnCriteria();
      
      virtual void problemSetup(const ProblemSpecP& ps, 
                                const ProblemSpecP& restart_prob_spec, 
                                SimulationStateP& sharedState);

      virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);

      void switchTest(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse*,
                      DataWarehouse*);


    private:
      unsigned int d_material;
      double d_temperature;   
      double d_BP;              // Number of Particles at Boundary
      
      SimulationStateP d_sharedState; 
      MPMLabel* Mlb;
      MPMICELabel* MIlb;
      ICELabel* Ilb;
      #define d_SMALL_NUM 1e-100
    };
} // End namespace Uintah

#endif 
