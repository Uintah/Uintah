#ifndef Packages_Uintah_CCA_Components_Switching_SimpleBurn_h
#define Packages_Uintah_CCA_Components_Switching_SimpleBurn_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/MPMICELabel.h>

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;

  class SimpleBurnCriteria : public SwitchingCriteria
    {
    public:
      SimpleBurnCriteria(ProblemSpecP& ps);
      virtual ~SimpleBurnCriteria();
      
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
      
      SimulationStateP d_sharedState; 
      MPMLabel* Mlb;
      MPMICELabel* MIlb;
      #define d_SMALL_NUM 1e-100
      #define d_TINY_RHO 1e-12
    };
} // End namespace Uintah

#endif 
