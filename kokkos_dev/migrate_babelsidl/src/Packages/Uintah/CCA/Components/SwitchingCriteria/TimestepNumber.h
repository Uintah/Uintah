#ifndef Packages_Uintah_CCA_Components_Switching_TimestepNumber_h
#define Packages_Uintah_CCA_Components_Switching_TimestepNumber_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SwitchingCriteria.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;

  class TimestepNumber : public SwitchingCriteria
    {
    public:
      // this function has a switch for all known SwitchingCriteria
    
      TimestepNumber(ProblemSpecP& ps);
      virtual ~TimestepNumber();
      
      virtual void problemSetup(const ProblemSpecP& ps, 
                                const ProblemSpecP& materials_ps, 
                                SimulationStateP& sharedState);

      virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);

      void switchTest(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse*,
                      DataWarehouse*);


    private:
      unsigned int d_timestep;
      SimulationStateP d_sharedState; 
    };
} // End namespace Uintah

#endif 
