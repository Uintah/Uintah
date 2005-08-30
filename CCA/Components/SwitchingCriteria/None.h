#ifndef Packages_Uintah_CCA_Components_Switching_None_h
#define Packages_Uintah_CCA_Components_Switching_None_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SwitchingCriteria.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;

  class None : public SwitchingCriteria
    {
    public:
      // this function has a switch for all known SwitchingCriteria
    
      None(ProblemSpecP& ps);
      virtual ~None();
      
      virtual void problemSetup(const ProblemSpecP& ps, 
                                SimulationStateP& sharedState);

      virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);

      void switchTest(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse*,
                      DataWarehouse*);      

    private:
      SimulationStateP d_sharedState; 
    };
} // End namespace Uintah

#endif 
