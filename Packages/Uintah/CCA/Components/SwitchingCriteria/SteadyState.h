#ifndef Packages_Uintah_CCA_Components_Switching_SteadyState_h
#define Packages_Uintah_CCA_Components_Switching_SteadyState_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SwitchingCriteria.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>


namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;
  class VarLabel;

  class SteadyState : public SwitchingCriteria
    {
    public:
      // this function has a switch for all known SwitchingCriteria
    
      SteadyState(ProblemSpecP& ps);
      virtual ~SteadyState();
      
      virtual void problemSetup(const ProblemSpecP& ps, 
                                SimulationStateP& sharedState);

      virtual void scheduleInitialize(const LevelP& level, SchedulerP& sched);
      virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
      virtual void scheduleDummy(const LevelP& level, SchedulerP& sched);

      void switchTest(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse*,
                      DataWarehouse*);

      void initialize(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse* old_dw,
                      DataWarehouse* new_dw);

      void dummy(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse* old_dw,
                      DataWarehouse* new_dw);



    private:
      unsigned int d_material;
      unsigned int d_numSteps;
      const VarLabel* heatFlux_CCLabel;
      const VarLabel* heatFluxSumLabel;
      const VarLabel* heatFluxSumTimeDerivativeLabel;
      SimulationStateP d_sharedState; 
    };
} // End namespace Uintah

#endif 
