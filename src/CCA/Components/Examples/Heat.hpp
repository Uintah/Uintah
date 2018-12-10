#ifndef Packages_Uintah_CCA_Components_Examples_Heat_hpp
#define Packages_Uintah_CCA_Components_Examples_Heat_hpp

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/Examples/ExamplesLabel.h>

namespace Uintah{
  class Heat : public ApplicationCommon {
  public:
    Heat(const ProcessorGroup* myworld,
	 const MaterialManagerP materialManager);
    
    virtual ~Heat();

    virtual void problemSetup(const ProblemSpecP&     ps,
                              const ProblemSpecP&     restart_ps,
                                    GridP&            grid);

    virtual void scheduleInitialize(const LevelP&     level,
                                          SchedulerP& sched);

    virtual void scheduleRestartInitialize(const LevelP&     level,
                                                 SchedulerP& sched);

    virtual void scheduleComputeStableTimeStep(const LevelP&     level,
                                                     SchedulerP& sched);

    virtual void scheduleTimeAdvance(const LevelP&     level,
                                           SchedulerP& sched);

  protected:
    ExamplesLabel* d_lb;
    SimpleMaterial* d_mat;
    double d_delt, d_alpha, d_r0, d_gamma;

  private:
    virtual void initialize(const ProcessorGroup* pg,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DataWarehouse*  old_dw,
                                  DataWarehouse*  new_dw);

    virtual void computeStableTimeStep(const ProcessorGroup* pg,
                                       const PatchSubset*    patches,
                                       const MaterialSubset* matls,
                                             DataWarehouse*  old_dw,
                                             DataWarehouse*  new_dw);

    virtual void timeAdvance(const ProcessorGroup* pg,
                             const PatchSubset*    patches,
                             const MaterialSubset* matls,
                                   DataWarehouse*  old_dw,
                                   DataWarehouse*  new_dw);



    Heat(const Heat&);
    Heat& operator=(const Heat&);
  };
}

#endif // End Packages_Uintah_CCA_Components_Examples_Heat_hpp
