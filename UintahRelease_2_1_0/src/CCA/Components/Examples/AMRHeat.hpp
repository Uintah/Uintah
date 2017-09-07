#ifndef Packages_Uintah_CCA_Components_Examples_AMRHeat_hpp
#define Packages_Uintah_CCA_Components_Examples_AMRHeat_hpp

#include <CCA/Components/Examples/Heat.hpp>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/NCVariable.h>

namespace Uintah {
  class AMRHeat : public Heat {
  public:
    AMRHeat(const ProcessorGroup* world);
    virtual ~AMRHeat();

    virtual void problemSetup(const ProblemSpecP&     ps,
                              const ProblemSpecP&     restart_ps,
                                    GridP&            grid,
                                    SimulationStateP& state);

    virtual void scheduleRefineInterface(const LevelP&     fineLevel,
                                               SchedulerP& scheduler,
                                               bool        needCoarseOld,
                                               bool        needCoarseNew);

    virtual void scheduleCoarsen(const LevelP&     coarseLevel,
                                       SchedulerP& sched);

    virtual void scheduleRefine (const PatchSet*   patches,
                                       SchedulerP& sched);

    virtual void scheduleErrorEstimate(const LevelP&     coarseLevel,
                                             SchedulerP& sched);

    virtual void scheduleInitialErrorEstimate(const LevelP&     coarseLevel,
                                                    SchedulerP& sched);

  private:
    double d_refine_threshold;
    IntVector d_refinement_ratio;

    void errorEstimate(const ProcessorGroup* pg,
                       const PatchSubset*    patches,
                       const MaterialSubset* matls,
                             DataWarehouse*  old_dw,
                             DataWarehouse*  new_dw);

    double computeError(const IntVector& c,
                        const Patch* patch,
                              constNCVariable<double>& temp);

    void refine(const ProcessorGroup* pg,
                const PatchSubset*    patches,
                const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw);

    void refineNode(NCVariable<double>& temp, constNCVariable<double>& coarse_temp,
                    IntVector fine_index,
                    const Level* fine_level, const Level* coarse_level);

    AMRHeat(const AMRHeat&);
    AMRHeat& operator=(const AMRHeat&);

  };
}
#endif // Packages_Uintah_CCA_Components_Examples_AMRHeat_hpp
