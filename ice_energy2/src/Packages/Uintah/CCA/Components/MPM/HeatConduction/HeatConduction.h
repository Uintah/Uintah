#ifndef UINTAH_HEAT_CONDUCTION_H
#define UINTAH_HEAT_CONDUCTION_H

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;
  class DataWarehouse;
  class ProcessorGroup;
  
  class HeatConduction {
  public:
    
    HeatConduction(SimulationStateP& ss,MPMLabel* lb, MPMFlags* mflags);
    ~HeatConduction();

    void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);
    
    void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
                                    const MaterialSet*);
    
    void scheduleIntegrateTemperatureRate(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);
    
    void computeInternalHeatRate(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);
    
    void solveHeatEquations(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* /*old_dw*/,
                            DataWarehouse* new_dw);
    
    void integrateTemperatureRate(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  private:
    MPMLabel* d_lb;
    MPMFlags* d_flag;
    SimulationStateP d_sharedState;
    int NGP, NGN;

    HeatConduction(const HeatConduction&);
    HeatConduction& operator=(const HeatConduction&);
    
  };
  
} // end namespace Uintah
#endif
