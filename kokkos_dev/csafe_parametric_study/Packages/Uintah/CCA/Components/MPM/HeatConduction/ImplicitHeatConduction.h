#ifndef UINTAH_IMPLICIT_HEAT_CONDUCTION_H
#define UINTAH_IMPLICIT_HEAT_CONDUCTION_H

#include <sci_defs/petsc_defs.h>

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <vector>
#include <string>
#include <math.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;
  class DataWarehouse;
  class ProcessorGroup;
  class Solver;


  
  class ImplicitHeatConduction {
  public:
    
    ImplicitHeatConduction(SimulationStateP& ss,MPMLabel* lb, MPMFlags* mflags);
    ~ImplicitHeatConduction();

    void problemSetup(std::string solver_type);

    void scheduleFormHCStiffnessMatrix( SchedulerP&, const PatchSet*,
                                        const MaterialSet*);

    void scheduleFormHCQ(               SchedulerP&, const PatchSet*,
                                        const MaterialSet*);

    void scheduleAdjustHCQAndHCKForBCs( SchedulerP&, const PatchSet*,
                                        const MaterialSet*);

    void scheduleDestroyHCMatrix(       SchedulerP&, const PatchSet*,
                                        const MaterialSet*);

    void scheduleCreateHCMatrix(        SchedulerP&, const PatchSet*,
                                        const MaterialSet*);

    void scheduleApplyHCBoundaryConditions(      SchedulerP&, const PatchSet*,
                                                 const MaterialSet*);

    void scheduleFindFixedHCDOF(                 SchedulerP&, const PatchSet*,
                                                 const MaterialSet*);

    void scheduleSolveForTemp(                   SchedulerP&, const PatchSet*,
                                                 const MaterialSet*);

    void scheduleGetTemperatureIncrement(        SchedulerP&, const PatchSet*,
                                                 const MaterialSet*);

    void destroyHCMatrix(                const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void createHCMatrix(                 const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void applyHCBoundaryConditions(      const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void findFixedHCDOF(                 const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void formHCStiffnessMatrix(          const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void formHCQ(                        const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void adjustHCQAndHCKForBCs(          const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void solveForTemp(                   const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void getTemperatureIncrement(        const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    void fillgTemperatureRate(           const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

  private:
    MPMLabel* lb;
    MPMFlags* d_flag;
    MaterialSubset* one_matl;
    bool do_IHC;
    bool d_HC_transient;
    const PatchSet* d_perproc_patches;

    Solver* d_HC_solver;

    SimulationStateP d_sharedState;
    int NGP, NGN;

    ImplicitHeatConduction(const ImplicitHeatConduction&);
    ImplicitHeatConduction& operator=(const ImplicitHeatConduction&);

    void findNeighbors(IntVector n,vector<int>& neigh, Array3<int>& l2g);

    inline bool compare(double num1, double num2) {
      double EPSILON=1.e-16;
      return (fabs(num1-num2) <= EPSILON);
    };

  };
  
} // end namespace Uintah
#endif
