#ifndef UINTAH_IMPLICIT_HEAT_CONDUCTION_H
#define UINTAH_IMPLICIT_HEAT_CONDUCTION_H

#include <sci_defs/petsc_defs.h>

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <vector>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;
  class DataWarehouse;
  class ProcessorGroup;
  class MPMPetscSolver;
  class SimpleSolver;

  
  class ImplicitHeatConduction {
  public:
    
    ImplicitHeatConduction(SimulationStateP& ss,MPMLabel* lb, MPMFlags* mflags);
    ~ImplicitHeatConduction();

    void scheduleInitialize(            const LevelP& level, SchedulerP&);

    void scheduleFormHCStiffnessMatrix( SchedulerP&, const PatchSet*,
                                        const MaterialSet*);

    void scheduleFormHCQ(               SchedulerP&, const PatchSet*,
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

    void actuallyInitializeHC(           const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

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
    bool do_IHC;
    const PatchSet* d_perproc_patches;
#ifdef HAVE_PETSC
    vector<MPMPetscSolver*> d_HC_solver;
#else
    vector<SimpleSolver*> d_HC_solver;
#endif

    SimulationStateP d_sharedState;
    int NGP, NGN;

    ImplicitHeatConduction(const ImplicitHeatConduction&);
    ImplicitHeatConduction& operator=(const ImplicitHeatConduction&);

    inline bool compare(double num1, double num2) {
      double EPSILON=1.e-16;
                                                                                
      return (abs(num1-num2) <= EPSILON);
    };

  };
  
} // end namespace Uintah
#endif
