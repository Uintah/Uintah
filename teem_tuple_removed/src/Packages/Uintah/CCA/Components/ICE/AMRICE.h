#ifndef Packages_Uintah_CCA_Components_ICE_AMRICE_h
#define Packages_Uintah_CCA_Components_ICE_AMRICE_h

#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/Core/Grid/Task.h>

namespace Uintah {
  class AMRICE : public ICE{
  public:
    AMRICE(const ProcessorGroup* myworld);
    virtual ~AMRICE();
    
    virtual void problemSetup(const ProblemSpecP& params, 
                              GridP& grid,
                              SimulationStateP& sharedState);
                              
    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);
                                    
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
                                         SchedulerP& scheduler,
                                         int step, 
                                         int nsteps);
                                         
    virtual void scheduleCoarsen(const LevelP& coarseLevel, 
                                 SchedulerP& sched);
   
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched);
  protected:

    virtual void refineBoundaries(const Patch* patch,
                                  CCVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, 
                                  double factor);
                                  
    virtual void refineBoundaries(const Patch* patch,
                                  CCVariable<Vector>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, 
                                  double factor);
                                  
    virtual void refineBoundaries(const Patch* patch,
                                  SFCXVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, 
                                  double factor);
                                  
    virtual void refineBoundaries(const Patch* patch,
                                  SFCYVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, 
                                  double factor);
                                  
    virtual void refineBoundaries(const Patch* patch,
                                  SFCZVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, 
                                  double factor);
                                  
    virtual void addRefineDependencies(Task* task, 
                                       const VarLabel* var,
                                       int step, 
                                       int nsteps);
  private:

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, 
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw);
                    
    void refineInterface(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse*, 
                         DataWarehouse* new_dw,
                         double factor);
                         
    template<class T>
    void fineToCoarseOperator(CCVariable<T>& q_CC,
                              const VarLabel* varLabel,
                              const int matl,
                              DataWarehouse* new_dw,
                              const double ratio,
                              const Patch* coarsePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel);
                                  
    void coarsen(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse*, DataWarehouse* new_dw);
                 
    void errorEstimate(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*, DataWarehouse* new_dw);
    AMRICE(const AMRICE&);
    AMRICE& operator=(const AMRICE&);
  };
}

#endif
