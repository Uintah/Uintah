#ifndef Packages_Uintah_CCA_Components_ICE_AMRICE_h
#define Packages_Uintah_CCA_Components_ICE_AMRICE_h

#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>

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
                                         
    virtual void scheduleRefine (const LevelP& fineLevel, 
                                 SchedulerP& sched); 
                                                                    
    virtual void scheduleCoarsen(const LevelP& coarseLevel, 
                                 SchedulerP& sched);


    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched);
                                               
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched);
  protected:
    void refineCoarseFineBoundaries(const Patch* patch,
                                    CCVariable<double>& val,
                                    DataWarehouse* new_dw,
                                    const VarLabel* label,
                                    int matl, 
                                    double factor);
                
    void refineCoarseFineBoundaries(const Patch* patch,
                                    CCVariable<Vector>& val,
                                    DataWarehouse* new_dw,
                                    const VarLabel* label,
                                    int matl, 
                                    double factor);
                                  
    void addRefineDependencies(Task* task, 
                               const VarLabel* var,
                               int step, 
                               int nsteps);
  private:

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, 
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw);
                    
    void refineCoarseFineInterface(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*, 
                                   DataWarehouse* new_dw,
                                   double factor);

    template<class T>
    void CoarseToFineOperator(CCVariable<T>& q_CC,
                              const VarLabel* varLabel,
                              const int indx,
                              DataWarehouse* new_dw,
                              const double ratio,
                              const Patch* finePatch,
                              const Level* fineLevel,
                              const Level* coarseLevel);
                              
    void refine(const ProcessorGroup*,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse*,
                DataWarehouse* new_dw);
                         
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
			  DataWarehouse*,
			  DataWarehouse* new_dw,
                       bool initial);
                       
    void compute_q_CC_gradient( constCCVariable<double>& q_CC,
                                CCVariable<Vector>& q_CC_grad,                   
                                const Patch* patch);
                               
    void compute_q_CC_gradient( constCCVariable<Vector>& q_CC,
                                CCVariable<Vector>& q_CC_grad,                   
                                const Patch* patch);

    void set_refineFlags( CCVariable<Vector>& q_CC_grad,
                          double threshold,
                          CCVariable<int>& refineFlag,
                          PerPatch<PatchFlagP>& refinePatchFlag,
                          const Patch* patch);                                                  
    AMRICE(const AMRICE&);
    AMRICE& operator=(const AMRICE&);
    
    //__________________________________
    // refinement criteria threshold knobs
    double d_rho_threshold;     
    double d_temp_threshold;    
    double d_press_threshold;   
    double d_vol_frac_threshold;
    double d_vel_threshold;     
  };
}

#endif
