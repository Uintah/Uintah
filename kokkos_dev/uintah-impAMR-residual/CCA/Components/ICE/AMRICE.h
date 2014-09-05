#ifndef Packages_Uintah_CCA_Components_ICE_AMRICE_h
#define Packages_Uintah_CCA_Components_ICE_AMRICE_h

#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
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
                                         
    virtual void scheduleRefine (const PatchSet* patches, 
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
                               Task::DomainSpec DS,
                               const MaterialSubset* matls,
                               int step, 
                               int nsteps);

  private:

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, 
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw);
                    
    template<class T>
    void refine_CF_interfaceOperator(const Patch* patch, 
                                     const Level* fineLevel,
                                     const Level* coarseLevel,
                                     CCVariable<T>& Q, 
                                     const VarLabel* label,
                                     double subCycleProgress_var, 
                                     int matl, 
                                     DataWarehouse* fine_new_dw,
                                     DataWarehouse* coarse_old_dw,
                                     DataWarehouse* coarse_new_dw);
                    
    void refineCoarseFineInterface(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*, 
                                   DataWarehouse* new_dw,
                                   const double factor);

    void scheduleSetBC_FineLevel(const PatchSet* patches,
                                 SchedulerP& scheduler,
                                 const bool isNewLevel);
                                                                    
    void setBC_FineLevel(const ProcessorGroup*,
                         const PatchSubset* patches,              
                         const MaterialSubset*,                   
                         DataWarehouse* fine_old_dw,              
                         DataWarehouse* fine_new_dw);     
                                   
    void iteratorTest(const Patch* finePatch,
                      const Level* fineLevel,
                      const Level* coarseLevel,
                      DataWarehouse* new_dw);
                      
    void refine(const ProcessorGroup*,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse*,
                DataWarehouse* new_dw);

    template<class T>
    void CoarseToFineOperator(CCVariable<T>& q_CC,
                              const VarLabel* varLabel,
                              const int indx,
                              DataWarehouse* new_dw,
                              const double ratio,
                              const Patch* finePatch,
                              const Level* fineLevel,
                              const Level* coarseLevel);
                         
    template<class T>
    void fineToCoarseOperator(CCVariable<T>& q_CC,
                              const CCVariable<double>& rho_CC_coarse,
                              constCCVariable<double>& cv_coarse,
                              const string& quantity,
                              const VarLabel* varLabel,
                              const int indx,
                              DataWarehouse* new_dw,
                              const Patch* coarsePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel);
                                  
    void coarsen(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse*, DataWarehouse* new_dw);
 
    //__________________________________
    //    refluxing
    void scheduleReflux_computeCorrectionFluxes(const LevelP& coarseLevel,
                                                SchedulerP& sched);
                                                    
    void reflux_computeCorrectionFluxes(const ProcessorGroup*,
                                        const PatchSubset* coarsePatches,
                                        const MaterialSubset* matls,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw); 
    
    void scheduleReflux_applyCorrection(const LevelP& coarseLevel,
                                        SchedulerP& sched);
                                        
    void reflux_applyCorrectionFluxes(const ProcessorGroup*,
                                      const PatchSubset* coarsePatches,
                                      const MaterialSubset* matls,
                                      DataWarehouse*,
                                      DataWarehouse* new_dw);                                      
 
    void scheduleReflux(const LevelP& coarseLevel,
                        SchedulerP& sched);

    template<class T>
    void refluxOperator(CCVariable<T>& q_CC_coarse,
                        CCVariable<double>& rho_CC_coarse,
                        constCCVariable<double>& cv,
                        const string& fineVarLabel,
                        const int indx,
                        const Patch* coarsePatch,
                        const Patch* finePatch,
                        const Level* coarseLevel,
                        const Level* fineLevel,
                        DataWarehouse* new_dw);

                
    void reflux(const ProcessorGroup*,
                const PatchSubset* coarsePatches,
                const MaterialSubset* matls,
                DataWarehouse*,
                DataWarehouse* new_dw);               

                 
    void errorEstimate(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw,
                       bool initial);
                       
    void compute_Mag_gradient( constCCVariable<double>& q_CC,
                               CCVariable<double>& mag_grad_q_CC,                   
                               const Patch* patch);
                               
    void compute_Mag_Divergence( constCCVariable<Vector>& q_CC,
                                 CCVariable<double>& mag_div_q_CC,                   
                                 const Patch* patch);

    void set_refineFlags( CCVariable<double>& q_CC_grad,
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
    
    bool d_regridderTest;
    int d_orderOfInterpolation;    
  };
}

#endif
