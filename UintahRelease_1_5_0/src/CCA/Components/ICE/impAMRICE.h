/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */#ifndef Packages_Uintah_CCA_Components_ICE_impAMRICE_h
#define Packages_Uintah_CCA_Components_ICE_impAMRICE_h
#include <CCA/Components/ICE/AMRICE.h>

namespace Uintah {
  class impAMRICE : public AMRICE{
  public:
    impAMRICE(const ProcessorGroup* myworld);
    virtual ~impAMRICE();
    
  void scheduleTimeAdvance( const LevelP& level, 
                             SchedulerP& sched);
               
  private:                     
  void scheduleMultiLevelPressureSolve( SchedulerP& sched,                 
                                        const GridP grid,                  
                                        const PatchSet*,                   
                                        const MaterialSubset* one_matl,    
                                        const MaterialSubset* press_matl,  
                                        const MaterialSubset* ice_matls,   
                                        const MaterialSubset* mpm_matls,   
                                        const MaterialSet* all_matls);     

  void multiLevelPressureSolve(const ProcessorGroup* pg,
                               const PatchSubset* patches,           
                               const MaterialSubset*,                
                               DataWarehouse* ParentOldDW,           
                               DataWarehouse* ParentNewDW,           
                               GridP grid,                           
                               const MaterialSubset* ice_matls,      
                               const MaterialSubset* mpm_matls);     

   void scheduleAddReflux_RHS(SchedulerP& sched,
                              const LevelP& coarseLevel,
                              const MaterialSubset* one_matl,
                              const MaterialSet* all_matls,
                              const bool OnOff);

  void compute_refluxFluxes_RHS(const ProcessorGroup*,
                                const PatchSubset* coarsePatches,
                                const MaterialSubset* matls,
                                DataWarehouse*,
                                DataWarehouse* new_dw);

  void apply_refluxFluxes_RHS(const ProcessorGroup*,
                              const PatchSubset* coarsePatches,
                              const MaterialSubset* matls,
                              DataWarehouse*,
                              DataWarehouse* new_dw);

  void scheduleCoarsen_delP(SchedulerP& sched, 
                            const LevelP& coarseLevel,
                            const MaterialSubset* press_matl,
                            const VarLabel* variable);

  void coarsen_delP(const ProcessorGroup*,
                    const PatchSubset* coarsePatches,
                    const MaterialSubset* matls,
                    DataWarehouse*,
                    DataWarehouse* new_dw,
                    const VarLabel* variable);

  void scheduleZeroMatrix_UnderFinePatches(SchedulerP& sched, 
                                          const LevelP& coarseLevel,
                                          const MaterialSubset* one_matl);

  void zeroMatrix_UnderFinePatches(const ProcessorGroup*,
                                   const PatchSubset* coarsePatches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw); 
                                   
  void schedule_matrixBC_CFI_coarsePatch(SchedulerP& sched, 
                                         const LevelP& coarseLevel,
                                         const MaterialSubset* one_matl,
                                         const MaterialSet* all_matls);

  void matrixBC_CFI_coarsePatch(const ProcessorGroup*,
                                const PatchSubset* coarsePatches,
                                const MaterialSubset* matls,
                                DataWarehouse*,
                                DataWarehouse* new_dw);

  void schedule_bogus_imp_delP(SchedulerP& sched,
                               const PatchSet* perProcPatches,
                               const MaterialSubset* press_matl,
                               const MaterialSet* all_matls);
                               
  void bogus_imp_delP(const ProcessorGroup*,
                      const PatchSubset* patches,   
                      const MaterialSubset*,        
                      DataWarehouse*,               
                      DataWarehouse* new_dw);  
  };
}
#endif
