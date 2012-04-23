/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H

#include <Core/Grid/Variables/Stencil7.h>
#include <CCA/Components/ICE/Advection/Advector.h>
#include <CCA/Components/ICE/customInitialize.h>
#include <CCA/Components/ICE/CustomBCs/LODI2.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/Turbulence.h>
#include <CCA/Components/ICE/ExchangeCoefficients.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/ModelInterface.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/Utils.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Vector.h>
#include <vector>
#include <string>


#define MAX_MATLS 16

namespace Uintah { 
  using namespace SCIRun;
  class ModelInfo; 
  class ModelInterface; 
  class Turbulence;
  class AnalysisModule;
  
    // The following two structs are used by computeEquilibrationPressure to store debug information:
    //
    struct  EqPress_dbgMatl{
      int    mat;
      double press_eos;
      double volFrac;
      double rhoMicro;
      double temp_CC;
      double rho_CC;
    };

    struct  EqPress_dbg{
      int    count;
      double sumVolFrac;
      double press_new;
      double delPress;
      vector<EqPress_dbgMatl> matl;
    };
    
    class ICE : public UintahParallelComponent, public SimulationInterface {
    public:
      ICE(const ProcessorGroup* myworld, const bool doAMR = false);
      virtual ~ICE();

      virtual bool restartableTimesteps();

      virtual double recomputeTimestep(double current_dt);
      
      virtual void problemSetup(const ProblemSpecP& params,
                                const ProblemSpecP& restart_prob_spec,
                                GridP& grid, SimulationStateP&);

      virtual void outputProblemSpec(ProblemSpecP& ps);
      
      virtual void addMaterial(const ProblemSpecP& params, 
                               GridP& grid,
                               SimulationStateP&);
      
      virtual void updateExchangeCoefficients(const ProblemSpecP& params, 
                                              GridP& grid,
                                              SimulationStateP&);
      
      virtual void scheduleInitialize(const LevelP& level, 
                                      SchedulerP&);

      virtual void scheduleInitializeAddedMaterial(const LevelP& level, 
                                                   SchedulerP&);

      virtual void restartInitialize();
      
      virtual void scheduleComputeStableTimestep(const LevelP&,
                                                SchedulerP&);
      
      virtual void scheduleTimeAdvance( const LevelP& level, 
                                        SchedulerP&);
                                             
      virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP&);


      void scheduleComputePressure(SchedulerP&, 
                                   const PatchSet*,
                                   const MaterialSubset*,
                                   const MaterialSet*);
                                   
      void schedulecomputeDivThetaVel_CC(SchedulerP& sched,
                                         const PatchSet* patches,         
                                         const MaterialSubset* ice_matls, 
                                         const MaterialSubset* mpm_matls, 
                                         const MaterialSet* all_matls);    

      void scheduleComputeTempFC(SchedulerP&,
                                 const PatchSet*,               
                                 const MaterialSubset*,         
                                 const MaterialSubset*,         
                                 const MaterialSet*);           
                                                                                        
      void scheduleComputeVel_FC(SchedulerP&, 
                                      const PatchSet*,                
                                      const MaterialSubset*,          
                                      const MaterialSubset*,          
                                      const MaterialSubset*,          
                                      const MaterialSet*);            
      
      void scheduleAddExchangeContributionToFCVel(SchedulerP&, 
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSet*, 
                                            bool);
      
      void scheduleComputeDelPressAndUpdatePressCC(SchedulerP&, 
                                             const PatchSet*,
                                             const MaterialSubset*, 
                                             const MaterialSubset*,
                                             const MaterialSubset*,
                                             const MaterialSet*);
      
      void scheduleComputePressFC(SchedulerP&, 
                              const PatchSet*,
                              const MaterialSubset*,
                              const MaterialSet*);
                              
     void scheduleComputeThermoTransportProperties(SchedulerP&, 
                                                  const LevelP& level,
                                                  const MaterialSet*);
      
      void scheduleAccumulateMomentumSourceSinks(SchedulerP&, 
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSubset*,
                                            const MaterialSubset*,
                                            const MaterialSet*);
      
      void scheduleAccumulateEnergySourceSinks(SchedulerP&, 
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSubset*,
                                            const MaterialSubset*,
                                            const MaterialSet*);
      
      void scheduleComputeLagrangianValues(SchedulerP&, 
                                          const PatchSet*,
                                          const MaterialSet*);
                 
      void scheduleComputeLagrangianSpecificVolume(SchedulerP&,
                                                   const PatchSet*,
                                                   const MaterialSubset*,
                                                   const MaterialSubset*,
                                                   const MaterialSubset*,
                                                   const MaterialSet*);

      void scheduleAddExchangeToMomentumAndEnergy(SchedulerP& sched,
                                                  const PatchSet*,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSet* );
                                                  
      void scheduleMaxMach_on_Lodi_BC_Faces(SchedulerP&, 
                                            const LevelP&,
                                            const MaterialSet*);
      
      void computesRequires_AMR_Refluxing(Task* t, 
                                          const MaterialSet* ice_matls);                                                                                         
      
      void scheduleAdvectAndAdvanceInTime(SchedulerP&, 
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSet*);
                                          
      void scheduleConservedtoPrimitive_Vars(SchedulerP& sched,
                                             const PatchSet* patch_set,
                                             const MaterialSubset* ice_matlsub,
                                             const MaterialSet* ice_matls,
                                             const string& where);
                             
      void scheduleTestConservation(SchedulerP&, 
                                    const PatchSet*,
                                    const MaterialSubset*,
                                    const MaterialSet*);
                                       
      void scheduleCheckNeedAddMaterial(SchedulerP&, 
                                        const LevelP& level,
                                        const MaterialSet*);

      void scheduleSetNeedAddMaterialFlag(SchedulerP&, 
                                          const LevelP& level,
                                          const MaterialSet*);
//__________________________________ 
//__________________________________ 
//  I M P L I C I T   I C E
                                         
      void scheduleSetupMatrix(  SchedulerP&,
                                 const LevelP&,                  
                                 const PatchSet*,
                                 const MaterialSubset*,              
                                 const MaterialSet*);                                 
      void scheduleSetupRHS(  SchedulerP&,                
                              const PatchSet*, 
                              const MaterialSubset*,             
                              const MaterialSet*,
                              bool insideOuterIterLoop,
                              const string& computes_or_modifies);
                               
      void scheduleCompute_maxRHS(SchedulerP& sched,
                                  const LevelP& level,
                                  const MaterialSubset* one_matl,
                                  const MaterialSet*);
                                                  
      void scheduleUpdatePressure(  SchedulerP&,
                                   const LevelP&,
                                   const PatchSet*,
                                   const MaterialSubset*,         
                                   const MaterialSubset*, 
                                   const MaterialSubset*,
                                   const MaterialSet*);
                                   
      void scheduleRecomputeVel_FC(SchedulerP& sched,
                                   const PatchSet* patches,                 
                                   const MaterialSubset*,         
                                   const MaterialSubset*,         
                                   const MaterialSubset*,        
                                   const MaterialSet*,           
                                   bool);  
                                                  
     void scheduleComputeDel_P(  SchedulerP& sched,
                                 const LevelP& level,               
                                 const PatchSet* patches,           
                                 const MaterialSubset* one_matl,    
                                 const MaterialSubset* press_matl,  
                                 const MaterialSet* all_matls);
                                   
      void scheduleImplicitPressureSolve(SchedulerP& sched,
                                         const LevelP& level,
                                         const PatchSet*,
                                         const MaterialSubset* one_matl,
                                         const MaterialSubset* press_matl,
                                         const MaterialSubset* ice_matls,
                                         const MaterialSubset* mpm_matls,
                                         const MaterialSet* all_matls);
//__________________________________ 
//  I M P L I C I T   A M R I C E                                                       
      void scheduleCoarsen_delP(SchedulerP& sched, 
                                const LevelP& level,
                                const MaterialSubset* press_matl,
                                const VarLabel* variable);
                                                                                                                      
      void schedule_matrixBC_CFI_coarsePatch(SchedulerP& sched, 
                                             const LevelP& coarseLevel,
                                             const MaterialSubset* one_matl,
                                             const MaterialSet* all_matls);
                                                                                       
      void scheduleMultiLevelPressureSolve(SchedulerP& sched,
                                         const GridP grid,
                                         const PatchSet*,
                                         const MaterialSubset* one_matl,
                                         const MaterialSubset* press_matl,
                                         const MaterialSubset* ice_matls,
                                         const MaterialSubset* mpm_matls,
                                         const MaterialSet* all_matls); 
                                         
      void scheduleZeroMatrix_UnderFinePatches(SchedulerP& sched, 
                                               const LevelP& coarseLevel,
                                               const MaterialSubset* one_matl);

      void zeroMatrix_UnderFinePatches(const ProcessorGroup*,
                                       const PatchSubset* coarsePatches,
                                       const MaterialSubset*,
                                       DataWarehouse*,
                                       DataWarehouse* new_dw);

      void schedule_bogus_imp_delP(SchedulerP& sched,
                                   const PatchSet* perProcPatches,
                                   const MaterialSubset* press_matl,
                                   const MaterialSet* all_matls);
                           
       void scheduleAddReflux_RHS(SchedulerP& sched,
                                  const LevelP& coarseLevel,
                                  const MaterialSubset* one_matl,
                                  const MaterialSet* all_matls,
                                  const bool OnOff);
                                   
//__________________________________ 
//   M O D E L S
      void scheduleComputeModelSources(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* matls);
      void scheduleUpdateVolumeFraction(SchedulerP& sched,
                                        const LevelP& level,
                                        const MaterialSubset* press_matl,
                                        const MaterialSet* matls);
                                        
      void scheduleComputeLagrangian_Transported_Vars(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet*);

      void setMPMICELabel(MPMICELabel* mil) {
       MIlb = mil;
      };

       void setWithMPM() {
         d_with_mpm = true;
       };

       void setWithRigidMPM() {
         d_with_rigid_mpm = true;
       };

      
    public:
      
      void actuallyInitialize(const ProcessorGroup*, 
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse*, 
                              DataWarehouse* new_dw);
                              
      void actuallyInitializeAddedMaterial(const ProcessorGroup*, 
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse*, 
                                           DataWarehouse* new_dw);
                              
      void initializeSubTask_hydrostaticAdj(const ProcessorGroup*, 
                                     const PatchSubset*,
                                     const MaterialSubset*,
                                     DataWarehouse*, 
                                     DataWarehouse* new_dw);
                                          
      void actuallyComputeStableTimestep(const ProcessorGroup*, 
                                         const PatchSubset* patch,  
                                         const MaterialSubset* matls,
                                         DataWarehouse*, 
                                         DataWarehouse*);

      void computeEquilibrationPressure(const ProcessorGroup*, 
                                        const PatchSubset* patch,  
                                        const MaterialSubset* matls,
                                        DataWarehouse*, 
                                        DataWarehouse*);
                                        
      void computeEquilPressure_1_matl(const ProcessorGroup*,  
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw, 
                                       DataWarehouse* new_dw);
      
      void computeVel_FC(const ProcessorGroup*, 
                         const PatchSubset*,                   
                         const MaterialSubset*,                
                         DataWarehouse*,                             
                         DataWarehouse*);  
                         
      void updateVel_FC(const ProcessorGroup*, 
                        const PatchSubset*,                   
                        const MaterialSubset*,                
                        DataWarehouse*,                             
                        DataWarehouse*,                   
                        bool);                    

      void computeTempFC(const ProcessorGroup*,  
                         const PatchSubset* patches,                    
                         const MaterialSubset*,               
                         DataWarehouse*,                         
                         DataWarehouse*);
                                                       
      template<class T> void computeTempFace(CellIterator it,
                                            IntVector adj_offset,
                                            constCCVariable<double>& rho_CC,
                                            constCCVariable<double>& Temp_CC,
                                            T& Temp_FC);
                                       
      template<class T> void computeVelFace(int dir, CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT, double gravity,
                                       constCCVariable<double>& rho_CC,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<Vector>& vel_CC,
                                       constCCVariable<double>& press_CC,
                                       T& vel_FC,
                                       T& gradP_FC,
                                       bool include_acc);
                                       
      template<class T> void updateVelFace(int dir, CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<double>& press_CC,
                                       T& vel_FC,
                                       T& grad_dp_FC);
                                       
      template<class V, class T>
        void add_vel_FC_exchange( CellIterator it,
                                       IntVector adj_offset,
                                       int numMatls,
                                       FastMatrix & K,
                                       double delT,
                                       StaticArray<constCCVariable<double> >& vol_frac_CC,
                                       StaticArray<constCCVariable<double> >& sp_vol_CC,
                                       V & vel_FC,
                                       T & sp_vol_FC,
                                       T & vel_FCME);
                                    

      void addExchangeContributionToFCVel(const ProcessorGroup*, 
                                          const PatchSubset* patch,  
                                          const MaterialSubset* matls,
                                          DataWarehouse*, 
                                          DataWarehouse*,
                                          bool);

      void computeDelPressAndUpdatePressCC(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse*,
                                           DataWarehouse*);

      void computePressFC(const ProcessorGroup*, 
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse*);

      template<class T> void computePressFace(CellIterator it, 
                                         IntVector adj_offset,
                                         constCCVariable<double>& sum_rho,
                                         constCCVariable<double>& press_CC,
                                         T& press_FC);

      void computeThermoTransportProperties(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ice_matls,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw);
                                           
      void accumulateMomentumSourceSinks(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse*);
      
      void accumulateEnergySourceSinks(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse*,
                                       DataWarehouse*);

      
      void computeLagrangianValues(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*);
      
      void computeLagrangianSpecificVolume(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse*,
                                           DataWarehouse*);

      void addExchangeToMomentumAndEnergy(const ProcessorGroup*,
                                          const PatchSubset*,
                                          const MaterialSubset*,
                                          DataWarehouse*,
                                          DataWarehouse*); 

      template< class V, class T>
      void update_q_CC(const std::string& desc,
                      CCVariable<T>& q_CC,
                      V& q_Lagrangian,
                      const CCVariable<T>& q_advected,
                      const CCVariable<double>& mass_new,
                      const CCVariable<double>& cv_new,
                      const Patch* patch); 
 
      void maxMach_on_Lodi_BC_Faces(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);
                       
      void advectAndAdvanceInTime(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*,
                                  DataWarehouse*);
                                  
      void conservedtoPrimitive_Vars(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

                                  
//__________________________________
//   RF TASKS    
      void actuallyComputeStableTimestepRF(const ProcessorGroup*, 
                                           const PatchSubset* patch,  
                                           const MaterialSubset* matls,
                                           DataWarehouse*, 
                                           DataWarehouse*);
                                                                 
      void computeRateFormPressure(const ProcessorGroup*,
                                   const PatchSubset* patch,
                                   const MaterialSubset* matls,
                                   DataWarehouse*, 
                                   DataWarehouse*);
                                   
      void computeDivThetaVel_CC(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);

      template<class T> void vel_PressDiff_FC(
                                       int dir, 
                                       CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT, double gravity,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<Vector>& vel_CC,
                                       constCCVariable<double>& vol_frac,
                                       constCCVariable<double>& rho_CC,
                                       constCCVariable<Vector>& D,
                                       constCCVariable<double>& speedSound,
                                       constCCVariable<double>& matl_press_CC,
                                       constCCVariable<double>& press_CC,
                                       T& vel_FC,
                                       T& pressDiff_FC);
                                                                    
      void computeFaceCenteredVelocitiesRF(const ProcessorGroup*, 
                                         const PatchSubset* patch,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse*);

      void accumulateEnergySourceSinks_RF(const ProcessorGroup*,  
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse* old_dw, 
                                          DataWarehouse* new_dw);   
                 
      void computeLagrangianSpecificVolumeRF(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse*,
                                           DataWarehouse*);  
                                           
      void addExchangeToMomentumAndEnergyRF(const ProcessorGroup*,
                                          const PatchSubset*,
                                          const MaterialSubset*,
                                          DataWarehouse*,
                                          DataWarehouse*);

//__________________________________ 
//  I M P L I C I T   I C E                                                                            
      void setupMatrix(const ProcessorGroup*,
                       const PatchSubset* patches,                      
                       const MaterialSubset* ,                          
                       DataWarehouse* old_dw,                           
                       DataWarehouse* new_dw);
                       
      void setupRHS(const ProcessorGroup*,
                    const PatchSubset* patches,                      
                    const MaterialSubset* ,                          
                    DataWarehouse* old_dw,                           
                    DataWarehouse* new_dw,
                    bool insideOuterIterLoop,
                    string computes_or_modifies);
                    
      void compute_maxRHS(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse*,
                          DataWarehouse* new_dw);
                       
       void updatePressure(const ProcessorGroup*,
                           const PatchSubset* patches,                      
                           const MaterialSubset* ,                          
                           DataWarehouse* old_dw,                           
                           DataWarehouse* new_dw);

      void computeDel_P(const ProcessorGroup*,
                         const PatchSubset* patches,                    
                         const MaterialSubset* ,                        
                         DataWarehouse* old_dw,                         
                         DataWarehouse* new_dw);
                                                     
      void implicitPressureSolve(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,     
                                 DataWarehouse* old_dw,     
                                 DataWarehouse* new_dw,     
                                 LevelP level,     
                                 const MaterialSubset*,
                                 const MaterialSubset*);

//__________________________________ 
//  I M P L I C I T   A M R I C E     
      void coarsen_delP(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse* new_dw,
                        const VarLabel* variable);
                           
                                  
      void matrixCoarseLevelIterator(Patch::FaceType patchFace,
                                       const Patch* coarsePatch,
                                       const Patch* finePatch,
                                       const Level* fineLevel,
                                       CellIterator& iter,
                                       bool& isRight_CP_FP_pair);
                                  
      void matrixBC_CFI_coarsePatch(const ProcessorGroup*,
                                    const PatchSubset* coarsePatches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw);
                                    
      void multiLevelPressureSolve(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,     
                                 DataWarehouse* old_dw,     
                                 DataWarehouse* new_dw,     
                                 GridP grid,     
                                 const MaterialSubset*,
                                 const MaterialSubset*);
                                 
       void bogus_imp_delP(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse*,
                           DataWarehouse* new_dw);
                                 
        void compute_refluxFluxes_RHS(const ProcessorGroup*,
                                      const PatchSubset* coarsePatches,
                                      const MaterialSubset*,
                                      DataWarehouse*,
                                      DataWarehouse* new_dw);
                                      
        void apply_refluxFluxes_RHS(const ProcessorGroup*,
                                    const PatchSubset* coarsePatches,
                                    const MaterialSubset*,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw);
                                                                              
//__________________________________ 
//   M O D E L S
                               
      void zeroModelSources(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*);

      void updateVolumeFraction(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse*,
                                DataWarehouse*);
                                
      void setNeedAddMaterialFlag(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*,
                                  DataWarehouse*);
                                
      void computeLagrangian_Transported_Vars(const ProcessorGroup*,  
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw, 
                                              DataWarehouse* new_dw);

//__________________________________ 
//   O T H E R                            
                               
      void TestConservation(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*);
                              
      void printData_problemSetup(const ProblemSpecP& prob_spec);

      void printData( int indx,
                      const  Patch* patch,
                      int include_GC,
                      const string& message1,
                      const string& message2, 
                      const  CCVariable<int>& q_CC);
                   
      void printData( int indx,
                      const  Patch* patch,
                      int include_GC,
                      const string& message1,
                      const string& message2, 
                      const  CCVariable<double>& q_CC); 

      void printVector( int indx,
                        const  Patch* patch,
                        int include_GC,
                        const string& message1,
                        const string& message2, 
                        int component, 
                        const CCVariable<Vector>& q_CC);

      void printStencil( int matl,
                         const Patch* patch,                 
                         int include_EC,                     
                         const string&    message1,          
                         const string&    message2,          
                         const CCVariable<Stencil7>& q_CC);   
                     
      void adjust_dbg_indices( const int include_EC,
                               const  Patch* patch,
                               const IntVector d_dbgBeginIndx,
                               const IntVector d_dbgEndIndx,  
                               IntVector& low,                 
                               IntVector& high);               
      
      void createDirs( const Patch* patch,
                        const string& desc, 
                        string& path);
      
      void find_gnuplot_origin_And_dx(const string variableType,
                                     const Patch*,
                                     IntVector&,
                                     IntVector&,
                                     double *,
                                     double *);

      void readData(const Patch* patch, int include_GC, const string& filename,
                  const string& var_name, CCVariable<double>& q_CC);
                 
      void hydrostaticPressureAdjustment(const Patch* patch, 
                      const CCVariable<double>& rho_micro_CC, 
                      CCVariable<double>& press_CC);
                      
      void getConstantExchangeCoefficients( FastMatrix& K,
                                    FastMatrix& H ); 
     
      void getVariableExchangeCoefficients( FastMatrix& ,
                                           FastMatrix& H,
                                           IntVector & c,
                                           StaticArray<constCCVariable<double> >& mass  );
                                                                       
      IntVector upwindCell_X(const IntVector& c, 
                             const double& var,              
                             double is_logical_R_face );    

      IntVector upwindCell_Y(const IntVector& c, 
                             const double& var,              
                             double is_logical_R_face );    
                                 
      IntVector upwindCell_Z(const IntVector& c, 
                             const double& var,              
                             double is_logical_R_face );   

      virtual bool needRecompile(double time, double dt,
                                 const GridP& grid);

      double getRefPress() const {
        return d_ref_press;
      }

      Vector getGravity() const {
        return d_gravity;
      }
                                 
      // Debugging switches
      bool switchDebug_Initialize;
      bool switchDebug_equil_press;
      bool switchDebug_vel_FC;
      bool switchDebug_Temp_FC;
      bool switchDebug_PressDiffRF;
      bool switchDebug_Exchange_FC;
      bool switchDebug_explicit_press;
      bool switchDebug_setupMatrix;
      bool switchDebug_setupRHS;
      bool switchDebug_updatePressure;
      bool switchDebug_computeDelP;
      bool switchDebug_PressFC;
      bool switchDebug_LagrangianValues;
      bool switchDebug_LagrangianSpecificVol;
      bool switchDebug_LagrangianTransportedVars;
      bool switchDebug_MomentumExchange_CC;
      bool switchDebug_Source_Sink;
      bool switchDebug_advance_advect;
      bool switchDebug_conserved_primitive;
      bool switchDebug_AMR_refine;
      bool switchDebug_AMR_refineInterface;
      bool switchDebug_AMR_coarsen;
      bool switchDebug_AMR_reflux;
      
      
      // debugging variables
      int d_dbgVar1;
      int d_dbgVar2;
      vector<IntVector>d_dbgIndices;
     
      // flags
      bool d_doAMR;
      bool d_doRefluxing;
      int  d_surroundingMatl_indx;
      bool d_impICE;
      bool d_recompile;
      bool d_canAddICEMaterial;
      bool d_with_mpm;
      bool d_with_rigid_mpm;
      
      int d_max_iter_equilibration;
      int d_max_iter_implicit;
      int d_iters_before_timestep_restart;
      double d_outer_iter_tolerance;
      
      // ADD HEAT VARIABLES
      vector<int>    d_add_heat_matls;
      vector<double> d_add_heat_coeff;
      double         d_add_heat_t_start, d_add_heat_t_final;
      bool           d_add_heat;
      
      double d_ref_press;
      
// For AMR staff

    protected:
    

    virtual void refineBoundaries(const Patch* patch,
                                  CCVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
                                  CCVariable<Vector>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
                                  SFCXVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
                                  SFCYVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
                                  SFCZVariable<double>& val,
                                  DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  int matl, double factor);
    virtual void addRefineDependencies(Task* task, const VarLabel* var,
                                       int step, int nsteps);

    MaterialSubset* d_press_matl;
    MaterialSet*    d_press_matlSet;

    private:
      friend class MPMICE;
      friend class AMRICE;
      friend class impAMRICE;
                   
       void printData_FC(int indx,
                      const  Patch* patch,
                      int include_GC,
                      const string& message1,
                      const string& message2, 
                      const SFCXVariable<double>& q_FC);
                    
       void printData_FC(int indx,
                      const  Patch* patch,
                      int include_GC,
                      const string& message1,
                      const string& message2, 
                      const SFCYVariable<double>& q_FC);
                    
       void printData_FC(int indx, 
                      const  Patch* patch,
                      int include_GC,
                      const string& message1,
                      const string& message2, 
                      const SFCZVariable<double>& q_FC);
                                
       template <class T>
       void printData_driver( int indx,
                              const  Patch* patch,
                              int include_GC,
                              const string& message1,
                              const string& message2,
                              const string& variableType, 
                              const  T& q_CC);
                              
      void printVector_driver( int indx,
                               const  Patch* patch,
                               int include_GC,
                               const string& message1,
                               const string& message2, 
                               int component, 
                               const CCVariable<Vector>& q_CC);
                              
       template <class T>                       
       void symmetryTest_driver( int indx,
                                 const  Patch* patch,
                                 const IntVector& cellShift,
                                 const string& message1,
                                 const string& message2, 
                                 const  T& q_CC);
               
       void symmetryTest_Vector( int indx,
                                 const  Patch* patch,
                                 const string& message1,
                                 const string& message2, 
                                 const CCVariable<Vector>& q_CC);
                                 
      ICELabel* lb; 
      MPMICELabel* MIlb;
      SimulationStateP d_sharedState;
      Output* dataArchiver;
      SchedulerP d_subsched;
      bool d_recompileSubsched;
      double d_EVIL_NUM;
      double d_SMALL_NUM; 
      double d_CFL;
      double d_delT_knob;
      int d_max_iceMatl_indx;
      Vector d_gravity;

      //__________________________________
      // needed by printData
      double d_dbgTime; 
      double d_dbgStartTime;
      double d_dbgStopTime;
      double d_dbgOutputInterval;
      double d_dbgNextDumpTime;
      double d_dbgSym_relative_tol;
      double d_dbgSym_absolute_tol;
      double d_dbgSym_cutoff_value;
      
      bool   d_dbgGnuPlot;
      bool   d_dbgTime_to_printData;
      bool   d_dbgSymmetryTest;
      vector<IntVector> d_dbgBeginIndx;
      vector<IntVector> d_dbgEndIndx;
      IntVector d_dbgSymPlanes;
      vector<int> d_dbgMatls;
      vector<int> d_dbgLevel; 
      int d_dbgSigFigs;
      
      //__________________________________
      // Misc
      customBC_var_basket* d_customBC_var_basket;
      customInitialize_basket* d_customInitialize_basket;
      
      Advector* d_advector;
      int  d_OrderOfAdvection;
      bool d_useCompatibleFluxes;
      bool d_clampSpecificVolume;
      Turbulence* d_turbulence;
      AnalysisModule* d_analysisModule;
      
      std::string d_delT_scheme;
      
      // exchange coefficients
      ExchangeCoefficients* d_exchCoeff;
      
      // flags for the conservation test
       struct conservationTest_flags{
        bool onOff;
        bool mass;
        bool momentum;
        bool energy;
        bool exchange;
       };
       conservationTest_flags* d_conservationTest;

      ICE(const ICE&);
      ICE& operator=(const ICE&);
      
      SolverInterface* d_solver;
      SolverParameters* d_solver_parameters;    

      //______________________________________________________________________
      //        models
      std::vector<ModelInterface*> d_models;
      ModelInfo* d_modelInfo;
      
      struct TransportedVariable {
       const MaterialSubset* matls;
       const MaterialSet* matlSet;
       const VarLabel* var;
       const VarLabel* src;
       const VarLabel* var_Lagrangian;
       const VarLabel* var_adv;
      };
      struct AMR_refluxVariable {
       const MaterialSubset* matls;
       const MaterialSet* matlSet;
       const VarLabel* var;
       const VarLabel* var_adv;
       const VarLabel* var_X_FC_flux;
       const VarLabel* var_Y_FC_flux;
       const VarLabel* var_Z_FC_flux;
       
       const VarLabel* var_X_FC_corr;
       const VarLabel* var_Y_FC_corr;
       const VarLabel* var_Z_FC_corr;
      };
      
      class ICEModelSetup : public ModelSetup {
      public:
       ICEModelSetup();
       virtual ~ICEModelSetup();
       virtual void registerTransportedVariable(const MaterialSet* matlSet,
                                           const VarLabel* var,
                                           const VarLabel* src);
                                           
       virtual void registerAMR_RefluxVariable(const MaterialSet* matls,
						     const VarLabel* var);  
                                                                                        
       std::vector<TransportedVariable*> tvars;
       std::vector<AMR_refluxVariable*> d_reflux_vars;
      };
      ICEModelSetup* d_modelSetup;
      
      //______________________________________________________________________
      //      FUNCTIONS
      inline bool isEqual(const Vector& a, const Vector& b){
        return ( a.x() == b.x() && a.y() == b.y() && a.z() == b.z());
      };
      
      inline bool isEqual(const double a, const double b){
        return a == b;
      };
      /*_____________________________________________________________________
       Purpose~  Returns if any CCVariable == value.  Useful for detecting
       uninitialized variables
       _____________________________________________________________________  */
      template<class T>
        bool isEqual(T value,
                     CellIterator &iter,
                     CCVariable<T>& q_CC, 
                     IntVector& cell )
      {   
        for(; !iter.done(); iter++){
          IntVector c = *iter;
          if (isEqual(q_CC[c],value)){
            cell = c;
            return true;   
          } 
        }
        cell = IntVector(0,0,0);
        return false;
      }
    };

} // End namespace Uintah

#endif
