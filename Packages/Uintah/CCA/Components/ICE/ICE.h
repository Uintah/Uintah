
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H

#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Components/ICE/customInitialize.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/LODI2.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/Turbulence.h>
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Vector.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>


#define MAX_MATLS 16

namespace Uintah { 
  using namespace SCIRun;
  class ModelInfo; 
  class ModelInterface; 
  class Turbulence;
    
    class ICE : public UintahParallelComponent, public SimulationInterface {
    public:
      ICE(const ProcessorGroup* myworld, const bool doAMR = false);
      virtual ~ICE();

      virtual bool restartableTimesteps();

      virtual double recomputeTimestep(double current_dt);
      
      virtual void problemSetup(const ProblemSpecP& params, 
                                GridP& grid,
                                SimulationStateP&);
      
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
                                        SchedulerP&, int step, int nsteps );
                                             
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
                                      const MaterialSet*,
                                      const bool);            
      
      void scheduleAddExchangeContributionToFCVel(SchedulerP&, 
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSet*, 
                                            const bool);
      
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
                                            const MaterialSet*,
                                            vector<PatchSubset*> &);
      
      void computesRequires_AMR_Refluxing(Task* t, 
                                          const double AMR_subCycleProgressVar,
                                          const MaterialSet* ice_matls);                                                                                         
      
      void scheduleAdvectAndAdvanceInTime(SchedulerP&, 
                                          const PatchSet*,
                                          const double AMR_subCycleProgressVar,
                                          const MaterialSubset*,
                                          const MaterialSubset*,
                                          const MaterialSubset*,
                                          const MaterialSet*); 
                             
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
                                 const MaterialSet*,
                                 const bool firstIter); 
                                 
      void scheduleSetupRHS(  SchedulerP&,                
                              const PatchSet*, 
                              const MaterialSubset*,             
                              const MaterialSet*,
                              const bool insideOuterIterLoop); 
                                                  
      void scheduleUpdatePressure(  SchedulerP&,
                                   const LevelP&,
                                   const PatchSet*,
                                   const MaterialSubset*,         
                                   const MaterialSubset*, 
                                   const MaterialSubset*,
                                   const MaterialSet*);
                                   
      void scheduleImplicitVel_FC(SchedulerP& sched,
                                  const PatchSet* patches,                 
                                  const MaterialSubset*,         
                                  const MaterialSubset*,         
                                  const MaterialSubset*,        
                                  const MaterialSet*,           
                                  const bool);  
                                                  
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
//   M O D E L S
      void scheduleComputeModelSources(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* matls);
      void scheduleUpdateVolumeFraction(SchedulerP& sched,
                                            const LevelP& level,
                                            const MaterialSet* matls);
                                        
      void scheduleComputeLagrangian_Transported_Vars(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet*);

      void setICELabel(ICELabel* Ilb) {
       delete lb;
       lb = Ilb;
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
      
      void computeVel_FC(const ProcessorGroup*, 
                         const PatchSubset*,                   
                         const MaterialSubset*,                
                         DataWarehouse*,                             
                         DataWarehouse*,                   
                         const bool);                      

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
                                       T& vel_FC);
                                       
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
                                          const bool);

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
                                  DataWarehouse*,
                                  const double AMR_subCycleProgressVar);
                                  
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
                       DataWarehouse* new_dw,
                       const bool firstIteration);
                       
      void setupRHS(const ProcessorGroup*,
                    const PatchSubset* patches,                      
                    const MaterialSubset* ,                          
                    DataWarehouse* old_dw,                           
                    DataWarehouse* new_dw,
                    const bool insideOuterIterLoop);
                       
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
                                 Scheduler* sched,
                                 const MaterialSubset*,
                                 const MaterialSubset*);
                                                
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
                                                           
      void Message(int abort, const string& message1, const string& message2,
                 const string& message3);

      void readData(const Patch* patch, int include_GC, const string& filename,
                  const string& var_name, CCVariable<double>& q_CC);
                 
      void hydrostaticPressureAdjustment(const Patch* patch, 
                      const CCVariable<double>& rho_micro_CC, 
                      CCVariable<double>& press_CC);
                      
      void getExchangeCoefficients( FastMatrix& K,
                                    FastMatrix& H ); 

      bool areAllValuesPositive( CCVariable<double> & src, 
                                 IntVector& neg_cell );
                                                                       
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
     
      // Debugging switches
      bool switchDebugInitialize;
      bool switchDebug_EQ_RF_press;
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
      bool switchDebugLagrangianValues;
      bool switchDebugLagrangianSpecificVol;
      bool switchDebugLagrangianTransportedVars;
      bool switchDebugMomentumExchange_CC;
      bool switchDebugSource_Sink;
      bool switchDebug_advance_advect;
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
      int  d_surroundingMatl_indx;
      bool d_RateForm;
      bool d_EqForm;
      bool d_impICE;
      bool d_recompile;
      bool d_canAddICEMaterial;
      
      int d_max_iter_equilibration;
      int d_max_iter_implicit;
      int d_iters_before_timestep_restart;
      double d_outer_iter_tolerance;
      
      // ADD HEAT VARIABLES
      vector<int>    d_add_heat_matls;
      vector<double> d_add_heat_coeff;
      double         d_add_heat_t_start, d_add_heat_t_final;
      bool           d_add_heat;
      
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

    private:
      friend class MPMICE;
      friend class AMRICE;
                   
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
      double d_EVIL_NUM;
      double d_SMALL_NUM; 
      double d_TINY_RHO;
      double d_initialDt;
      double d_CFL;
      double d_delT_knob;
      
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
      IntVector d_dbgBeginIndx;
      IntVector d_dbgEndIndx;
      IntVector d_dbgSymPlanes;
      vector<int> d_dbgMatls;
      int d_dbgLevel; 
      int d_dbgSigFigs;
      
      //__________________________________
      // Misc
      customBC_var_basket* d_customBC_var_basket;
      customInitialize_basket* d_customInitialize_basket;
      
      Advector* d_advector;
      bool d_useCompatibleFluxes;
      Turbulence* d_turbulence;
      std::string d_delT_scheme;
      
      // exchange coefficients
      vector<double> d_K_mom, d_K_heat;
      
      // convective ht model
      bool d_convective;
      int d_conv_fluid_matlindex;
      int d_conv_solid_matlindex;

       // flags for the conservation test
       struct conservationTest_flags{
        bool onOff;
        bool momentum;
        bool energy;
        bool exchange;
       };
       conservationTest_flags* d_conservationTest;

      ICE(const ICE&);
      ICE& operator=(const ICE&);
      
      SolverInterface* solver;
      SolverParameters* solver_parameters;    

      //______________________________________________________________________
      //        models
      std::vector<ModelInterface*> d_models;
      ModelInfo* d_modelInfo;
      
      struct TransportedVariable {
       const MaterialSubset* matls;
       const VarLabel* var;
       const VarLabel* src;
       const VarLabel* var_Lagrangian;
      };
      struct AMR_refluxVariable {
       const MaterialSubset* matls;
       const VarLabel* var_CC;
       const VarLabel* var_X_FC_flux;
       const VarLabel* var_Y_FC_flux;
       const VarLabel* var_Z_FC_flux;
      };
      
      class ICEModelSetup : public ModelSetup {
      public:
       ICEModelSetup();
       virtual ~ICEModelSetup();
       virtual void registerTransportedVariable(const MaterialSubset* matls,
                                           const VarLabel* var,
                                           const VarLabel* src);
                                           
       virtual void registerAMR_RefluxVariable(const MaterialSubset* matls,
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
                     const Patch* patch,
                     CCVariable<T>& q_CC, 
                     IntVector& cell )
      {   
        for (CellIterator iter = patch->getExtraCellIterator(); 
                         !iter.done(); iter++) {
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
