
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
/*`==========TESTING==========*/
#include <Packages/Uintah/CCA/Ports/SolverInterface.h> 
/*==========TESTING==========`*/
namespace Uintah {
/*`==========TESTING==========*/
  class SolverInterface;
  class SolverParameters; 
/*==========TESTING==========`*/
using namespace SCIRun;
    
    class ICE : public UintahParallelComponent, public SimulationInterface {
    public:
      ICE(const ProcessorGroup* myworld);
      virtual ~ICE();
      
      virtual void problemSetup(const ProblemSpecP& params, 
                                GridP& grid,
                                SimulationStateP&);
      
      virtual void scheduleInitialize(const LevelP& level, 
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
                                              
      void scheduleComputeFaceCenteredVelocities(SchedulerP&, 
                                                const PatchSet*,
                                                const MaterialSubset*,
                                                const MaterialSubset*,
                                                const MaterialSubset*,
                                                const MaterialSet*);
      
      void scheduleAddExchangeContributionToFCVel(SchedulerP&, 
                                            const PatchSet*,
                                            const MaterialSet*);
      
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
      
      void scheduleAccumulateMomentumSourceSinks(SchedulerP&, 
                                            const PatchSet*,
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
                                                   const MaterialSet*);

      void scheduleAddExchangeToMomentumAndEnergy(SchedulerP& sched,
                                                  const PatchSet*,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSet* );
                                                        
      void scheduleAdvectAndAdvanceInTime(SchedulerP&, 
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSubset*,
                                          const MaterialSet*); 

      void scheduleMassExchange(SchedulerP&, 
                                const PatchSet*,
                                const MaterialSet*);
                             
      void schedulePrintConservedQuantities(SchedulerP&, const PatchSet*,
                                       const MaterialSubset*,
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
      
      void computeFaceCenteredVelocities(const ProcessorGroup*, 
                                         const PatchSubset* patch,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse*);

      template<class T> void computeVelFace(int dir, CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT, double gravity,
                                       constCCVariable<double>& rho_CC,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<Vector>& vel_CC,
                                       constCCVariable<double>& press_CC,
                                       T& vel_FC);

      void addExchangeContributionToFCVel(const ProcessorGroup*, 
                                          const PatchSubset* patch,  
                                          const MaterialSubset* matls,
                                          DataWarehouse*, 
                                          DataWarehouse*);

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
                                  
      void advectAndAdvanceInTime(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*,
                                  DataWarehouse*);
                                  
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
                          
      void scheduleImplicitPressureSolve(SchedulerP&,
                                         const LevelP&,                       
                                         SolverInterface*,                   
                                         const SolverParameters*, 
                                         const PatchSet* patches, 
                                         const MaterialSubset*,                  
                                         const MaterialSubset*,           
                                         const MaterialSubset*,           
                                         const MaterialSet* );             
                                            
      void setupMatrix(const ProcessorGroup*,
                       const PatchSubset* patches,                      
                       const MaterialSubset* ,                          
                       DataWarehouse* old_dw,                           
                       DataWarehouse* new_dw);
                       
       void updatePressure(const ProcessorGroup*,
                           const PatchSubset* patches,                      
                           const MaterialSubset* ,                          
                           DataWarehouse* old_dw,                           
                           DataWarehouse* new_dw); 
                                                
      void petscExample(const PatchSubset* patches);
      
      void petscMapping( const PatchSubset* patches,
                         int numlrows,
                         int numlcolumns,
                         int gobalrows,
                         int gobalcolumns);
//__________________________________ 
//   O T H E R                            
                               
      void printConservedQuantities(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*,
                                    DataWarehouse*);

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
                        const string& message2, int component, 
                        const CCVariable<Vector>& q_CC);
                     
      void adjust_dbg_indices( const int include_EC,
                               const  Patch* patch,
                               const IntVector d_dbgBeginIndx,
                               const IntVector d_dbgEndIndx,  
                               IntVector& low,                 
                               IntVector& high);               
      
      void createDirs( const string& desc, string& path);
      
      void find_gnuplot_origin_And_dx(const Patch*,
                                     const IntVector,
                                     const IntVector,
                                     double *,
                                     double *);
                                                           
      void Message(int abort, const string& message1, const string& message2,
                 const string& message3);

      void readData(const Patch* patch, int include_GC, const string& filename,
                  const string& var_name, CCVariable<double>& q_CC);
                 
      void hydrostaticPressureAdjustment(const Patch* patch, 
                      const CCVariable<double>& rho_micro_CC, 
                      CCVariable<double>& press_CC);
                      
      void massExchange(const ProcessorGroup*,
                        const PatchSubset* patch, 
                        const MaterialSubset* matls,  
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw);  
                                                      
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
     
      // Debugging switches
      bool switchDebugInitialize;
      bool switchDebug_EQ_RF_press;
      bool switchDebug_vel_FC;
      bool switchDebug_PressDiffRF;
      bool switchDebug_Exchange_FC;
      bool switchDebug_explicit_press;
      bool switchDebug_PressFC;
      bool switchDebugLagrangianValues;
      bool switchDebugLagrangianSpecificVol;
      bool switchDebugMomentumExchange_CC;
      bool switchDebugSource_Sink;
      bool switchDebug_advance_advect;
      bool switchDebug_advectQFirst;
      bool switchTestConservation; 
      
      bool d_massExchange;
      bool d_RateForm;
      bool d_EqForm;
      bool d_add_delP_Dilatate; 
      bool d_impICE;
      int d_dbgVar1;
      int d_dbgVar2;
      int d_max_iter_equilibration;
      vector<int>    d_add_heat_matls;
      vector<double> d_add_heat_coeff;
      double         d_add_heat_iters;
      bool           d_add_heat;
     
    private:
      friend class MPMICE;
      
      void computeTauX( const Patch* patch,
                        const CCVariable<double>& rho_CC,     
                        const CCVariable<double>& sp_vol_CC,  
                        const CCVariable<Vector>& vel_CC,     
                        const double viscosity,               
                        const Vector dx,                      
                        SFCXVariable<Vector>& tau_X_FC);      
                          
      void computeTauY( const Patch* patch,
                        const CCVariable<double>& rho_CC,     
                        const CCVariable<double>& sp_vol_CC,  
                        const CCVariable<Vector>& vel_CC,     
                        const double viscosity,               
                        const Vector dx,                      
                        SFCYVariable<Vector>& tau_Y_FC);      
                          
      void computeTauZ( const Patch* patch,
                        const CCVariable<double>& rho_CC,     
                        const CCVariable<double>& sp_vol_CC,  
                        const CCVariable<Vector>& vel_CC,     
                        const double viscosity,               
                        const Vector dx,                      
                        SFCZVariable<Vector>& tau_Z_FC);      
                   
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
      
      ICELabel* lb; 
      MPMICELabel* MIlb;
      SimulationStateP d_sharedState;
      Output* dataArchiver;
      double d_SMALL_NUM; 
      double d_TINY_RHO;
      double d_pressure;
      double d_initialDt;
      double d_CFL;
      double d_dbgTime; 
      double d_dbgStartTime;
      double d_dbgStopTime;
      double d_dbgOutputInterval;
      double d_dbgNextDumpTime;
      double d_dbgOldTime;
      bool   d_dbgGnuPlot;
      IntVector d_dbgBeginIndx;
      IntVector d_dbgEndIndx; 
      vector<int> d_dbgMatls; 
      int d_dbgSigFigs;
      
      Advector* d_advector;
      
     // exchange coefficients -- off diagonal terms
      vector<double> d_K_mom, d_K_heat;

      ICE(const ICE&);
      ICE& operator=(const ICE&);
/*`==========TESTING==========*/
      SolverInterface* solver;
      SolverParameters* solver_parameters; 
/*==========TESTING==========`*/    
      
    };

#define SURROUND_MAT 0        /* Mat index of surrounding material, assumed */
 

} // End namespace Uintah

#endif







