
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>


namespace Uintah { 
using namespace SCIRun;
 class ModelInfo; 
 class ModelInterface; 
    
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
                                       
//__________________________________ 
//  I M P L I C I T   I C E
                                         
      void scheduleSetupMatrix(  SchedulerP&,
                                 const LevelP&,                  
                                 const PatchSet*,
                                 const MaterialSubset*,              
                                 const MaterialSet*); 
                                 
      void scheduleSetupRHS(  SchedulerP&,
                              const LevelP&,                  
                              const PatchSet*, 
                              const MaterialSubset*,             
                              const MaterialSet*); 
                                                  
      void scheduleUpdatePressure(  SchedulerP&,
                                   const LevelP&,
                                   const PatchSet*,
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
                                   
      // Model support
      void scheduleModelMassExchange(SchedulerP& sched,
				     const LevelP& level,
				     const MaterialSet* matls);
      void scheduleModelMomentumAndEnergyExchange(SchedulerP& sched,
						  const LevelP& level,
						  const MaterialSet* matls);


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
      void setupMatrix(const ProcessorGroup*,
                       const PatchSubset* patches,                      
                       const MaterialSubset* ,                          
                       DataWarehouse* old_dw,                           
                       DataWarehouse* new_dw);
                       
      void setupRHS(const ProcessorGroup*,
                    const PatchSubset* patches,                      
                    const MaterialSubset* ,                          
                    DataWarehouse* old_dw,                           
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
      bool switchDebugMomentumExchange_CC;
      bool switchDebugSource_Sink;
      bool switchDebug_advance_advect;
      bool switchTestConservation; 
     
      
      bool d_massExchange;
      bool d_RateForm;
      bool d_EqForm;
      bool d_impICE;
      int d_dbgVar1;
      int d_dbgVar2;
      int d_max_iter_equilibration;
      int d_max_iter_implicit;
      double d_outer_iter_tolerance;
      
      // ADD HEAT VARIABLES
      vector<int>    d_add_heat_matls;
      vector<double> d_add_heat_coeff;
      double         d_add_heat_t_start, d_add_heat_t_final;
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
                        
      template <class T> 
      void q_conduction(CellIterator iter, 
                        IntVector adj_offset,
                        const double thermalCond,
                        const double dx,
                        const CCVariable<double>& rho_CC,      
                        const CCVariable<double>& sp_vol_CC,   
                        const CCVariable<double>& Temp_CC,
                        T& q_FC);
                        
      void computeQ_conduction_FC(DataWarehouse* new_dw,
                                  const Patch* patch,
                                  const CCVariable<double>& rho_CC,     
                                  const CCVariable<double>& sp_vol_CC,  
                                  const CCVariable<double>& Temp_CC,
                                  const double thermalConduct,
                                  SFCXVariable<double>& q_X_FC,
                                  SFCYVariable<double>& q_Y_FC,
                                  SFCZVariable<double>& q_Z_FC);    
                   
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
      double d_delT_knob;
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
      std::string d_advect_type;
      std::string d_delT_scheme;
      
     // exchange coefficients -- off diagonal terms
      vector<double> d_K_mom, d_K_heat;

      ICE(const ICE&);
      ICE& operator=(const ICE&);
      
      SolverInterface* solver;
      SolverParameters* solver_parameters;    

      // For attachable models
      std::vector<ModelInterface*> d_models;
      ModelInfo* d_modelInfo;
    };

#define SURROUND_MAT 0        /* Mat index of surrounding material, assumed */
 

} // End namespace Uintah

#endif
