
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
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>

#define ADVECT_TEST
//#undef ADVECT_TEST

namespace Uintah {

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
      
      virtual void scheduleComputeStableTimestep(const LevelP&,
                                                SchedulerP&);
      
      virtual void scheduleTimeAdvance(
                                      const LevelP&,
				          SchedulerP&);
                                                     
      void scheduleComputeEquilibrationPressure(SchedulerP&, 
                                              const PatchSet*,
                                              const MaterialSubset*,
						    const MaterialSet*);
      
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
					         const MaterialSet*);
      
      void scheduleComputeLagrangianValues(SchedulerP&, 
                                          const PatchSet*,
                                          const MaterialSubset*,
					       const MaterialSet*);
                 
      void scheduleAddExchangeToMomentumAndEnergy(SchedulerP&, 
                                                  const PatchSet*,
						        const MaterialSet*);
      
      void scheduleAdvectAndAdvanceInTime(SchedulerP&, 
                                          const PatchSet*,
					       const MaterialSet*);

      void scheduleMassExchange(SchedulerP&, 
                                const PatchSet*,
				    const MaterialSet*);
                             
      void schedulePrintConservedQuantities(SchedulerP&, const PatchSet*,
					    const MaterialSubset*,
					    const MaterialSet*);
      
      void setICELabel(ICELabel* Ilb) {
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
      
      void addExchangeToMomentumAndEnergy(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse*,
                                          DataWarehouse*);
      
      void advectAndAdvanceInTime(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*,
                                  DataWarehouse*);
                                  
      void printConservedQuantities(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*,
                                    DataWarehouse*);
            
      void setBC(CCVariable<double>& variable,const std::string& type, 
		 const Patch* p, const int mat_id);
               
      void setBC(CCVariable<double>& press_CC, const CCVariable<double>& rho,
               const std::string& type, const Patch* p, const int mat_id);

      void setBC(CCVariable<Vector>& variable,const std::string& type,
		 const Patch* p, const int mat_id);
               
      void setBC(SFCXVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p, const int mat_id);
               
      void setBC(SFCYVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p, const int mat_id);
               
      void setBC(SFCZVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p, const int mat_id);   

      void setBC(SFCXVariable<Vector>& variable,const std::string& type,
		 const Patch* p, const int mat_id);   

      void printData(const  Patch* patch,int include_GC,char message1[],
		     char message2[], const  CCVariable<int>& q_CC);
                   
      void printData(const  Patch* patch,int include_GC,char message1[],
		     char message2[], const  CCVariable<double>& q_CC); 

      void printVector(const  Patch* patch,int include_GC,char message1[],
		       char message2[], int component, 
		       const CCVariable<Vector>& q_CC);
                   
      void Message(int abort, char message1[],char message2[],char message3[]);

      void readData(const Patch* patch, int include_GC, char filename[],
		    char var_name[], CCVariable<double>& q_CC);
                 
      void hydrostaticPressureAdjustment(const Patch* patch, 
                      const CCVariable<double>& rho_micro_CC, 
                      CCVariable<double>& press_CC);
                      
      void massExchange(const ProcessorGroup*,
                        const PatchSubset* patch, 
                        const MaterialSubset* matls,  
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw);  
                                                      
      void getExchangeCoefficients( DenseMatrix& K,
                                    DenseMatrix& H ); 

      void matrixInverse(  int numMatls,
                           DenseMatrix& a,
                           DenseMatrix& aInverse);
                          
      void matrixSolver( int numMatls,
                         DenseMatrix& a, 
                         vector<double>& b, 
                         vector<double>& X  );
                         
      void multiplyMatrixAndVector( int numMatls,
                                    DenseMatrix& a, 
                                    vector<double>& b, 
                                    vector<double>& X  );
                                                             
      double conditionNumber( const int numMatls,
                              const DenseMatrix& a);


                                        
      // Debugging switches
      bool switchDebugInitialize;
      bool switchDebug_equilibration_press;
      bool switchDebug_vel_FC;
      bool switchDebug_Exchange_FC;
      bool switchDebug_explicit_press;
      bool switchDebug_PressFC;
      bool switchDebugLagrangianValues;
      bool switchDebugMomentumExchange_CC;
      bool switchDebugSource_Sink;
      bool switchDebug_advance_advect;
      bool switchDebug_advectQFirst;
      bool switchTestConservation; 
      
      bool d_massExchange;
      
      int d_max_iter_equilibration;
     
    private:
      friend class MPMICE;
      
      void computeTauX_Components( const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const double viscosity,
                          const Vector dx,
                          SFCXVariable<Vector>& tau_X_FC);
                          
      void computeTauY_Components( const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const double viscosity,
                          const Vector dx,
                          SFCYVariable<Vector>& tau_Y_FC);
                          
      void computeTauZ_Components( const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const double viscosity,
                          const Vector dx,
                          SFCZVariable<Vector>& tau_Z_FC);
                   
       void printData_FC(const  Patch* patch,int include_GC,char message1[],
		      char message2[], const SFCXVariable<double>& q_FC);
                    
       void printData_FC(const  Patch* patch,int include_GC,char message1[],
		      char message2[], const SFCYVariable<double>& q_FC);
                    
       void printData_FC(const  Patch* patch,int include_GC,char message1[],
		      char message2[], const SFCZVariable<double>& q_FC);
      
      ICELabel* lb; 
      MPMICELabel* MIlb;
      SimulationStateP d_sharedState;
      Output* dataArchiver;
      double d_SMALL_NUM;
      double d_pressure;
      double d_initialDt;
      double d_CFL;
      double d_dbgTime; 
      double d_dbgStartTime;
      double d_dbgStopTime;
      double d_dbgOutputInterval;
      double d_dbgNextDumpTime;
      double d_dbgOldTime;

      Advector* d_advector;
      
     // exchange coefficients -- off diagonal terms
      vector<double> d_K_mom, d_K_heat;

      ICE(const ICE&);
      ICE& operator=(const ICE&);

      
    };

#define SURROUND_MAT 0        /* Mat index of surrounding material, assumed */
 

} // End namespace Uintah

#endif







