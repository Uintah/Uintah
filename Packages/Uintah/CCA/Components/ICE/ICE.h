
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Ports/CFDInterface.h>
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

namespace Uintah {

using namespace SCIRun;
    
    class ICE : public UintahParallelComponent, public CFDInterface {
    public:
      ICE(const ProcessorGroup* myworld);
      virtual ~ICE();
      
      struct fflux { double d_fflux[6]; };          //face flux
      struct eflux { double d_eflux[12]; };         //edge flux
      struct cflux { double d_cflux[8]; };          //corner flux
      
      virtual void problemSetup(const ProblemSpecP& params, 
                                GridP& grid,
				    SimulationStateP&);
      
      virtual void scheduleInitialize(const LevelP& level, 
                                      SchedulerP&);
      
      virtual void scheduleComputeStableTimestep(const LevelP&,
                                                SchedulerP&);
      
      virtual void scheduleTimeAdvance(double t, 
                                      double dt,
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
               
      void setBC(CCVariable<double>& press_CC, CCVariable<double>& rho,
               const std::string& type, const Patch* p, const int mat_id);
               
      void setBC(CCVariable<double>& press_CC, 
                StaticArray<CCVariable<double> >& rho_micro_CC,
                StaticArray<CCVariable<double> >& rho_CC,
                StaticArray<CCVariable<double> >& vol_frac_CC,
                StaticArray<CCVariable<Vector> >& vel_CC,
                DataWarehouse* old_dw,
                const string& kind, 
                const Patch* patch, 
                const int mat_id);

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
                         
      void backoutGCPressFromVelFC(const Patch* patch,
                                Patch::FaceType face,
                                DataWarehouse* old_dw,
                                CCVariable<double>& press_CC, 
                          const StaticArray<CCVariable<double> >& rho_micro_CC,
                          const StaticArray<CCVariable<double> >& rho_CC,
                          const StaticArray<CCVariable<double> >& vol_frac_CC,
                          const StaticArray<CCVariable<Vector> >& vel_CC);
                             
      void getExchangeCoefficients( DenseMatrix& K,
                                    DenseMatrix& H );  
                                        
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
      friend const TypeDescription* fun_getTypeDescription(fflux*);
      friend const TypeDescription* fun_getTypeDescription(eflux*);
      friend const TypeDescription* fun_getTypeDescription(cflux*);
      
      friend class MPMICE;
      
      void influxOutfluxVolume(const SFCXVariable<double>& uvel_CC,
			       const SFCYVariable<double>& vvel_CC,
			       const SFCZVariable<double>& wvel_CC,
			       const double& delT, const Patch* patch,
			       CCVariable<fflux>& OFS, 
			       CCVariable<eflux>& OFE,
			       CCVariable<cflux>& OFC);
      
      void outflowVolCentroid(const SFCXVariable<double>& uvel_CC,
			      const SFCYVariable<double>& vvel_CC,
			      const SFCZVariable<double>& wvel_CC,
			      const double& delT, const Vector& dx,
			      CCVariable<fflux>& r_out_x,
			      CCVariable<fflux>& r_out_y,
			      CCVariable<fflux>& r_out_z,
			      CCVariable<eflux>& r_out_x_CF,
			      CCVariable<eflux>& r_out_y_CF,
			      CCVariable<eflux>& r_out_z_CF);
      
      void advectQFirst(const CCVariable<double>& q_CC,
			const Patch* patch,
			const CCVariable<fflux>& OFS,
			const CCVariable<eflux>& OFE,
			const CCVariable<cflux>& OFC,
			CCVariable<double>& q_advected);
      
      void advectQFirst(const CCVariable<Vector>& q_CC,
			const Patch* patch,
			const CCVariable<fflux>& OFS,
			const CCVariable<eflux>& OFE,
			const CCVariable<cflux>& OFC,
			CCVariable<Vector>& q_advected);

      void qOutfluxFirst(const CCVariable<double>& q_CC,const Patch* patch,
			 CCVariable<fflux>& q_out,
			 CCVariable<eflux>& q_out_EF,
			 CCVariable<cflux>& q_out_CF);
      
      
      void qInfluxFirst(const CCVariable<fflux>& q_out,
		        const CCVariable<eflux>& q_out_EF,
		        const CCVariable<cflux>& q_out_CF, 
		        const Patch* patch,
		        CCVariable<fflux>& q_in,
		        CCVariable<eflux>& q_in_EF,
		        CCVariable<cflux>& q_in_CF);

      void qOutfluxSecond(CCVariable<fflux>& OFS,
			  CCVariable<fflux>& IFS,
			  CCVariable<fflux>& r_out_x,
			  CCVariable<fflux>& r_out_y,
			  CCVariable<fflux>& r_out_z,
			  CCVariable<eflux>& r_out_x_CF,
			  CCVariable<eflux>& r_out_y_CF,
			  CCVariable<eflux>& r_out_z_CF,
			  const Vector& dx);

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

      
     // exchange coefficients -- off diagonal terms
      vector<double> d_K_mom, d_K_heat;

      ICE(const ICE&);
      ICE& operator=(const ICE&);
      
      const VarLabel* IFS_CCLabel;
      const VarLabel* OFS_CCLabel;
      const VarLabel* IFE_CCLabel;
      const VarLabel* OFE_CCLabel;
      const VarLabel* IFC_CCLabel;
      const VarLabel* OFC_CCLabel;
      const VarLabel* q_outLabel;
      const VarLabel* q_out_EFLabel;
      const VarLabel* q_out_CFLabel;
      const VarLabel* q_inLabel;
      const VarLabel* q_in_EFLabel;
      const VarLabel* q_in_CFLabel;
      
    };

 /*______________________________________________________________________
 *      Needed by Advection Routines
 *______________________________________________________________________*/   
#define TOP        0          /* index used to designate the top cell face    */
#define BOTTOM     1          /* index used to designate the bottom cell face */
#define RIGHT      2          /* index used to designate the right cell face  */
#define LEFT       3          /* index used to designate the left cell face   */
#define FRONT      4          /* index used to designate the front cell face  */
#define BACK       5          /* index used to designate the back cell face   */
#define SURROUND_MAT 0        /* Mat index of surrounding material, assumed */  
            //__________________________________
            //   E D G E   F L U X E S
#define TOP_R               0               /* edge on top right of cell    */
#define TOP_FR              1               /* edge on top front of cell    */
#define TOP_L               2               /* edge on top left of cell     */
#define TOP_BK              3               /* edge on top back of cell     */

#define BOT_R               4               /* edge on bottom right of cell */
#define BOT_FR              5               /* edge on bottom front of cell */
#define BOT_L               6               /* edge on bottom left of cell  */
#define BOT_BK              7               /* edge on bottom back of cell  */

#define RIGHT_BK            8               /* edge along right back of cell*/
#define RIGHT_FR            9               /* edge along right front of cell  */
#define LEFT_FR             10              /* edge along left front of cell*/
#define LEFT_BK             11              /* edge alone left back of cell  */
     
            //__________________________________
            //   C O R N E R   F L U X E S
#define TOP_R_BK            0               /* top, RIGHT, back corner      */
#define TOP_R_FR            1               /* top, RIGHT, front corner     */
#define TOP_L_BK            2               /* top, LEFT, back corner       */
#define TOP_L_FR            3               /* top, LEFT, front corner      */
#define BOT_R_BK            4               /* bottom, RIGHT, back corner   */
#define BOT_R_FR            5               /* bottom, RIGHT, front corner  */
#define BOT_L_BK            6               /* bottom, LEFT, back corner    */
#define BOT_L_FR            7               /* bottom, LEFT, front corner   */

} // End namespace Uintah

#endif
