
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/CFDInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/Vector.h>


namespace Uintah {

using namespace SCIRun;

    class ICE : public UintahParallelComponent, public CFDInterface {
    public:
      ICE(const ProcessorGroup* myworld);
      virtual ~ICE();
      
      struct fflux { double d_fflux[6]; };
      struct eflux { double d_eflux[12]; };
      
      virtual void problemSetup(const ProblemSpecP& params, 
				GridP& grid,
				SimulationStateP&);
      
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&,
				      DataWarehouseP&);
      
      
      virtual void scheduleComputeStableTimestep(const LevelP&,
						 SchedulerP&,
						 DataWarehouseP&);
      
      virtual void scheduleTimeAdvance(double t, 
				       double dt, 
				       const LevelP&, 
				       SchedulerP&,
				       DataWarehouseP&, 
				       DataWarehouseP&);
      
      void scheduleStep1a(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep1b(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep1c(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep1d(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep2(const Patch* patch, 
			 SchedulerP&,
			 DataWarehouseP&, 
			 DataWarehouseP&);
      
      void scheduleStep3(const Patch* patch, 
			 SchedulerP&,
			 DataWarehouseP&, 
			 DataWarehouseP&);
      
      void scheduleStep4a(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep4b(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep5a(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);
      
      void scheduleStep5b(const Patch* patch, 
			  SchedulerP&,
			  DataWarehouseP&, 
			  DataWarehouseP&);

      void scheduleStep6and7(const Patch* patch, 
			     SchedulerP&,
			     DataWarehouseP&, 
			     DataWarehouseP&);

      void setICELabel(ICELabel* Ilb)
	{
	  lb = Ilb;
	};
      
    public:
      
      void actuallyInitialize(const ProcessorGroup*,
			      const Patch* patch,
			      DataWarehouseP&  old_dw,
			      DataWarehouseP& new_dw);
      
      void actuallyComputeStableTimestep(const ProcessorGroup*,
					 const Patch* patch,
					 DataWarehouseP&,
					 DataWarehouseP&);
      
      
      
      void actually_Bottom_of_main_loop(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP&,
					DataWarehouseP&);
      
      void actually_Top_of_main_loop(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP&,
				     DataWarehouseP&);
      
      // compute speedSound
      void actuallyStep1a(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      // calculateEquilibrationPressure
      void actuallyStep1b(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      // computeFCVelocity
      void actuallyStep1c(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      // momentumExchangeFCVelocity
      void actuallyStep1d(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      // computeDivFCVelocity
      // computeExplicitDelPress
      void actuallyStep2(const ProcessorGroup*,
			 const Patch* patch,
			 DataWarehouseP&,
			 DataWarehouseP&);
      
      // computeFaceCenteredPressure
      void actuallyStep3(const ProcessorGroup*,
			 const Patch* patch,
			 DataWarehouseP&,
			 DataWarehouseP&);
      
      void actuallyStep4a(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      void actuallyStep4b(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      void actuallyStep5a(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      void actuallyStep5b(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP&,
			  DataWarehouseP&);
      
      void actuallyStep6and7(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP&,
			     DataWarehouseP&);
      
      
    private:
      friend const TypeDescription* fun_getTypeDescription(fflux*);
      friend const TypeDescription* fun_getTypeDescription(eflux*);
      
      void setBC(CCVariable<double>& variable,const std::string& type, 
		 const Patch* p);
      void setBC(CCVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p);
      
      void setBC(SFCXVariable<double>& variable,const std::string& type, 
		 const Patch* p);
      void setBC(SFCXVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p);
      void setBC(SFCYVariable<double>& variable,const std::string& type, 
		 const Patch* p);
      void setBC(SFCYVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p);
      void setBC(SFCZVariable<double>& variable,const std::string& type, 
		 const Patch* p);
      void setBC(SFCZVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p);
      
      void influxOutfluxVolume(const SFCXVariable<double>& uvel_CC,
			       const SFCYVariable<double>& vvel_CC,
			       const SFCZVariable<double>& wvel_CC,
			       const double& delT, const Patch* patch,
			       CCVariable<fflux>& OFS, CCVariable<eflux>& OFE,
			       CCVariable<fflux>& IFS, CCVariable<eflux>& IFE);
      
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
			const CCVariable<fflux>& IFS,
			const CCVariable<eflux>& IFE,
			CCVariable<fflux>& q_out,
			CCVariable<eflux>& q_out_EF,
			CCVariable<fflux>& q_in,
			CCVariable<eflux>& q_in_EF,
			CCVariable<double>& q_advected);
      
      void qOutfluxFirst(const CCVariable<double>& q_CC,
			 const Patch* patch,
			 CCVariable<fflux>& q_out,
			 CCVariable<eflux>& q_out_EF);
      
      
      void qInflux(const CCVariable<fflux>& q_out,
		   const CCVariable<eflux>& q_out_EF,
		   const Patch* patch,
		   CCVariable<fflux>& q_in,
		   CCVariable<eflux>& q_in_EF);
      
      void qOutfluxSecond(CCVariable<fflux>& OFS,
			  CCVariable<fflux>& IFS,
			  CCVariable<fflux>& r_out_x,
			  CCVariable<fflux>& r_out_y,
			  CCVariable<fflux>& r_out_z,
			  CCVariable<eflux>& r_out_x_CF,
			  CCVariable<eflux>& r_out_y_CF,
			  CCVariable<eflux>& r_out_z_CF,
			  const Vector& dx);
      
      ICELabel* lb; 
      SimulationStateP d_sharedState;
      double d_SMALL_NUM;
      double d_pressure;
      Vector d_K_mom, d_K_heat; // exchange coefficients -- off diagonal terms
      
      ICE(const ICE&);
      ICE& operator=(const ICE&);
      
      const VarLabel* IFS_CCLabel;
      const VarLabel* OFS_CCLabel;
      const VarLabel* IFE_CCLabel;
      const VarLabel* OFE_CCLabel;
      const VarLabel* q_outLabel;
      const VarLabel* q_out_EFLabel;
      const VarLabel* q_inLabel;
      const VarLabel* q_in_EFLabel;
      
    };
    
#define TOP        0          /* index used to designate the top cell face    */
#define BOTTOM     1          /* index used to designate the bottom cell face */
#define RIGHT      2          /* index used to designate the right cell face  */
#define LEFT       3          /* index used to designate the left cell face   */
#define FRONT      4          /* index used to designate the front cell face  */
#define BACK       5          /* index used to designate the back cell face   */
    
    // Definitions for edge corners (just called corners in Todd's ICE)
    // T=TOP, B=BOTTOM, R=RIGHT, L=LEFT, F=FRONT, b=BACK
#define TR   0 
#define TL   1
#define BL   2
#define BR   3
#define TF   4
#define Tb   5
#define BF   6
#define Bb   7
#define FR   8
#define FL   9
#define bR   10
#define bL   11

} // End namespace Uintah

#endif
