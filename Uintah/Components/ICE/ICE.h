
#ifndef UINTAH_HOMEBREW_ICE_H
#define UINTAH_HOMEBREW_ICE_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CellIterator.h>
#include <SCICore/Geometry/Vector.h>

using SCICore::Geometry::Vector;

namespace Uintah {
  namespace ICESpace {
    
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
      
      void setBC(FCVariable<double>& variable,const std::string& type, 
		 const Patch* p);
      void setBC(FCVariable<double>& variable,const std::string& type,
		 const std::string& comp, const Patch* p);
      
      void influxOutfluxVolume(const FCVariable<double>& uvel_CC,
			       const FCVariable<double>& vvel_CC,
			       const FCVariable<double>& wvel_CC,
			       const double& delT, const Patch* patch,
			       CCVariable<fflux>& OFS, CCVariable<eflux>& OFE,
			       CCVariable<fflux>& IFS, CCVariable<eflux>& IFE);
      
      void outflowVolCentroid(const FCVariable<double>& uvel_CC,
			      const FCVariable<double>& vvel_CC,
			      const FCVariable<double>& wvel_CC,
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
      
      struct ic {
	Vector d_velocity;
	double d_micro_density;
	double d_temperature;
	double d_volume_fraction;
      };
      
      vector<struct ic> d_ic;
      
      
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
  }
}

#endif

// $Log$
// Revision 1.37  2000/11/21 21:53:24  jas
// Added methods for the different schedules that make up scheduleTimeAdvance.
//
// Revision 1.36  2000/11/15 00:51:54  guilkey
// Changed code to take advantage of the ICEMaterial stuff I committed
// recently in preparation for coupling the two codes.
//
// Revision 1.35  2000/11/14 04:02:11  jas
// Added getExtraCellIterator and things now appear to be working up to
// face centered velocity calculations.
//
// Revision 1.34  2000/11/02 21:33:06  jas
// Added new bc implementation.  Things now work thru step 1b.  Neumann bcs
// are now set correctly.
//
// Revision 1.33  2000/10/31 04:16:17  jas
// Fixed some errors in speed of sound and equilibration pressure calculation.
// Added initial conditions.
//
// Revision 1.32  2000/10/25 22:22:13  jas
// Change the fflux and eflux struct so that the data members begin with d_.
// This makes g++ happy.
//
// Revision 1.31  2000/10/24 23:07:21  guilkey
// Added code for steps6and7.
//
// Revision 1.30  2000/10/20 23:58:55  guilkey
// Added part of advection code.
//
// Revision 1.29  2000/10/18 21:02:17  guilkey
// Added code for steps 4 and 5.
//
// Revision 1.28  2000/10/16 20:31:00  guilkey
// Step3 added
//
// Revision 1.27  2000/10/16 19:10:34  guilkey
// Combined step1e with step2 and eliminated step1e.
//
// Revision 1.26  2000/10/16 18:32:40  guilkey
// Implemented "step1e" of the ICE algorithm.
//
// Revision 1.25  2000/10/13 00:01:11  guilkey
// More work on ICE
//
// Revision 1.24  2000/10/10 20:35:07  jas
// Move some stuff around.
//
// Revision 1.23  2000/10/04 23:38:21  jas
// All of the steps are in place with just dummy functions.  delT is
// hardwired in for the moment so that we can actually do multiple
// time steps with empty functions.
//
// Revision 1.22  2000/10/04 20:19:03  jas
// Get rid of Labels.  Now in ICELabel.
//
// Revision 1.21  2000/10/04 19:26:46  jas
// Changes to get ICE into UCF conformance.  Only skeleton for now.
//





