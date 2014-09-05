//----- Properties.h --------------------------------------------------

#ifndef Uintah_Component_Arches_Properties_h
#define Uintah_Component_Arches_Properties_h

/***************************************************************************
CLASS
    Properties
       Sets up the Properties ????
       
GENERAL INFORMATION
    Properties.h - Declaration of Properties class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#ifdef PetscFilter
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#endif
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Core/Geometry/IntVector.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
class MixingModel;
class TimeIntegratorLabel;
class PhysicalConstants;
class Properties {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      // Constructor taking
      //   [in] 

      Properties(const ArchesLabel* label, const MPMArchesLabel* MAlb,
                 PhysicalConstants* phys_const, bool calcReactingScalar,
		 bool calcEnthalpy, bool calcVariance);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      // Destructor

      ~Properties();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database

      void problemSetup(const ProblemSpecP& params);

      // GROUP: Compute properties 
      ///////////////////////////////////////////////////////////////////////
      // Compute properties for inlet/outlet streams

      void computeInletProperties(const InletStream& inStream,
				  Stream& outStream);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule the recomputation of proprties

      void sched_reComputeProps(SchedulerP&, const PatchSet* patches,
				const MaterialSet* matls,
				const TimeIntegratorLabel* timelabels,
			        bool modify_density, bool initialize,
                                bool d_EKTCorrection,
                                bool doing_EKT_now);


      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of proprties for the first actual time 
      // step in an MPMArches run

      void sched_computePropsFirst_mm(SchedulerP&, const PatchSet* patches,
				      const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the computation of density reference array here

      void sched_computeDenRefArray(SchedulerP&, const PatchSet* patches,
				    const MaterialSet* matls,
			            const TimeIntegratorLabel* timelabels);

      void sched_averageRKProps(SchedulerP&, const PatchSet* patches,
				const MaterialSet* matls,
			        const TimeIntegratorLabel* timelabels);

      void sched_saveTempDensity(SchedulerP&, const PatchSet* patches,
				const MaterialSet* matls,
			        const TimeIntegratorLabel* timelabels);

      void sched_computeDrhodt(SchedulerP& sched, const PatchSet* patches,
				const MaterialSet* matls,
			        const TimeIntegratorLabel* timelabels,
                                bool d_EKTCorrection,
                                bool doing_EKT_now);


      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get the number of mixing variables

      inline int getNumMixVars() const{ 
	return d_numMixingVars; 
      }

      // GROUP: Set Methods :
      ///////////////////////////////////////////////////////////////////////
      // Set the boundary consition pointer

      inline void setBC(BoundaryCondition* bc) {
	d_bc = bc;
      }

#ifdef PetscFilter
      inline void setFilter(Filter* filter) {
	d_filter = filter;
      }
#endif
      inline void set3dPeriodic(bool periodic) {
	d_3d_periodic = periodic;
      }

      inline double getAdiabaticAirEnthalpy() const{
	return d_H_air;
      }

      inline double getCarbonContent(double f) const{
	return d_carbon_fuel*f+d_carbon_air*(1.0-f);
      }

protected :

private:

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      // Carry out actual recomputation of properties

      void reComputeProps(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const TimeIntegratorLabel* timelabels,
			  bool modify_density, bool initialize,
                          bool d_EKTCorrection,
                          bool doing_EKT_now);

      ///////////////////////////////////////////////////////////////////////
      // Carry out actual computation of properties for the first actual
      // time step in an MPMArches run

      void computePropsFirst_mm(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Carry out actual computation of density reference array

      void computeDenRefArray(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw,
			      const TimeIntegratorLabel* timelabels);

      void averageRKProps(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const TimeIntegratorLabel* timelabels);

      void saveTempDensity(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const TimeIntegratorLabel* timelabels);

      void computeDrhodt(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const TimeIntegratorLabel* timelabels,
                          bool d_EKTCorrection,
                          bool doing_EKT_now);

      // GROUP: Constructors Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const Properties&   

      Properties(const Properties&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const Properties&   

      Properties& operator=(const Properties&);

private:

      // Variable labels used by simulation controller
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      bool d_reactingFlow;
      PhysicalConstants* d_physicalConsts;
      bool d_calcReactingScalar;
      bool d_calcEnthalpy;
      bool d_calcVariance;
      bool d_radiationCalc;
      bool d_DORadiationCalc;

      bool d_co_output;
      bool d_sulfur_chem;
      bool d_soot_precursors;

      bool d_filter_drhodt;
      bool d_first_order_drhodt;
      int d_numMixingVars;
      double d_opl;
      IntVector d_denRef;
      MixingModel* d_mixingModel;
      BoundaryCondition* d_bc;
      bool d_empirical_soot;
      bool d_3d_periodic;
      bool d_inverse_density_average;
      double d_H_air;
      bool d_tabulated_soot;
      double d_f_stoich, d_carbon_fuel, d_carbon_air;
#ifdef PetscFilter
      Filter* d_filter;
#endif
}; // end class Properties
} // End namespace Uintah


#endif

