//----- MomentumSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_MomentumSolver_h
#define Uintah_Components_Arches_MomentumSolver_h

/**************************************
CLASS
   MomentumSolver
   
   Class MomentumSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   MomentumSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class MomentumSolver linearizes and solves momentum
   equation on a grid hierarchy


WARNING
   none

************************************************************************/
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
namespace Uintah {
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
class TurbulenceModel;
class PhysicalConstants;
class Source;
#ifdef PetscFilter
class Filter;
#endif
class BoundaryCondition;
class LinearSolver;
class TimeIntegratorLabel;

class MomentumSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of the Momentum solver.
      // PRECONDITIONS
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      MomentumSolver(const ArchesLabel* label, const MPMArchesLabel* MAlb,
		     TurbulenceModel* turb_model, 
		     BoundaryCondition* bndry_cond,
		     PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~MomentumSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of the linearized momentum equation. 
      void solve(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 const TimeIntegratorLabel* timelabels,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized momentum matrix
      void sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
		 		   const TimeIntegratorLabel* timelabels,
				   int index);
 

      void solveVelHat(const LevelP& level,
		       SchedulerP&,
		       const TimeIntegratorLabel* timelabels);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized eqn
      void sched_buildLinearMatrixVelHat(SchedulerP&, const PatchSet* patches,
					 const MaterialSet* matls,
					 const TimeIntegratorLabel* timelabels);
 
      void sched_averageRKHatVelocities(SchedulerP& sched,
					const PatchSet* patches,
				 	const MaterialSet* matls,
				        const TimeIntegratorLabel* timelabels);

	void sched_computeNonlinearTerms(SchedulerP&, 
					const PatchSet* patches,
					const MaterialSet* matls,
					const ArchesLabel* d_lab,
					const TimeIntegratorLabel* timelabels);


#ifdef PetscFilter
      inline void setDiscretizationFilter(Filter* filter) {
        d_discretize->setFilter(filter);
      }
#endif
      const bool& getPressureCorrectionFlag() const
	{
	  return d_pressure_correction;
	}
protected: 

private:

      // GROUP: Constructors (private):
      ////////////////////////////////////////////////////////////////////////
      // Default constructor.
      MomentumSolver();

      // GROUP: Action Methods (private):
      ///////////////////////////////////////////////////////////////////////
      // Actually build the linearized momentum matrix
      void buildLinearMatrix(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
		 	     const TimeIntegratorLabel* timelabels,
			     int index);


      void buildLinearMatrixVelHat(const ProcessorGroup* pc,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*,
				   DataWarehouse*,
				   const TimeIntegratorLabel* timelabels);

      void averageRKHatVelocities(const ProcessorGroup*,
			          const PatchSubset* patches,
			          const MaterialSubset* matls,
			          DataWarehouse* old_dw,
			          DataWarehouse* new_dw,
				  const TimeIntegratorLabel* timelabels);

	void computeNonlinearTerms(const ProcessorGroup* pc,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw,
					const ArchesLabel* d_lab,
					const TimeIntegratorLabel* timelabels);

	void filterNonlinearTerms(const ProcessorGroup* pc,
					const Patch* patch,
					int index,
					CellInformation* cellinfo,
					ArchesVariables* vars);

   
private:

      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;
      bool d_central;
      bool d_pressure_correction;
      // computes coefficients
      Discretization* d_discretize;
      // computes sources
      Source* d_source;
      // linear solver
      LinearSolver* d_linearSolver;
      // turbulence model
      TurbulenceModel* d_turbModel;
      // boundary condition
      BoundaryCondition* d_boundaryCondition;
      // physical constants
      PhysicalConstants* d_physicalConsts;
      bool d_3d_periodic;
      bool d_filter_divergence_constraint;


}; // End class MomentumSolver
} // End namespace Uintah


#endif

