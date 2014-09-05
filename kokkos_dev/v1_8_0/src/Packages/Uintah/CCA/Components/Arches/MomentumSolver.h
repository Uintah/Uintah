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
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized momentum matrix
      void sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);
 
      void sched_velocityLinearSolve(SchedulerP& sched, const PatchSet* patches,
				     const MaterialSet* matls,
				     int index);
      void solvePred(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized momentum matrix
      void sched_buildLinearMatrixPred(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);
      void solveCorr(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized momentum matrix
      void sched_buildLinearMatrixCorr(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);

      void solveInterm(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized momentum matrix
      void sched_buildLinearMatrixInterm(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);

      void solveVelHatPred(const LevelP& level,
			   SchedulerP&,
			   const int Runge_Kutta_current_step,
			   const bool Runge_Kutta_last_step);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized eqn
      void sched_buildLinearMatrixVelHatPred(SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls);
 
      void solveVelHatCorr(const LevelP& level,
			   SchedulerP&,
			   const int Runge_Kutta_current_step,
			   const bool Runge_Kutta_last_step);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized eqn
      void sched_buildLinearMatrixVelHatCorr(SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls);

      void solveVelHatInterm(const LevelP& level,
			   SchedulerP&,
			   const int Runge_Kutta_current_step,
			   const bool Runge_Kutta_last_step);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized eqn
      void sched_buildLinearMatrixVelHatInterm(SchedulerP&, const PatchSet* patches,
					     const MaterialSet* matls);
 
      void sched_averageRKHatVelocities(SchedulerP& sched,
					 const PatchSet* patches,
				 	 const MaterialSet* matls,
			   		 const int Runge_Kutta_current_step,
			   		 const int Runge_Kutta_last_step);


#ifdef PetscFilter
      inline void setDiscretizationFilter(Filter* filter) {
        d_discretize->setFilter(filter);
      }
#endif
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
			     int index);

      void buildLinearMatrixPred(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     int index);

      void buildLinearMatrixCorr(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     int index);
   
      void velocityLinearSolve(const ProcessorGroup* pc,
			       const PatchSubset* patches,
			       const MaterialSubset* /*matls*/,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw,
			       int index);

      void buildLinearMatrixInterm(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     int index);


      void buildLinearMatrixVelHatPred(const ProcessorGroup* pc,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse*,
				       DataWarehouse* );

      void buildLinearMatrixVelHatCorr(const ProcessorGroup* pc,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse*,
				       DataWarehouse*);

      void buildLinearMatrixVelHatInterm(const ProcessorGroup* pc,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse*,
				       DataWarehouse*);


      void averageRKHatVelocities(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw,
			   const int Runge_Kutta_current_step,
			   const int Runge_Kutta_last_step);

   
private:

      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;
      bool d_central;
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


}; // End class MomentumSolver
} // End namespace Uintah


#endif

