//----- ReactiveScalarSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ReactiveScalarSolver_h
#define Uintah_Component_Arches_ReactiveScalarSolver_h

/**************************************
CLASS
   ReactiveScalarSolver
   
   Class ReactiveScalarSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   ReactiveScalarSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ReactiveScalarSolver linearizes and solves scalar
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

namespace Uintah {
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class ReactiveScalarSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of the ReactiveScalar solver.
      // PRECONDITIONS
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      ReactiveScalarSolver(const ArchesLabel* label, const MPMArchesLabel* MAlb, 
		   TurbulenceModel* turb_model, 
		   BoundaryCondition* bndry_cond,
		   PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~ReactiveScalarSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of linearized reactive scalar equation
      void solvePred(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrixPred(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for ReactiveScalar[index]
      void sched_reactscalarLinearSolvePred(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of linearized reactive scalar equation
      void solveCorr(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrixCorr(SchedulerP&, const PatchSet* patches,
				       const MaterialSet* matls,
				       int index);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for ReactiveScalar[index]
      void sched_reactscalarLinearSolveCorr(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of linearized reactive scalar equation
      void solveInterm(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrixInterm(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for ReactiveScalar[index]
      void sched_reactscalarLinearSolveInterm(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   int index);
protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default : Construct an empty instance of the Pressure solver.
      ReactiveScalarSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here

      void buildLinearMatrixPred(const ProcessorGroup* pc,
				 const PatchSubset* patches,
				 const MaterialSubset* /*matls*/,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 int index);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solve the Linear System for ReactiveScalar[index]
      //    [in] 
      //        add documentation here
      void reactscalarLinearSolvePred(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      int index);

      ///////////////////////////////////////////////////////////////////////
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here

      void buildLinearMatrixCorr(const ProcessorGroup* pc,
				 const PatchSubset* patches,
				 const MaterialSubset* /*matls*/,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 int index);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solve the Linear System for ReactiveScalar[index]
      //    [in] 
      //        add documentation here
      void reactscalarLinearSolveCorr(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      int index);

      ///////////////////////////////////////////////////////////////////////
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here

      void buildLinearMatrixInterm(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     int index);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solve the Linear System for ReactiveScalar[index]
      //    [in] 
      //        add documentation here
      void reactscalarLinearSolveInterm(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      int index);

private:

      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      ArchesVariables* d_reactscalarVars;
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
#ifdef multimaterialform
      // set the values in problem setup
      MultiMaterialInterface* d_mmInterface;
      MultiMaterialSGSModel* d_mmSGSModel;
#endif


}; // End class ReactiveScalarSolver
} // End namespace Uintah


#endif

