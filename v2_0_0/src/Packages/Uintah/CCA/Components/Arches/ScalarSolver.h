//----- ScalarSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ScalarSolver_h
#define Uintah_Component_Arches_ScalarSolver_h

/**************************************
CLASS
   ScalarSolver
   
   Class ScalarSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   ScalarSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ScalarSolver linearizes and solves scalar
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
  class TimeIntegratorLabel;

class ScalarSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of the Scalar solver.
      // PRECONDITIONS
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      ScalarSolver(const ArchesLabel* label, const MPMArchesLabel* MAlb, 
		   TurbulenceModel* turb_model, 
		   BoundaryCondition* bndry_cond,
		   PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~ScalarSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of linearized scalar equation
      void solve(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 const TimeIntegratorLabel* timelabels,
		 int index);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrix(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels,
				   int index);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for Scalar[index]
      void sched_scalarLinearSolve(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels,
				   int index);

protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default : Construct an empty instance of the Pressure solver.
      ScalarSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here
      void buildLinearMatrix(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels,
			     int index);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solver the Linear System for Scalar[index]
      //    [in] 
      //        add documentation here
      void scalarLinearSolve(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels,
			     int index);


private:
      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      ArchesVariables* d_scalarVars;
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
      int d_conv_scheme;

#ifdef multimaterialform
      // set the values in problem setup
      MultiMaterialInterface* d_mmInterface;
      MultiMaterialSGSModel* d_mmSGSModel;
#endif


}; // End class ScalarSolver
} // End namespace Uintah


#endif

