//----- EnthalpySolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_EnthalpySolver_h
#define Uintah_Component_Arches_EnthalpySolver_h

/**************************************
CLASS
   EnthalpySolver
   
   Class EnthalpySolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   EnthalpySolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class EnthalpySolver linearizes and solves scalar
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

class EnthalpySolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of the Enthalpy solver.
      // PRECONDITIONS
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      EnthalpySolver(const ArchesLabel* label, const MPMArchesLabel* MAlb, 
		   TurbulenceModel* turb_model, 
		   BoundaryCondition* bndry_cond,
		   PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~EnthalpySolver();

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
		 double time, double delta_t);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrix(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   double delta_t);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for Enthalpy[index]
      void sched_enthalpyLinearSolve(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   double delta_t);


      void solvePred(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 double time, double delta_t);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrixPred(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   double delta_t);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for Enthalpy[index]
      void sched_enthalpyLinearSolvePred(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   double delta_t);
      void solveCorr(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 double time, double delta_t);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrixCorr(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   double delta_t);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for Enthalpy[index]
      void sched_enthalpyLinearSolveCorr(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   double delta_t);

protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default : Construct an empty instance of the Pressure solver.
      EnthalpySolver();

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
			     double delta_t);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solver the Linear System for Enthalpy[index]
      //    [in] 
      //        add documentation here
      void enthalpyLinearSolve(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     double delta_t);

      void buildLinearMatrixPred(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     double delta_t);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solver the Linear System for Enthalpy[index]
      //    [in] 
      //        add documentation here
      void enthalpyLinearSolvePred(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     double delta_t);

      void buildLinearMatrixCorr(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     double delta_t);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solver the Linear System for Enthalpy[index]
      //    [in] 
      //        add documentation here
      void enthalpyLinearSolveCorr(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     double delta_t);

private:
      ArchesVariables* d_enthalpyVars;
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

      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      bool d_radiationCalc;

}; // End class EnthalpySolver
} // End namespace Uintah


#endif

