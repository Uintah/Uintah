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

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
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
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
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
  class RHSSolver;
  class TimeIntegratorLabel;

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
      void solve(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 const TimeIntegratorLabel* timelabels);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrix(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for ReactiveScalar
      void sched_reactscalarLinearSolve(SchedulerP&, const PatchSet* patches,
				        const MaterialSet* matls,
				        const TimeIntegratorLabel* timelabels);
      
      inline void setMMS(bool doMMS) {
        d_doMMS=doMMS;
      }

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

      void buildLinearMatrix(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solve the Linear System for ReactiveScalar
      //    [in] 
      //        add documentation here
      void reactscalarLinearSolve(const ProcessorGroup* pc,
				  const PatchSubset* patches,
				  const MaterialSubset* /*matls*/,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw,
				  const TimeIntegratorLabel* timelabels);


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
      RHSSolver* d_rhsSolver;
      // turbulence model
      TurbulenceModel* d_turbModel;
      // boundary condition
      BoundaryCondition* d_boundaryCondition;
      // physical constants
      PhysicalConstants* d_physicalConsts;
      int d_conv_scheme;
      int d_limiter_type;
      int d_boundary_limiter_type;
      bool d_central_limiter;
      bool d_dynScalarModel;
      double d_turbPrNo;
      bool d_doMMS;

#ifdef multimaterialform
      // set the values in problem setup
      MultiMaterialInterface* d_mmInterface;
      MultiMaterialSGSModel* d_mmSGSModel;
#endif


}; // End class ReactiveScalarSolver
} // End namespace Uintah


#endif

