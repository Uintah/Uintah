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
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/CCVariable.h>

namespace Uintah {
namespace ArchesSpace {

class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class MomentumSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of the Momentum solver.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      //
      MomentumSolver(TurbulenceModel* turb_model, 
		     BoundaryCondition* bndry_cond,
		     PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~MomentumSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule Solve of the linearized momentum equation. 
      //
      void solve(const LevelP& level,
		 SchedulerP& sched,
		 DataWarehouseP& old_dw,
		 DataWarehouseP& new_dw, 
		 double time, double delta_t, int index);
   
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the build of the linearized momentum matrix
      //
      void sched_buildLinearMatrix(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t, int index);

protected: 

private:

      // GROUP: Constructors (private):
      ////////////////////////////////////////////////////////////////////////
      //
      // Default constructor.
      MomentumSolver();

      // GROUP: Action Methods (private):
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually build the linearized momentum matrix
      //
      void buildLinearMatrix(const ProcessorContext* pc,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw,
			     double delta_t, int index);

private:

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
      const VarLabel* d_pressureLabel;
      const VarLabel* d_velocityLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;

      // const VarLabel* (computed)
      const VarLabel* d_velConvCoefLabel;
      const VarLabel* d_velCoefLabel;
      const VarLabel* d_velLinSrcLabel;
      const VarLabel* d_velNonLinSrcLabel;

      // DataWarehouse generation
      int d_generation;

}; // End class MomentumSolver

}  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.5  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
