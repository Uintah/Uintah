//----- PressureSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_PressureSolver_h
#define Uintah_Components_Arches_PressureSolver_h

/**************************************
CLASS
   PressureSolver
   
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


GENERAL INFORMATION
   PressureSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


WARNING
   none

************************************************************************/

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/VarLabel.h>

namespace Uintah {
namespace ArchesSpace {

class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class PressureSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of the Pressure solver.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      //
      PressureSolver(int nDim,
		     TurbulenceModel* turb_model, 
		     BoundaryCondition* bndry_cond,
		     PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~PressureSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule Solve of linearized pressure equation
      //
      void solve(const LevelP& level,
		 SchedulerP& sched,
		 DataWarehouseP& old_dw,
		 DataWarehouseP& new_dw,
		 double time, double delta_t);
   
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the build of the linearized eqn
      //
      void sched_buildLinearMatrix(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t);
 
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the creation of the .. more documentation here
      //
      void sched_normPressure(const LevelP& level,
			      SchedulerP& sched,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw);  

protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      //
      // Default : Construct an empty instance of the Pressure solver.
      //
      PressureSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here
      //
      void buildLinearMatrix(const ProcessorContext* pc,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw,
			     double delta_t);

      ///////////////////////////////////////////////////////////////////////
      //
      // Actually do normPressure
      //    [in] 
      //        add documentation here
      //
      void normPressure(const Patch* patch,
			SchedulerP& sched,
			const DataWarehouseP& old_dw,
			DataWarehouseP& new_dw);

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

      // Whether the analysis is 2d or 3d
      int d_NDIM;
      // Maximum number of iterations to take before stopping/giving up.
      int d_maxIterations;
      // underrealaxation parameter, read from an input database
      double d_underrelax;
      //reference points for the solvers
      Vector d_pressRef;

      // const VarLabel* (required)
      const VarLabel* d_pressureLabel;
      const VarLabel* d_uVelocityLabel;
      const VarLabel* d_vVelocityLabel;
      const VarLabel* d_wVelocityLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;

      // const VarLabel* (computed)
      const VarLabel* d_uVelConvCoefLabel;
      const VarLabel* d_vVelConvCoefLabel;
      const VarLabel* d_wVelConvCoefLabel;
      const VarLabel* d_uVelCoefLabel;
      const VarLabel* d_vVelCoefLabel;
      const VarLabel* d_wVelCoefLabel;
      const VarLabel* d_uVelLinSrcLabel;
      const VarLabel* d_vVelLinSrcLabel;
      const VarLabel* d_wVelLinSrcLabel;
      const VarLabel* d_uVelNonLinSrcLabel;
      const VarLabel* d_vVelNonLinSrcLabel;
      const VarLabel* d_wVelNonLinSrcLabel;
      const VarLabel* d_presCoefLabel;
      const VarLabel* d_presLinSrcLabel;
      const VarLabel* d_presNonLinSrcLabel;

      // DataWarehouse generation
      int d_generation;

}; // End class PressureSolver

}  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.16  2000/06/07 06:13:55  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.15  2000/06/04 23:57:47  bbanerje
// Updated Arches to do ScheduleTimeAdvance.
//
// Revision 1.14  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
