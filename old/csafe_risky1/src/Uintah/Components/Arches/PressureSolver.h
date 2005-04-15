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

#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Components/Arches/ArchesLabel.h>
namespace Uintah {
class ProcessorGroup;
namespace ArchesSpace {
class ArchesVariables;
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
      PressureSolver(const ArchesLabel* label,
		     TurbulenceModel* turb_model, 
		     BoundaryCondition* bndry_cond,
		     PhysicalConstants* physConst,
		     const ProcessorGroup* myworld);

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
 
      void sched_pressureLinearSolve(const LevelP& level, 
				     SchedulerP& sched, 
				     DataWarehouseP& new_dw,
				     DataWarehouseP& matrix_dw);

protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      //
      // Default : Construct an empty instance of the Pressure solver.
      //
      PressureSolver(const ProcessorGroup* myworld);

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here
      //
      void buildLinearMatrix(const ProcessorGroup* pc,
			     const Patch* patch,
			     DataWarehouseP& new_dw,
			     DataWarehouseP& matrix_dw,
			     double delta_t);
      void buildLinearMatrixPress(const ProcessorGroup* pc,
			     const Patch* patch,
			     DataWarehouseP& new_dw,
			     DataWarehouseP& matrix_dw,
			     double delta_t);

      void pressureLinearSolve_all(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& new_dw,
				   DataWarehouseP& matrix_dw,
				   LevelP level, SchedulerP sched);
      void pressureLinearSolve(const ProcessorGroup* pc,
			       const Patch* patch,
			       DataWarehouseP& new_dw,
			       DataWarehouseP& matrix_dw,
			       ArchesVariables& pressureVars);
      
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually do normPressure
      //    [in] 
      //        add documentation here
      //
      void normPressure(const ProcessorGroup* pc,
			const Patch* patch,
			ArchesVariables* vars);

  

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

      // Maximum number of iterations to take before stopping/giving up.
      int d_maxIterations;
      // underrealaxation parameter, read from an input database
      double d_underrelax;
      //reference points for the solvers
      IntVector d_pressRef;

      // const VarLabel* (required)
      const ArchesLabel* d_lab;

   const ProcessorGroup* d_myworld;
}; // End class PressureSolver

}  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.28.4.1  2000/10/19 05:17:29  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.30  2000/10/07 05:38:51  sparker
// Changed d_pressureVars into a few different local variables so that
// they will be freed at the end of the task.
//
// Revision 1.29  2000/10/04 16:46:24  rawat
// Parallel solver for pressure is working
//
// Revision 1.28  2000/09/20 18:05:34  sparker
// Adding support for Petsc and per-processor tasks
//
// Revision 1.27  2000/08/11 21:26:36  rawat
// added linear solver for pressure eqn
//
// Revision 1.26  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.24  2000/07/13 06:32:10  bbanerje
// Labels are once more consistent for one iteration.
//
// Revision 1.23  2000/07/12 23:59:21  rawat
// added wall bc for u-velocity
//
// Revision 1.22  2000/07/11 15:46:28  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.21  2000/07/07 23:07:45  rawat
// added inlet bc's
//
// Revision 1.20  2000/07/03 05:30:15  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.19  2000/06/22 23:06:36  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.18  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.17  2000/06/17 07:06:25  sparker
// Changed ProcessorContext to ProcessorGroup
//
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
