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
#include <SCICore/Geometry/Vector.h>

namespace Uintah {
class ProcessorGroup;
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
      void buildLinearMatrix(const ProcessorGroup* pc,
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
      const VarLabel* d_cellTypeLabel;
      const VarLabel* d_pressureINLabel;
      const VarLabel* d_pressureSPBCLabel;
      const VarLabel* d_uVelocitySPBCLabel;
      const VarLabel* d_vVelocitySPBCLabel;
      const VarLabel* d_wVelocitySPBCLabel;
      const VarLabel* d_uVelocitySIVBCLabel;
      const VarLabel* d_vVelocitySIVBCLabel;
      const VarLabel* d_wVelocitySIVBCLabel;
      const VarLabel* d_densityCPLabel;
      const VarLabel* d_viscosityCTSLabel;

      // const VarLabel* (computed)
      const VarLabel* d_uVelConvCoefPBLMLabel;
      const VarLabel* d_vVelConvCoefPBLMLabel;
      const VarLabel* d_wVelConvCoefPBLMLabel;
      const VarLabel* d_uVelCoefPBLMLabel;
      const VarLabel* d_vVelCoefPBLMLabel;
      const VarLabel* d_wVelCoefPBLMLabel;
      const VarLabel* d_uVelLinSrcPBLMLabel;
      const VarLabel* d_vVelLinSrcPBLMLabel;
      const VarLabel* d_wVelLinSrcPBLMLabel;
      const VarLabel* d_uVelNonLinSrcPBLMLabel;
      const VarLabel* d_vVelNonLinSrcPBLMLabel;
      const VarLabel* d_wVelNonLinSrcPBLMLabel;
      const VarLabel* d_presCoefPBLMLabel;
      const VarLabel* d_presLinSrcPBLMLabel;
      const VarLabel* d_presNonLinSrcPBLMLabel;

      // DataWarehouse generation
      int d_generation;

}; // End class PressureSolver

}  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
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
