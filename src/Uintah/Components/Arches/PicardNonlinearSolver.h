//----- PicardNonlinearSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_PicardNonlinearSolver_h
#define Uintah_Component_Arches_PicardNonlinearSolver_h

/**************************************
CLASS
   NonlinearSolver
   
   Class PicardNonlinearSolver is a subclass of NonlinearSolver
   which implements the Fixed Point Picard iteration.[Ref Kumar's thesis]

GENERAL INFORMATION
   PicardNonlinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000


KEYWORDS


DESCRIPTION
   Class PicardNonlinearSolver implements the
   Fixed Point Picard iteration method is used by
   ImplicitIntegrator to solve set of nonlinear equations

WARNING
   none
****************************************/

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/NonlinearSolver.h>

namespace Uintah {
namespace ArchesSpace {

class PressureSolver;
class MomentumSolver;
class ScalarSolver;
class TurbulenceModel;
class Properties;
class BoundaryCondition;
class PhysicalConstants;
const double MACHINEPRECISSION = 14.0; //used to compute residual

class PicardNonlinearSolver: public NonlinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Solver initialized with all input data 
      //
      PicardNonlinearSolver(Properties* props, 
			    BoundaryCondition* bc,
			    TurbulenceModel* turbModel, 
			    PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for PicardNonlinearSolver.
      //
      virtual ~PicardNonlinearSolver();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& input_db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      //
      // Solve the nonlinear system. (also does some actual computations)
      // The code returns 0 if there are no errors and
      // 1 if there is a nonlinear failure.
      //    [in] 
      //        documentation here
      //    [out] 
      //        documentation here
      //
      virtual int nonlinearSolve(const LevelP&,
				 SchedulerP& sched,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw,
				 double time, double deltat);
  
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the Initialization of non linear solver
      //    [in] 
      //        data User data needed for solve 
      //
      void sched_setInitialGuess(const LevelP&, 
				 SchedulerP& sched,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);

      // GROUP: Action Computations :
      ///////////////////////////////////////////////////////////////////////
      //
      // Compute the residual
      //    [in] 
      //        documentation here
      //
      double computeResidual(const LevelP&, 
			     SchedulerP& sched,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);
  
protected :

private:

      // GROUP: Constructors (private):
      ////////////////////////////////////////////////////////////////////////
      //
      // Should never be used
      //
      PicardNonlinearSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      //
      // Actually Initialize the non linear solver
      //    [in] 
      //        data User data needed for solve 
      //
      void setInitialGuess(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);

private:

      // Total number of nonlinear iterates
      int d_nonlinear_its;
      // nonlinear residual tolerance
      double d_resTol;
      // Pressure Eqn Solver
      PressureSolver* d_pressSolver;
      // Momentum Eqn Solver 
      MomentumSolver* d_momSolver;
      // Scalar solver
      ScalarSolver* d_scalarSolver;
      // physcial constatns
      PhysicalConstants* d_physicalConsts;
      // properties...solves density, temperature and specie concentrations
      Properties* d_props;
      // Turbulence Model
      TurbulenceModel* d_turbModel;
      // Boundary conditions
      BoundaryCondition* d_boundaryCondition;

      // const VarLabel*
      const VarLabel* d_pressureSPBCLabel;
      const VarLabel* d_uVelocitySPBCLabel;
      const VarLabel* d_vVelocitySPBCLabel;
      const VarLabel* d_wVelocitySPBCLabel;
      const VarLabel* d_pressureINLabel;
      const VarLabel* d_uVelocitySPLabel;
      const VarLabel* d_vVelocitySPLabel;
      const VarLabel* d_wVelocitySPLabel;
      const VarLabel* d_scalarSPLabel;
      const VarLabel* d_densityCPLabel;
      const VarLabel* d_viscosityCTSLabel;

      // generation variable for DataWarehouse creation
      int d_generation;
  
}; // End class PicardNonlinearSolver

} // End namespace ArchesSpace
} // End namespace Uintah

#endif

//
// $Log$
// Revision 1.17  2000/07/11 15:46:27  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.16  2000/06/21 07:51:00  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.15  2000/06/18 01:20:15  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.14  2000/06/17 07:06:25  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.13  2000/06/13 06:02:31  bbanerje
// Added some more StencilMatrices and vector<CCVariable> types.
//
// Revision 1.12  2000/06/07 06:13:55  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.11  2000/06/04 23:57:46  bbanerje
// Updated Arches to do ScheduleTimeAdvance.
//
// Revision 1.10  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

