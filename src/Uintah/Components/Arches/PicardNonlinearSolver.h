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
      void sched_initialize(const LevelP&, 
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
      void initialize(const ProcessorContext* pc,
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
      const VarLabel* d_pressureLabel;
      const VarLabel* d_uVelocityLabel;
      const VarLabel* d_vVelocityLabel;
      const VarLabel* d_wVelocityLabel;
      const VarLabel* d_xScalarLabel;
      const VarLabel* d_yScalarLabel;
      const VarLabel* d_zScalarLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;

      // generation variable for DataWarehouse creation
      int d_generation;
  
}; // End class PicardNonlinearSolver

} // End namespace ArchesSpace
} // End namespace Uintah

#endif

//
// $Log$
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

