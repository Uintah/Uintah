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

#ifndef Uintah_Component_Arches_PicardNonlinearSolver_h
#define Uintah_Component_Arches_PicardNonlinearSolver_h

#include "Arches.h"
#include "NonlinearSolver.h"


namespace Uintah {
    namespace Components {

#ifndef LACKS_NAMESPACE
using namespace Uintah::Components;
#endif

class PressureSolver;
class MomentumSolver;
class ScalarSolver;
class TurbulenceModel;
class Properties;
class BoundaryCondition;
class PhysicalConstants;
const double MACHINEPRECISSION = 14.0; //used to compute residual
class PicardNonlinearSolver:
public NonlinearSolver
{
public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  PicardNonlinearSolver(Properties* props, BoundaryCondition* bc,
			TurbulenceModel* turbModel, 
			PhysicalConstants* physConst);



  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for PicardNonlinearSolver.
  virtual ~PicardNonlinearSolver();


  ////////////////////////////////////////////////////////////////////////
  // Solve the nonlinear system.
  // The code returns 0 if there are no errors and
  // 1 if there is a nonlinear failure.
  //    [in] data User data needed for solve 
  virtual int nonlinearSolve(double time, double deltat, const LevelP&,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);
  
  virtual void problemSetup(const ProblemSpecP& input_db);
  void computeResidual(const LevelP&, SchedulerP& sched,
		       const DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw);
  void sched_initialize(const LevelP&, SchedulerP& sched,
			    const DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);
  

private:
  void initialize(const ProcessorContext* pc,
		  const Region* region,
		  const DataWarehouseP& old_dw,
		  DataWarehouseP& new_dw);
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
  
};

    }
}

#endif

