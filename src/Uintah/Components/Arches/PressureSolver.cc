#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using Uintah::Components::PressureSolver;
using namespace std;

PressureSolver::PressureSolver()
{
}

PressureSolver::PressureSolver(TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst)
: d_turbModel(turb_model), d_boundaryCondition(bndry_cond),
  d_physicalConsts(physConst)
{
}

PressureSolver::~PressureSolver()
{
}

void PressureSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Pressure Solver");
  db->require("pressureReference", d_pressRef);
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "Secondorder") 
    d_discretize = new Discretization();
  else 
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff, db);
  // make source and boundary_condition objects
  d_source = Source(d_turbModel);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "RBGaussSeidel")
    d_linearSolver = new RBGSSolver();
  else 
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol, db);
  d_linearSolver->problemSetup(db);
}

void PressureSolver::solve(double time, double delta_t,
			   const LevelP& level,
			   SchedulerP& sched,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw)
{
  //copy pressure, velocities, scalar, density and viscosity from
  // old_dw to new_dw
  sched_begin(level, sched, old_dw, new_dw);
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  DataWarehouseP matrix_dw = sched->createDataWarehouse();
  //computes stencil coefficients and source terms
  buildLinearMatrix(time, delta_t, level, sched, new_dw, matrix_dw);
  //residual at the start of linear solve
  // this can be part of linear solver
#if 0
  calculateResidual(level, sched, new_dw, matrix_dw);
  calculateOrderMagnitude(level, sched, new_dw, matrix_dw);
#endif
  d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);
  // if linearSolver succesful then copy pressure from new_dw
  // to old_dw
  sched_update(level, sched, old_dw, new_dw, matrix_dw);
  
}

void PressureSolver::buildLinearMatrix(double time, double delta_t,
				       const LevelP& level,
				       SchedulerP& sched,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
  // compute all three componenets of velocity stencil coefficients
  for(int index = 1; index <= NDIM; ++index) {
    d_discretize->sched_calculateVelocityCoeff(index, level, sched, 
					       old_dw,new_dw);
    d_source->sched_calculateVelocitySource(index, level, sched, 
					    old_dw, new_dw);
    d_boundaryCondition->sched_velocityBC(index, level, sched,
					  old_dw, new_dw);
    // similar to mascal
    d_source->sched_modifyMassSource(index, level, sched,
				     old_dw, new_dw);
    d_discretize->sched_calculateDiagonal(index, level, sched,
					  old_dw, new_dw);
  }
  d_discretize->sched_calculatePressureCoeff(level, sched,
					     old_dw, new_dw);
  d_source->sched_calculatePressureSource(level, sched,
					  old_dw, new_dw);
  d_boundaryCondition->sched_pressureBC(level, sched,
					old_dw, new_dw);
  sched_modifyCoeff(level, sched, old_dw, new_dw);

}



