#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
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
			       BoundaryCondition* bndry_cond)
: d_turbModel(turb_model), d_boundaryCondition(bndry_cond)
{
}

PressureSolver::~PressureSolver()
{
}

void PressureSolver::problemSetup(const ProblemSpecP& params,
				  DataWarehouseP& dw)
{
  ProblemSpecP db = params->findBlock("Pressure Solver");
  Array3Index pressRef;
  db->require("pressureReference", pressRef);
  dw->put(pressRef, "pressureReference");
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
  if (linear_sol == "GaussSiedel")
    d_linearSolver = new LineGS();
  else 
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol, db);
  d_linearSolver->problemSetup(db, dw);
}

void PressureSolver::solve(const LevelP& level,
			  SchedulerP& sched,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw)
{
  //copy pressure, velocities, density and viscosity from
  // old_dw to new_dw
  sched_begin(level, sched, old_dw, new_dw);
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  DataWarehouseP matrix_dw = sched->createDataWarehouse();
  //computes stencil coefficients and source terms
  buildLinearMatrix(level, sched, new_dw, matrix_dw);
  //residual at the start of linear solve
  calculateResidual(level, sched, new_dw, matrix_dw);
  calculateOrderMagnitude(level, sched, new_dw, matrix_dw);
  d_linearSolver->sched_solve(level, sched, new_dw, matrix_dw);
  // if linearSolver succesful then copy pressure from new_dw
  // to old_dw
  sched_update(level, sched, old_dw, new_dw, matrix_dw);
  
}

void PressureSolver::buildLinearMatrix(const LevelP& level,
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
  }
  d_discretize->sched_calculatePressureCoeff(level, sched,
					     old_dw, new_dw);
  d_source->sched_calculatePressureSource(level, sched,
					  old_dw, new_dw);
  d_boundaryCondition->sched_pressureBC(level, sched,
					old_dw, new_dw);
  sched_modifyCoeff(level, sched, old_dw, new_dw);

}

void  PressureSolver::calculateResidual(const LevelP& level,
				       SchedulerP& sched,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{

#if 0
  //pass a string to identify the eqn for which residual 
  // needs to be solved, in this case pressure string
  d_linearSolver->sched_computeResidual(level, sched,
					old_dw, new_dw);
  // reduces from all processors to compute L1 norm
  new_dw->put(d_residual, "PressResidual", Reduction::Sum); 
#else
  cerr << "PressureSolver::calculateResidual needs thought\n";
#endif
}

void  PressureSolver::calculateOrderMagnitude(const LevelP& level,
					      SchedulerP& sched,
					      const DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
#if 0
  //pass a string to identify the eqn for which residual 
  // needs to be solved, in this case pressure string
  d_linearSolver->sched_calculateOrderMagnitude(level, sched,
						old_dw, new_dw);
  // reduces from all processors to compute L1 norm
  new_dw->put(d_ordermagnitude, "PressOMG", Reduction::Sum);
#else
  cerr << "PressureSolver::calculateOrderMagnitude needs thought\n";
#endif
}


