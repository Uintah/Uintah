#include <Uintah/Components/Arches/PressureSolver.h>
#include <SCICore/Util/NotFinished.h>

PressureSolver::PressureSolver()
{
}

PressureSolver::~PressureSolver()
{
}

void PressureSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Pressure Solver");
  db->require("ipref", d_ipref);
  db->require("jpref", d_jpref);
  db->require("kpref", d_kpref);
  db->require("underrelax", d_underrelax);
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "Secondorder") 
    d_discretize = new Discretization();
  else 
    throw InvalidValue("Finite Differencing scheme 
                        not supported" + finite_diff, db);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "GaussSiedel")
    d_linearSolver = new LineGS();
  else 
    throw InvalidValue("linear solver option
                        not supported" + linear_sol, db);
  d_linearSolver->problemSetup(db);
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
  DataWarehouseP matrix_dw = scheduler->createDataWarehouse();
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
    d_boundaryCondition->sched_VelocityBC(index, level, sched,
					  old_dw, new_dw);
  }
  d_discretize->sched_calculatePressureCoeff(level, sched,
					     old_dw, new_dw);
  d_source->sched_calculatePressureSource(level, sched
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

  //pass a string to identify the eqn for which residual 
  // needs to be solved, in this case pressure string
  d_linearSolver->sched_computeResidual(level, sched,
					old_dw, new_dw);
  // reduces from all processors to compute L1 norm
  new_dw->get(d_residual, "PressResidual"); 


}

void  PressureSolver::calculateOrderMagnitude(const LevelP& level,
					      SchedulerP& sched,
					      const DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{

  //pass a string to identify the eqn for which residual 
  // needs to be solved, in this case pressure string
  d_linearSolver->sched_calculateOrderMagnitude(level, sched,
						old_dw, new_dw);
  // reduces from all processors to compute L1 norm
  new_dw->get(d_ordermagnitude, "PressOMG"); 


}


