#include <Uintah/Components/Arches/PressureSolver.h>
#include <SCICore/Util/NotFinished.h>

PressureSolver::PressureSolver()
{
}

PressureSolver::~PressureSolver()
{
}

void PressureSolver::problemSetup(DatabaseP& db)
{
  if (db->keyExists("ipref")) {
    d_ipref = db->getInt("ipref");
  } else {
    cerr << "ipref not in input database" << endl;
  }
  if (db->keyExists("jpref")) {
    d_jpref = db->getInt("jpref");
  } else {
    cerr << "jpref not in input database" << endl;
  }
  if (db->keyExists("kpref")) {
    d_kpref = db->getInt("kpref");
  } else {
    cerr << "kpref not in input database" << endl;
  }
  if (db->keyExists("underrelax")) {
    d_underrelax = db->getDouble("underrelax");
  } else {
    cerr << "underrelax not in input database" << endl;
  }
  int finite_diff;
  if (db->keyExists("finite_difference")) {
    finite_diff = db->getInt("finite_difference");
  } else {
    cerr << "finite_difference not in input database" << endl;
  } 
  if (finite_diff == 1) {
    d_discretize = new Discretization();
  } else {
    cerr << "invalid option for discretization" << endl;
  }
  int linear_sol;
  if (db->keyExists("linear_solver")) {
    linear_sol = db->getInt("linear_solver");
  } else {
    cerr << "linear_solver not in input database" << endl;
  } 
  if (linear_sol == 1) {
    d_linearSolver = new LineGS();
  } else {
    cerr << "invalid option for linear solver" << endl;
  }
  
  if (db->keyExists("Linear Solver")) {
    DatabaseP& linearSolDB = db->getDatabase("Linear Solver");
  } else {
    cerr << "Linear Solver DB not in input database" << endl;
  }
  d_linearSolver->problemSetup(linearSolDB);
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
  d_discretize->sched_modifyPressureCoeff(level, sched,
					  old_dw, new_dw);

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


