#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#include <Uintah/Components/Arches/Arches.h>
#include <SCICore/Util/NotFinished.h>
#include <math.h>

PicardNonlinearSolver::PicardNonlinearSolver()
{
}

PicardNonlinearSolver::~PicardNonlinearSolver()
{
}

void PicardNonlinearSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Picard Solver");
  db->require("max_iter", d_nonlinear_its);
  db->require("res_tol", d_residual);
  bool calPress;
  db->require("cal_pressure", calPress);
  if (calPress) {
    d_pressSolver = new PressureSolver();
    d_pressSolver->problemSetup(db);
  }
  bool calMom;
  db->require("cal_momentum", calMom);
  if (calMom) {
    d_momSolver = new MomentumSolver();
    d_momSolver->problemSetup(db);
  }
  bool calScalar;
  db->require("cal_scalar", calScalar);
  if (calScalar) {
    d_scalarSolver = new ScalarSolver();
    d_scalarSolver->problemSetup(db);
  }
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "Smagorinsky") 
    d_turbModel = new SmagorinskyModel();
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel, db);
  
  d_turbModel->problemSetup(db);

  d_props = new Properties();
  d_props->problemSetup(db);
}

int PicardNonlinearSolver::nonlinearSolve(const LevelP& level,
					  SchedulerP& sched,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
  int nlIterations = 0;
 //initializes and allocates vars for new_dw
  initialize(level, sched, old_dw, new_dw);
  double nlResidual;
  do{
    //create a new data warehouse to store new values during the
    // non-linear iteration, if the iteration is succesful then
    // it copies the dw to new_dw
    DataWarehouseP nonlinear_dw = scheduler->createDataWarehouse();

    // linearizes and solves pressure eqn
    d_pressSolver->solve(level, sched, new_dw, nonlinear_dw);
    // if external boundary then recompute velocities using new pressure
    // and puts them in nonlinear_dw
    d_boundaryCondition->computePressureBC(pc, level, new_dw,
					   nonlinear_dw);
    // x-velocity    
    for (int index = 1; index <= Arches::NDIM; ++index) {
      d_momSolver->solve(index, level, sched, new_dw, nonlinear_dw);
    }
    
    // equation for scalars
    int index = 1;
    // in this case we're only solving for one scalar...but
    // the same subroutine can be used to solve different scalars
    d_scalarSolver->solve(index, level, sched, new_dw, nonlinear_dw);
    // update properties
    d_properties->computeProperties(level, sched, new_dw, nonlinear_dw);
    // LES Turbulence model to compute turbulent viscosity
    // that accounts for sub-grid scale turbulence
    d_turbModel->computeTurbulenceSubmodel(level, sched, new_dw, nonlinear_dw);
    //correct inlet velocities to account for change in properties
    d_boundaryCondition->computeInletVelocityBC(level, sched, new_dw, 
						nonlinear_dw);
    // residual represents the degrees of inaccuracies
    nlResidual = computeResidual(level, sched, new_dw, nonlinear_dw);
    ++nlIterations;
    new_dw = nonlinear_dw;
  }while((nlIterations < d_nonlinear_its)||(nlResidual > d_residual));
       
  return(0);
}

double PicardNonlinearSolver::computeResidual(const LevelP& level,
					    SchedulerP& sched,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
  double nlresidual;
  double residual;
  double omg;
  
  residual = d_pressSolver->getResidual();
  omg = d_pressSolver->getOrderMagnitude();
  nlresidual = MACHINEPRECISSION + log(residual/omg);
  for (int index = 1; index <= Arches::NDIM; ++index) {
    residual = d_momSolver->getResidual(index);
    omg = d_momSolver->getOrderMagnitude(index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
  //for multiple scalars iterate
  int index = 1;
  residual = d_scalarSolver->getResidual(index);
  omg = d_scalarSolver->getOrderMagnitude(index);
  nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  return nlresidual;
}
  
  

void PicardNonlinearSolver::scheduler_initialize(const LevelP& level,
						 SchedulerP& sched,
						 const DataWarehouseP& old_dw,
						 DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = new Task("PicardNonlinearSolver::initialize",region,
			   old_dw, new_dw, this,
			   PicardNonlinearSolver::initialize);
      tsk->requires(old_dw, "pressure", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "scalar", region, 1,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      // do we need 0 or 1...coz we need to use ghost cell information
      // for computing stencil coefficients
      tsk->computes(new_dw "pressure", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "scalar", region, 1,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}



void PicardNonlinearSolver::initialize(const ProcessorContext* pc,
				       const Region* region,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 1);
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<Vector> scalar;
  old_dw->get(scalar, "scalar", region, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  CCVariable<double> viscosity;
  old_dw->get(viscosity, "viscosity", region, 1);
  // Create vars for new_dw
  CCVariable<double> pressure_new;
  new_dw->allocate(pressure_new,"pressure",region, 1);
  pressure_new = pressure; // copy old into new
  FCVariable<Vector> velocity_new;
  new_dw->allocate(velocity_new,"velocity",region, 1);
  velocity_new = velocity; // copy old into new
  CCVariable<Vector> scalar_new;
  new_dw->allocate(scalar_new,"scalar",region, 1);
  scalar_new = scalar; // copy old into new
  CCVariable<double> density_new;
  new_dw->allocate(density_new,"density",region, 1);
  density_new = density; // copy old into new
  CCVariable<double> viscosity_new;
  new_dw->allocate(viscosity_new,"viscosity",region, 1);
  viscosity_new = viscosity; // copy old into new
  new_dw->put(pressure_new, "pressure", region, 1);
  new_dw->put(velocity_new, "velocity", region, 1);
  new_dw->put(scalar_new, "scalar", region, 1);
  new_dw->put(density_new, "density", region, 1);
  new_dw->put(viscosity_new, "viscosity", region, 1);
}


