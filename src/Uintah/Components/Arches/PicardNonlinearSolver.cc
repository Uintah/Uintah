#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#include <SCICore/Util/NotFinished.h>
#include <math.h>

PicardNonlinearSolver::PicardNonlinearSolver()
{
}

PicardNonlinearSolver::~PicardNonlinearSolver()
{
}

void PicardNonlinearSolver::problemSetup(DatabaseP& db)
{
  if (db->keyExists("max_iter")) {
    d_nonlinear_its = db->getInt("max_iter");
  } else {
    cerr << "max_iter not in input database" << endl;
  }
  if (db->keyExists("res_tol")) {
    d_residual = db->getDouble("nonlinear_solver");
  } else {
    cerr << "res_tol not in input database" << endl;
  }
  bool calPress;
  if (db->keyExists("cal_pressure")) {
    calPress = db->getBool("cal_pressure");
  } else {
    cerr << "cal_pressure not in input database" << endl;
  } 
  if (calPress) {
    d_pressSolver = new PressureSolver();
  }
  if (db->keyExists("Pressure Solver")) {
    DatabaseP& PressureDB = db->getDatabase("Pressure Solver");
  } else {
    cerr << "Pressure Solver DB not in input database" << endl;
  }
  d_pressSolver->problemSetup(PressureDB);
  bool calMom;
  if (db->keyExists("cal_momentum")) {
    calMom = db->getBool("cal_momentum");
  } else {
    cerr << "cal_momentum not in input database" << endl;
  } 
  if (calMom) {
    d_momSolver = new MomentumSolver();
  }
  if (db->keyExists("Momentum Solver")) {
    DatabaseP& MomentumDB = db->getDatabase("Momentum Solver");
  } else {
    cerr << "Momentum Solver DB not in input database" << endl;
  }
  d_momSolver->problemSetup(MomentumDB);
  bool calScalar;
  if (db->keyExists("cal_scalar")) {
    calScalar = db->getBool("cal_scalar");
  } else {
    cerr << "cal_scalar not in input database" << endl;
  } 
  if (calScalar) {
    d_scalarSolver = new ScalarSolver();
  }
  if (db->keyExists("Scalar Solver")) {
    DatabaseP& scalarDB = db->getDatabase("Scalar Solver");
  } else {
    cerr << "Scalar Solver DB not in input database" << endl;
  }
  d_scalarSolver->problemSetup(scalarDB);
  int turbModel;
  if (db->keyExists("turbulence_model")) {
    turbModel = db->getInt("turbulence_model");
  } else {
    cerr << "turbulence_model type not in db" << endl;
  }
  if (turbModel == 0) {
    d_turbModel = new SmagorinskyModel();
  } else {
    cerr << "wrong turbulence option" << endl;
  }
  if (db->keyExists("Turbulence Model")) {
    DatabaseP& turbDB = db->getDatabase("Turbulence Model");
  } else {
    cerr << "Turbulence Model DB not in database" << endl;
  }
  d_turbModel->problemSetup(turbDB);

  d_props = new Properties();
  if (db->keyExists("Properties")) {
    DatabaseP& propsDB = db->getDatabase("Properties");
  } else {
    cerr << "Properties DB not in the database" << endl;
  }
  d_props->problemSetup(propsDB);
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
    int index = 1;
    d_momSolver->solve(index, level, sched, new_dw, nonlinear_dw);
    ++index;
    d_momSolver->solve(index, level, sched, new_dw, nonlinear_dw);
    ++index;
    d_momSolver->solve(index, level, sched, new_dw, nonlinear_dw);
    
    // equation for scalars
    index = 1;
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
  int index = 0;
  residual = d_momSolver->getResidual(index);
  omg = d_momSolver->getOrderMagnitude(index);
  nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  ++index;
  residual = d_momSolver->getResidual(index);
  omg = d_momSolver->getOrderMagnitude(index);
  nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  ++index;
  residual = d_momSolver->getResidual(index);
  omg = d_momSolver->getOrderMagnitude(index);
  nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  //for multiple scalars iterate
  int index = 0;
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


