#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/Properties.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/PressureSolver.h>
#if 0
  #include <Uintah/Components/Arches/MomentumSolver.h>
  #include <Uintah/Components/Arches/ScalarSolver.h>
#endif
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <SCICore/Util/NotFinished.h>
#include <math.h>

using namespace Uintah::ArchesSpace;

PicardNonlinearSolver::PicardNonlinearSolver(Properties* props, 
					     BoundaryCondition* bc,
					     TurbulenceModel* turbModel,
					     PhysicalConstants* physConst):
  d_props(props), d_boundaryCondition(bc), d_turbModel(turbModel),
  d_physicalConsts(physConst)
{
}

PicardNonlinearSolver::~PicardNonlinearSolver()
{
}

void PicardNonlinearSolver::problemSetup(const ProblemSpecP& params)
{
#if 0
  ProblemSpecP db = params->findBlock("PicardSolver");
  db->require("max_iter", d_nonlinear_its);
  dw->put(nonlinear_its, "max_nonlinear_its");
  db->require("res_tol", d_resTol);
  bool calPress;
  db->require("cal_pressure", calPress);
  if (calPress) {
    d_pressSolver = scinew PressureSolver(d_turbModel, d_boundaryCondition,
				       d_physicalConsts);
    d_pressSolver->problemSetup(db);
  }
  bool calMom;
  db->require("cal_momentum", calMom);
  if (calMom) {
#if 0
    d_momSolver = scinew MomentumSolver(d_turbModel, d_boundaryCondition,
				     d_physicalConsts);
    d_momSolver->problemSetup(db);
#endif
  }
  bool calScalar;
  db->require("cal_scalar", calScalar);
  if (calScalar) {
#if 0
    d_scalarSolver = scinew ScalarSolver(d_turbModel, d_boundaryCondition,
				      d_physicalConsts);
    d_scalarSolver->problemSetup(db);
#endif
  }
  
#endif
}

int PicardNonlinearSolver::nonlinearSolve(double time, double delta_t,
					  const LevelP& level,
					  SchedulerP& sched,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
#if 0
  int nlIterations = 0;
 //initializes and allocates vars for new_dw
  sched_initialize(level, sched, old_dw, new_dw);
  double nlResidual;
  do{
    //create a new data warehouse to store new values during the
    // non-linear iteration, if the iteration is succesful then
    // it copies the dw to new_dw
    DataWarehouseP nonlinear_dw = sched->createDataWarehouse();

    //correct inlet velocities to account for change in properties
    d_boundaryCondition->sched_setInletVelocityBC(level, sched, new_dw, 
						  new_dw);
    // linearizes and solves pressure eqn
    d_pressSolver->solve(time, delta_t, level, sched, new_dw, new_dw);
    // if external boundary then recompute velocities using new pressure
    // and puts them in nonlinear_dw
    d_boundaryCondition->sched_computePressureBC(pc, level, new_dw,
						 new_dw);
    // x-velocity    
    for (int index = 1; index <= Arches::NDIM; ++index) {
      d_momSolver->solve(time, delta_t, index, level, sched, new_dw, new_dw);
    }
    
    // equation for scalars
    for (int index = 1;index <= d_props->getNumMixVars(); index ++) {
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve different scalars
      d_scalarSolver->solve(time, delta_t, index, level, sched, new_dw, new_dw);
    }
    // update properties
    d_props->sched_computeProps(level, sched, new_dw, new_dw);
    // LES Turbulence model to compute turbulent viscosity
    // that accounts for sub-grid scale turbulence
    d_turbModel->sched_computeTurbSubmodel(level, sched, new_dw, new_dw);
    // not sure...but we need to execute tasks somewhere
    ProcessorContext* pc = ProcessorContext::getRootContext();
    scheduler->execute(pc);
     ++nlIterations;
    // residual represents the degrees of inaccuracies
    nlResidual = computeResidual(level, sched, new_dw, new_dw);
  }while((nlIterations < d_nonlinear_its)||(nlResidual > d_resTol));
#endif
  return(0);
}


#if 0
double PicardNonlinearSolver::computeResidual(const LevelP& level,
					    SchedulerP& sched,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
  double nlresidual;
#if 0
  SoleVariable<double> residual;
  SoleVariable<double> omg;
  // not sure of the syntax...this operation is supposed to get 
  // L1norm of the residual over the whole level
  new_dw->get(residual,"pressResidual");
  new_dw->get(omg,"pressomg");
  nlresidual = MACHINEPRECISSION + log(residual/omg);
  for (int index = 1; index <= Arches::NDIM; ++index) {
    new_dw->get(residual,"velocityResidual", index);
    new_dw->get(omg,"velocityomg", index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
  //for multiple scalars iterate
  for (int index = 1;index <= d_props->getNumMixVars(); index ++) {
    new_dw->get(residual,"scalarResidual", index);
    new_dw->get(omg,"scalaromg", index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
#endif
  return nlresidual;
}
#endif  

void PicardNonlinearSolver::sched_initialize(const LevelP& level,
					     SchedulerP& sched,
					     const DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = scinew Task("PicardNonlinearSolver::initialize",patch,
			   old_dw, new_dw, this,
			   PicardNonlinearSolver::initialize);
      tsk->requires(old_dw, "pressure", patch, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "velocity", patch, 0,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "scalar", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", patch, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", patch,0,
		    CCVariable<double>::getTypeDescription());
      // do we need 0 or 1...coz we need to use ghost cell information
      // for computing stencil coefficients
      tsk->computes(new_dw "pressure", patch, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "velocity", patch, 0,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "scalar", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "density", patch, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "viscosity", patch, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
#endif
}



void PicardNonlinearSolver::initialize(const ProcessorContext* pc,
				       const Patch* patch,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
#if 0
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", patch, 0);
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 0);
  CCVariable<Vector> scalar;
  old_dw->get(scalar, "scalar", patch, 0);
  CCVariable<double> density;
  old_dw->get(density, "density", patch, 0);
  CCVariable<double> viscosity;
  old_dw->get(viscosity, "viscosity", patch, 0);
  // Create vars for new_dw
  CCVariable<double> pressure_new;
  new_dw->allocate(pressure_new,"pressure",patch, 1);
  pressure_new = pressure; // copy old into new
  FCVariable<Vector> velocity_new;
  new_dw->allocate(velocity_new,"velocity",patch, 1);
  velocity_new = velocity; // copy old into new
  CCVariable<Vector> scalar_new;
  new_dw->allocate(scalar_new,"scalar",patch, 1);
  scalar_new = scalar; // copy old into new
  CCVariable<double> density_new;
  new_dw->allocate(density_new,"density",patch, 1);
  density_new = density; // copy old into new
  CCVariable<double> viscosity_new;
  new_dw->allocate(viscosity_new,"viscosity",patch, 1);
  viscosity_new = viscosity; // copy old into new
  new_dw->put(pressure_new, "pressure", patch, 0);
  new_dw->put(velocity_new, "velocity", patch, 0);
  new_dw->put(scalar_new, "scalar", patch, 0);
  new_dw->put(density_new, "density", patch, 0);
  new_dw->put(viscosity_new, "viscosity", patch, 0);
#endif
}


