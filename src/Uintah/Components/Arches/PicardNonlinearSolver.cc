//----- PicardNonlinearSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/Properties.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/MomentumSolver.h>
#include <Uintah/Components/Arches/ScalarSolver.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <SCICore/Util/NotFinished.h>
#include <math.h>

using namespace Uintah::ArchesSpace;

//****************************************************************************
// Private constructor for PicardNonlinearSolver
//****************************************************************************
PicardNonlinearSolver::PicardNonlinearSolver():NonlinearSolver()
{
}

//****************************************************************************
// Default constructor for PicardNonlinearSolver
//****************************************************************************
PicardNonlinearSolver::PicardNonlinearSolver(Properties* props, 
					     BoundaryCondition* bc,
					     TurbulenceModel* turbModel,
					     PhysicalConstants* physConst):
                                                NonlinearSolver(),
                                                d_props(props), 
                                                d_boundaryCondition(bc), 
                                                d_turbModel(turbModel),
                                                d_physicalConsts(physConst),
						d_generation(0)
{
  d_pressureLabel = scinew VarLabel("pressure",
				   CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velocityLabel = scinew VarLabel("velocity",
				    CCVariable<Vector>::getTypeDescription() );
  d_scalarLabel = scinew VarLabel("scalar",
				    CCVariable<Vector>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
				   CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
PicardNonlinearSolver::~PicardNonlinearSolver()
{
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
PicardNonlinearSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PicardSolver");
  db->require("max_iter", d_nonlinear_its);
  
  // ** WARNING ** temporarily commented out
  // dw->put(nonlinear_its, "max_nonlinear_its");

  db->require("res_tol", d_resTol);
  int nDim = 3;
  //db->require("num_dim", nDim);
  bool calPress;
  db->require("cal_pressure", calPress);
  if (calPress) {
    d_pressSolver = scinew PressureSolver(nDim,
					  d_turbModel, d_boundaryCondition,
					  d_physicalConsts);
    d_pressSolver->problemSetup(db);
  }
  bool calMom;
  db->require("cal_momentum", calMom);
  if (calMom) {
    d_momSolver = scinew MomentumSolver(d_turbModel, d_boundaryCondition,
				     d_physicalConsts);
    d_momSolver->problemSetup(db);
  }
  bool calScalar;
  db->require("cal_mixturescalar", calScalar);
  if (calScalar) {
    d_scalarSolver = scinew ScalarSolver(d_turbModel, d_boundaryCondition,
				      d_physicalConsts);
    d_scalarSolver->problemSetup(db);
  }
}

//****************************************************************************
// Schedule non linear solve and carry out some actual operations
//****************************************************************************
int PicardNonlinearSolver::nonlinearSolve(double time, double delta_t,
					  const LevelP& level,
					  SchedulerP& sched,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
  //initializes and allocates vars for new_dw
  sched_initialize(level, sched, old_dw, new_dw);

#ifdef THIS_SHOULD_GO_INTO_SIMULATION_CONTROLLER
  int nlIterations = 0;
  double nlResidual = 2.0*d_resTol;;
  do{
    //create a new data warehouse to store new values during the
    // non-linear iteration, if the iteration is succesful then
    // it copies the dw to new_dw
    DataWarehouseP nonlinear_dw = sched->createDataWarehouse(d_generation);
    ++d_generation;

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

//****************************************************************************
// Schedule initialize 
//****************************************************************************
void 
PicardNonlinearSolver::sched_initialize(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = scinew Task("PicardNonlinearSolver::initialize",patch,
			   old_dw, new_dw, this,
			   &PicardNonlinearSolver::initialize);

      // do we need 0 or 1...coz we need to use ghost cell information
      // for computing stencil coefficients
      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_velocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_scalarLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      tsk->computes(new_dw, d_pressureLabel, matlIndex, patch);
      tsk->computes(new_dw, d_velocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_scalarLabel, matlIndex, patch);
      tsk->computes(new_dw, d_velocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_viscosityLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Actual initialize 
//****************************************************************************
void 
PicardNonlinearSolver::initialize(const ProcessorContext* ,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  int matlIndex = 0;
  int nofGhostCells = 0;

  CCVariable<double> pressure;
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // (tmp) velocity should be FCVariable
  CCVariable<Vector> velocity;
  old_dw->get(velocity, d_velocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  CCVariable<Vector> scalar;
  old_dw->get(scalar, d_scalarLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);


  // Create vars for new_dw
  CCVariable<double> pressure_new;
  new_dw->allocate(pressure_new, d_pressureLabel, matlIndex, patch);
  pressure_new = pressure; // copy old into new

  // (tmp) velocity should be FCVariable
  CCVariable<Vector> velocity_new;
  new_dw->allocate(velocity_new, d_velocityLabel, matlIndex, patch);
  velocity_new = velocity; // copy old into new

  CCVariable<Vector> scalar_new;
  new_dw->allocate(scalar_new, d_scalarLabel, matlIndex, patch);
  scalar_new = scalar; // copy old into new

  CCVariable<double> density_new;
  new_dw->allocate(density_new, d_densityLabel, matlIndex, patch);
  density_new = density; // copy old into new

  CCVariable<double> viscosity_new;
  new_dw->allocate(viscosity_new, d_viscosityLabel, matlIndex, patch);
  viscosity_new = viscosity; // copy old into new

  // Copy the variables into the new datawarehouse
  new_dw->put(pressure_new, d_pressureLabel, matlIndex, patch);
  new_dw->put(velocity_new, d_velocityLabel, matlIndex, patch);
  new_dw->put(scalar_new, d_scalarLabel, matlIndex, patch);
  new_dw->put(density_new, d_densityLabel, matlIndex, patch);
  new_dw->put(viscosity_new, d_viscosityLabel, matlIndex, patch);
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


//
// $Log$
// Revision 1.19  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
