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

using namespace Uintah::Arches;
using namespace std;

RBGSSolver::RBGSSolver()
{
}

RBGSSolver::~RBGSSolver()
{
}

void RBGSSolver::problemSetup(const ProblemSpecP& params,
				  DataWarehouseP& dw)
{
  ProblemSpecP db = params->findBlock("Linear Solver");
  db->require("max_iter", d_maxSweeps);
  db->require("res_tol", d_residual);
  db->require("underrelax", d_underrelax);
}

void RBGSSolver::sched_pressureSolve(const LevelP& level,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("RBGSSolver::press_residual",
			      patch, old_dw, new_dw, this,
			      RBGSSolver::press_residual);
      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(new_dw, "pressureCoeff", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "pressure", patch, 0,
		    CCVariable<double>::getTypeDescription());
      // computes global residual
      tsk->computes(new_dw, "pressureResidual", patch, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
    {
      Task* tsk = scinew Task("RBGSSolver::press_underrelax",
			      patch, old_dw, new_dw, this,
			      RBGSSolver::press_underrelax);
      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(old_dw, "pressure", patch, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "pressureCoeff", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(new_dw, "pressureNonlinearSource", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureCoeff", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureNonlinearSource", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }
    {
      // use a recursive task based on number of sweeps reqd
      Task* tsk = scinew Task("RBGSSolver::press_lisolve",
			      patch, old_dw, new_dw, this,
			      RBGSSolver::press_lisolve);
      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(new_dw, "pressureCoeff", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(new_dw, "pressureNonlinearSource", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "pressure", patch, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "pressure", patch, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
    // add another task taht computes the liner residual
    
}

void RBGSSolver::press_underrelax(const ProcessorContext*,
				  const Patch* patch,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", patch, 0);
  CCVariable<Vector> pressCoeff;
  new_dw->get(pressCoeff,"pressureCoeff",patch, 0);
  CCVariable<double> pressNlSrc;
  new_dw->get(pressNlSrc,"pressNonlinearSrc",patch, 0);
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();

  //fortran call
  FORT_UNDERELAX(pressCoeff, pressNonlinearSrc, pressure,
		 lowIndex, highIndex, d_underrelax);
  new_dw->put(pressCoeff, "pressureCoeff", patch, 0);
  new_dw->put(pressNlSrc, "pressureNonlinearSource", patch, 0);
}

void RBGSSolver::press_lisolve(const ProcessorContext*,
			       const Patch* patch,
			       const DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", patch, 0);
  CCVariable<Vector> pressCoeff;
  new_dw->get(pressCoeff,"pressureCoeff",patch, 0);
  CCVariable<double> pressNlSrc;
  new_dw->get(pressNlSrc,"pressNonlinearSrc",patch, 0);
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();

  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(pressCoeff, pressNonlinearSrc, pressure,
	      lowIndex, highIndex);
  new_dw->put(pressure, "pressure", patch, 0);
}

