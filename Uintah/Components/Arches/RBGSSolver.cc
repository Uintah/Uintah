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

using Uintah::Components::RBGSSolver;
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
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("RBGSSolver::press_residual",
			   region, old_dw, new_dw, this,
			   RBGSSolver::press_residual);
      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(new_dw, "pressureCoeff", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "pressure", region, 0,
		    CCVariable<double>::getTypeDescription());
      // computes global residual
      tsk->computes(new_dw, "pressureResidual", region, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
    {
      Task* tsk = new Task("RBGSSolver::press_underrelax",
			   region, old_dw, new_dw, this,
			   RBGSSolver::press_underrelax);
      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(old_dw, "pressure", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "pressureCoeff", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(new_dw, "pressureNonlinearSource", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureCoeff", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureNonlinearSource", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }
    {
      // use a recursive task based on number of sweeps reqd
      Task* tsk = new Task("RBGSSolver::press_lisolve",
			   region, old_dw, new_dw, this,
			   RBGSSolver::press_lisolve);
      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(new_dw, "pressureCoeff", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(new_dw, "pressureNonlinearSource", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "pressure", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "pressure", region, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
    // add another task taht computes the liner residual
    
}

void RBGSSolver::press_underrelax(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 0);
  CCVariable<Vector> pressCoeff;
  new_dw->get(pressCoeff,"pressureCoeff",region, 0);
  CCVariable<double> pressNlSrc;
  new_dw->get(pressNlSrc,"pressNonlinearSrc",region, 0);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  //fortran call
  FORT_UNDERELAX(pressCoeff, pressNonlinearSrc, pressure,
		 lowIndex, highIndex, d_underrelax);
  new_dw->put(pressCoeff, "pressureCoeff", region, 0);
  new_dw->put(pressNlSrc, "pressureNonlinearSource", region, 0);
}

void RBGSSolver::press_lisolve(const ProcessorContext*,
			       const Region* region,
			       const DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 0);
  CCVariable<Vector> pressCoeff;
  new_dw->get(pressCoeff,"pressureCoeff",region, 0);
  CCVariable<double> pressNlSrc;
  new_dw->get(pressNlSrc,"pressNonlinearSrc",region, 0);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(pressCoeff, pressNonlinearSrc, pressure,
	      lowIndex, highIndex);
  new_dw->put(pressure, "pressure", region, 0);
}

