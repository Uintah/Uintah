//----- RBGSSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Default constructor for RBGSSolver
//****************************************************************************
RBGSSolver::RBGSSolver()
{
  d_pressureLabel = scinew VarLabel("pressure",
				    CCVariable<double>::getTypeDescription() );
  d_xPresCoefLabel = scinew VarLabel("xPressureCoeff",
				    CCVariable<double>::getTypeDescription() );
  d_yPresCoefLabel = scinew VarLabel("yPressureCoeff",
				    CCVariable<double>::getTypeDescription() );
  d_zPresCoefLabel = scinew VarLabel("zPressureCoeff",
				    CCVariable<double>::getTypeDescription() );
  d_xPresNonLinSrcLabel = scinew VarLabel("xPressureNonlinearSource",
				    CCVariable<double>::getTypeDescription() );
  d_yPresNonLinSrcLabel = scinew VarLabel("yPressureNonlinearSource",
				    CCVariable<double>::getTypeDescription() );
  d_zPresNonLinSrcLabel = scinew VarLabel("zPressureNonlinearSource",
				    CCVariable<double>::getTypeDescription() );
  d_presResidualLabel = scinew VarLabel("pressureResidual",
				    CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
RBGSSolver::~RBGSSolver()
{
}

//****************************************************************************
// Problem setup
//****************************************************************************
void 
RBGSSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("LinearSolver");
  db->require("max_iter", d_maxSweeps);
  db->require("res_tol", d_residual);
  db->require("underrelax", d_underrelax);
}

//****************************************************************************
// Underrelaxation
//****************************************************************************
void 
RBGSSolver::sched_underrelax(const LevelP& level,
			     SchedulerP& sched,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{
}

//****************************************************************************
// Schedule pressure solve
//****************************************************************************
void 
RBGSSolver::sched_pressureSolve(const LevelP& level,
				SchedulerP& sched,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("RBGSSolver::press_residual",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::press_residCalculation);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_xPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_yPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_zPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // computes global residual
      tsk->computes(new_dw, d_presResidualLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
    {
      Task* tsk = scinew Task("RBGSSolver::press_underrelax",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::press_underrelax);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_xPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_yPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_zPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_xPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_yPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_zPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // computes 
      tsk->computes(new_dw, d_xPresCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_yPresCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_zPresCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_xPresNonLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_yPresNonLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_zPresNonLinSrcLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
    {
      // use a recursive task based on number of sweeps reqd
      Task* tsk = scinew Task("RBGSSolver::press_lisolve",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::press_lisolve);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      // not sure if the var is of type CCVariable or FCVariable
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_xPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_yPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_zPresCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_xPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_yPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_zPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // Computes
      tsk->computes(new_dw, d_pressureLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
    // add another task that computes the linear residual
  }    
}

//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
RBGSSolver::sched_velSolve(const LevelP& level,
			   SchedulerP& sched,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw,
			   int index)
{
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RBGSSolver::sched_scalarSolve(const LevelP& level,
			      SchedulerP& sched,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      int index)
{
}

//****************************************************************************
// Actual compute of pressure underrelaxation
//****************************************************************************
void 
RBGSSolver::press_underrelax(const ProcessorContext*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{
  int numGhostCells = 0;
  int matlIndex = 0;

  // Get the pressure from the old DW and pressure coefficients and non-linear
  // source terms from the new DW
  CCVariable<double> pressure;
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> xPressCoeff;
  new_dw->get(xPressCoeff, d_xPresCoefLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> yPressCoeff;
  new_dw->get(yPressCoeff, d_yPresCoefLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> zPressCoeff;
  new_dw->get(zPressCoeff, d_zPresCoefLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> xPressNlSrc;
  new_dw->get(xPressNlSrc, d_xPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> yPressNlSrc;
  new_dw->get(yPressNlSrc, d_yPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> zPressNlSrc;
  new_dw->get(zPressNlSrc, d_zPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
 
  // Get the patch bounds
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  //fortran call
#ifdef WONT_COMPILE_YET
  FORT_UNDERELAX(pressCoeff, pressNonlinearSrc, pressure,
		 lowIndex, highIndex, d_underrelax);
#endif

  // Write the pressure Coefficients and nonlinear source terms into new DW
  new_dw->put(xPressCoeff, d_xPresCoefLabel, matlIndex, patch);
  new_dw->put(yPressCoeff, d_yPresCoefLabel, matlIndex, patch);
  new_dw->put(zPressCoeff, d_zPresCoefLabel, matlIndex, patch);
  new_dw->put(xPressNlSrc, d_xPresNonLinSrcLabel, matlIndex, patch);
  new_dw->put(yPressNlSrc, d_yPresNonLinSrcLabel, matlIndex, patch);
  new_dw->put(zPressNlSrc, d_zPresNonLinSrcLabel, matlIndex, patch);
}

//****************************************************************************
// Actual linear solve
//****************************************************************************
void 
RBGSSolver::press_lisolve(const ProcessorContext*,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw)
{
  int numGhostCells = 0;
  int matlIndex = 0;

  // Get the pressure from the old DW and pressure coefficients and non-linear
  // source terms from the new DW
  CCVariable<double> pressure;
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> xPressCoeff;
  new_dw->get(xPressCoeff, d_xPresCoefLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> yPressCoeff;
  new_dw->get(yPressCoeff, d_yPresCoefLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> zPressCoeff;
  new_dw->get(zPressCoeff, d_zPresCoefLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> xPressNlSrc;
  new_dw->get(xPressNlSrc, d_xPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> yPressNlSrc;
  new_dw->get(yPressNlSrc, d_yPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> zPressNlSrc;
  new_dw->get(zPressNlSrc, d_zPresNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
 
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

#ifdef WONT_COMPILE_YET
  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(pressCoeff, pressNonlinearSrc, pressure,
	      lowIndex, highIndex);
#endif

  new_dw->put(pressure, d_pressureLabel, matlIndex, patch);
}

//****************************************************************************
// Calculate pressure residuals
//****************************************************************************
void 
RBGSSolver::press_residCalculation(const ProcessorContext* ,
				   const Patch* ,
				   DataWarehouseP& ,
				   DataWarehouseP& )
{
}

//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
RBGSSolver::vel_underrelax(const ProcessorContext* ,
			   const Patch* ,
			   DataWarehouseP& ,
			   DataWarehouseP& , 
			   int index)
{
  index = 0;
}

//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
RBGSSolver::vel_lisolve(const ProcessorContext* ,
			const Patch* ,
			DataWarehouseP& ,
			DataWarehouseP& , 
			int index)
{
  index = 0;
}

//****************************************************************************
// Calculate Velocity residuals
//****************************************************************************
void 
RBGSSolver::vel_residCalculation(const ProcessorContext* ,
				 const Patch* ,
				 DataWarehouseP& ,
				 DataWarehouseP& , 
				 int index)
{
  index = 0;
}

//****************************************************************************
// Scalar Underrelaxation
//****************************************************************************
void 
RBGSSolver::scalar_underrelax(const ProcessorContext* ,
			      const Patch* ,
			      DataWarehouseP& ,
			      DataWarehouseP& , 
			      int index)
{
  index= 0;
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RBGSSolver::scalar_lisolve(const ProcessorContext* ,
			   const Patch* ,
			   DataWarehouseP& ,
			   DataWarehouseP& , 
			   int index)
{
  index = 0;
}

//****************************************************************************
// Calculate Scalar residuals
//****************************************************************************
void 
RBGSSolver::scalar_residCalculation(const ProcessorContext* ,
				    const Patch* ,
				    DataWarehouseP& ,
				    DataWarehouseP& , 
				    int index)
{
  index = 0;
}

//
// $Log$
// Revision 1.5  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.4  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

