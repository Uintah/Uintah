//----- RBGSSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Default constructor for RBGSSolver
//****************************************************************************
RBGSSolver::RBGSSolver()
{
  // Pressure Solve requires (inputs)
  d_pressureINLabel = scinew VarLabel("pressureIN",
				    CCVariable<double>::getTypeDescription() );
  d_presCoefPBLMLabel = scinew VarLabel("presCoefPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );

  // Pressure Solve Computes
  d_presResidualPSLabel = scinew VarLabel("presResidualPS",
				    CCVariable<double>::getTypeDescription() );
  d_presCoefPSLabel = scinew VarLabel("presCoefPS",
				    CCVariable<double>::getTypeDescription() );
  d_presNonLinSrcPSLabel = scinew VarLabel("presNonLinSrcPS",
				    CCVariable<double>::getTypeDescription() );
  d_pressurePSLabel = scinew VarLabel("pressurePS",
				    CCVariable<double>::getTypeDescription() );

  // Momentum Solve requires (inputs)
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelCoefMBLMLabel = scinew VarLabel("uVelCoefMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelCoefMBLMLabel = scinew VarLabel("vVelCoefMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelCoefMBLMLabel = scinew VarLabel("wVelCoefMBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				    SFCZVariable<double>::getTypeDescription() );

  // Momentum Solve Computes
  d_uVelResidualMSLabel = scinew VarLabel("uVelResidualMS",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelResidualMSLabel = scinew VarLabel("vVelResidualMS",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelResidualMSLabel = scinew VarLabel("wVelResidualMS",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelCoefMSLabel = scinew VarLabel("uVelCoefMS",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelCoefMSLabel = scinew VarLabel("vVelCoefMS",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelCoefMSLabel = scinew VarLabel("wVelCoefMS",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMSLabel = scinew VarLabel("uVelNonLinSrcMS",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMSLabel = scinew VarLabel("vVelNonLinSrcMS",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMSLabel = scinew VarLabel("wVelNonLinSrcMS",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				    SFCZVariable<double>::getTypeDescription() );

  // Momentum Solve Requires/Computes
  d_scalarSPLabel = scinew VarLabel("scalarSP",
				    CCVariable<double>::getTypeDescription() );
  d_scalCoefSBLMLabel = scinew VarLabel("scalCoefSBLM",
				    CCVariable<double>::getTypeDescription() );
  d_scalNonLinSrcSBLMLabel = scinew VarLabel("scalNonLinSrcSBLM",
				    CCVariable<double>::getTypeDescription() );
  d_scalResidualSSLabel = scinew VarLabel("scalResidualSS",
				    CCVariable<double>::getTypeDescription() );
  d_scalCoefSSLabel = scinew VarLabel("scalCoefSS",
				    CCVariable<double>::getTypeDescription() );
  d_scalNonLinSrcSSLabel = scinew VarLabel("scalNonLinSrcSS",
				    CCVariable<double>::getTypeDescription() );
  d_scalarSSLabel = scinew VarLabel("scalarSS",
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
      tsk->requires(old_dw, d_pressureINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_presCoefPBLMLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // computes global residual
      tsk->computes(new_dw, d_presResidualPSLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
    {
      Task* tsk = scinew Task("RBGSSolver::press_underrelax",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::press_underrelax);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      tsk->requires(old_dw, d_pressureINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_presCoefPBLMLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_presNonLinSrcPBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

      // computes 
      tsk->computes(new_dw, d_presCoefPSLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presNonLinSrcPSLabel, matlIndex, patch);

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
      tsk->requires(old_dw, d_pressureINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_presCoefPSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_presNonLinSrcPSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

      // Computes
      tsk->computes(new_dw, d_pressurePSLabel, matlIndex, patch);

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
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("RBGSSolver::vel_residual",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::vel_residCalculation,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      switch(index) {
      case 0:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_uVelCoefMBLMLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	// computes global residual
	tsk->computes(new_dw, d_uVelResidualMSLabel, matlIndex, patch);
	break;
      case 1:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_vVelCoefMBLMLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	// computes global residual
	tsk->computes(new_dw, d_vVelResidualMSLabel, matlIndex, patch);
	break;
      case 2:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_wVelCoefMBLMLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	// computes global residual
	tsk->computes(new_dw, d_wVelResidualMSLabel, matlIndex, patch);
	break;
      default:
	throw InvalidValue("Valid velocity index = 0 or 1 or 2");
      }
      sched->addTask(tsk);
    }
    {
      Task* tsk = scinew Task("RBGSSolver::vel_underrelax",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::vel_underrelax,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      switch(index) {
      case 0:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_uVelCoefMBLMLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_uVelNonLinSrcMBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

	// computes 
	tsk->computes(new_dw, d_uVelCoefMSLabel, matlIndex, patch);
	tsk->computes(new_dw, d_uVelNonLinSrcMSLabel, matlIndex, patch);

	break;

      case 1:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_vVelCoefMBLMLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

	// computes 
	tsk->computes(new_dw, d_vVelCoefMSLabel, matlIndex, patch);
	tsk->computes(new_dw, d_vVelNonLinSrcMSLabel, matlIndex, patch);

	break;

      case 2:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_wVelCoefMBLMLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_wVelNonLinSrcMBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

	// computes 
	tsk->computes(new_dw, d_wVelCoefMSLabel, matlIndex, patch);
	tsk->computes(new_dw, d_wVelNonLinSrcMSLabel, matlIndex, patch);

	break;

      default:
	throw InvalidValue("Valid velocity index = 0 or 1 or 2");
      }

      sched->addTask(tsk);
    }
    {
      // use a recursive task based on number of sweeps reqd
      Task* tsk = scinew Task("RBGSSolver::vel_lisolve",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::vel_lisolve,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      switch(index) {
      case 0:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_uVelCoefMSLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_uVelNonLinSrcMSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
	// Computes
	tsk->computes(new_dw, d_uVelocityMSLabel, matlIndex, patch);
	break;
      case 1:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_vVelCoefMSLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_vVelNonLinSrcMSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
	// Computes
	tsk->computes(new_dw, d_vVelocityMSLabel, matlIndex, patch);
	break;
      case 2:
	// coefficient for the variable for which solve is invoked
	tsk->requires(old_dw, d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_wVelCoefMSLabel, matlIndex, patch, 
		      Ghost::None, numGhostCells);
	tsk->requires(new_dw, d_wVelNonLinSrcMSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
	// Computes
	tsk->computes(new_dw, d_wVelocityMSLabel, matlIndex, patch);
	break;
      default:
	throw InvalidValue("Valid velocity index = 0 or 1 or 2");
      }

      sched->addTask(tsk);
    }
    // add another task that computes the linear residual
  }
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
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("RBGSSolver::scalar_residual",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::scalar_residCalculation,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      tsk->requires(old_dw, d_scalarSPLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalCoefSBLMLabel, index, patch, Ghost::None,
		    numGhostCells);

      // computes global residual
      tsk->computes(new_dw, d_scalResidualSSLabel, index, patch);

      sched->addTask(tsk);
    }
    {
      Task* tsk = scinew Task("RBGSSolver::scalar_underrelax",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::scalar_underrelax,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      tsk->requires(old_dw, d_scalarSPLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalCoefSBLMLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalNonLinSrcSBLMLabel, index, patch, 
		    Ghost::None, numGhostCells);

      // computes 
      tsk->computes(new_dw, d_scalCoefSSLabel, index, patch);
      tsk->computes(new_dw, d_scalNonLinSrcSSLabel, index, patch);

      sched->addTask(tsk);
    }
    {
      // use a recursive task based on number of sweeps reqd
      Task* tsk = scinew Task("RBGSSolver::scalar_lisolve",
			      patch, old_dw, new_dw, this,
			      &RBGSSolver::scalar_lisolve,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      // coefficient for the variable for which solve is invoked
      tsk->requires(old_dw, d_scalarSPLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalCoefSSLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalNonLinSrcSSLabel, index, patch, 
		    Ghost::None, numGhostCells);

      // Computes
      tsk->computes(new_dw, d_scalarSSLabel, index, patch);

      sched->addTask(tsk);
    }
    // add another task that computes the linear residual
  }    
}

//****************************************************************************
// Actual compute of pressure underrelaxation
//****************************************************************************
void 
RBGSSolver::press_underrelax(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  // Patch based variables
  CCVariable<double> pressure;
  StencilMatrix<CCVariable<double> > pressCoeff;
  CCVariable<double> pressNonLinSrc;

  // Get the pressure from the old DW and pressure coefficients and non-linear
  // source terms from the new DW
  old_dw->get(pressure, d_pressureINLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) {
     new_dw->get(pressCoeff[ii], d_presCoefPBLMLabel, matlIndex, patch, 
		 Ghost::None, numGhostCells);
  }
  new_dw->get(pressNonLinSrc, d_presNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = pressure.getFortLowIndex();
  IntVector domHi = pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
#ifdef WONT_COMPILE_YET
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 pressure.getPointer(),
		 pressCoeff[StencilMatrix::AP].getPointer(), 
		 pressCoeff[StencilMatrix::AE].getPointer(), 
		 pressCoeff[StencilMatrix::AW].getPointer(), 
		 pressCoeff[StencilMatrix::AN].getPointer(), 
		 pressCoeff[StencilMatrix::AS].getPointer(), 
		 pressCoeff[StencilMatrix::AT].getPointer(), 
		 pressCoeff[StencilMatrix::AB].getPointer(), 
		 pressNonLinSrc.getPointer(), 
		 d_underrelax);
#endif

  // Write the pressure Coefficients and nonlinear source terms into new DW
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefPSLabel, matlIndex, patch);
  }
  new_dw->put(pressNonLinSrc, d_presNonLinSrcPSLabel, matlIndex, patch);
}

//****************************************************************************
// Actual linear solve
//****************************************************************************
void 
RBGSSolver::press_lisolve(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  // Variables
  CCVariable<double> pressure;
  StencilMatrix<CCVariable<double> > pressCoeff;
  CCVariable<double> pressNonLinSrc;

  // Get the pressure from the old DW and pressure coefficients and non-linear
  // source terms from the new DW
  old_dw->get(pressure, d_pressureINLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(pressCoeff[ii], d_presCoefPSLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  new_dw->get(pressNonLinSrc, d_presNonLinSrcPSLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = pressure.getFortLowIndex();
  IntVector domHi = pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  pressure.getPointer(),
		  pressCoeff[StencilMatrix::AP].getPointer(), 
		  pressCoeff[StencilMatrix::AE].getPointer(), 
		  pressCoeff[StencilMatrix::AW].getPointer(), 
		  pressCoeff[StencilMatrix::AN].getPointer(), 
		  pressCoeff[StencilMatrix::AS].getPointer(), 
		  pressCoeff[StencilMatrix::AT].getPointer(), 
		  pressCoeff[StencilMatrix::AB].getPointer(), 
		  pressNonLinSrc.getPointer());
#endif

  new_dw->put(pressure, d_pressurePSLabel, matlIndex, patch);
}

//****************************************************************************
// Calculate pressure residuals
//****************************************************************************
void 
RBGSSolver::press_residCalculation(const ProcessorGroup* ,
				   const Patch* ,
				   DataWarehouseP& ,
				   DataWarehouseP& )
{
}

//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
RBGSSolver::vel_underrelax(const ProcessorGroup* ,
			   const Patch* patch,
			   DataWarehouseP& old_dw ,
			   DataWarehouseP& new_dw, 
			   int index)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  switch(index) {
  case 0:
    {
      // Patch based variables
      SFCXVariable<double> velocity;
      StencilMatrix<SFCXVariable<double> > velCoeff;
      SFCXVariable<double> velNonLinSrc;

      // Get the velocity from the old DW and velocity coefficients and non-linear
      // source terms from the new DW
      old_dw->get(velocity, d_uVelocityCPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->get(velCoeff[ii], d_uVelCoefMBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      }
      new_dw->get(velNonLinSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
 
      // Get the patch bounds and the variable bounds
      IntVector domLo = velocity.getFortLowIndex();
      IntVector domHi = velocity.getFortHighIndex();
      IntVector idxLo = patch->getSFCXFORTLowIndex();
      IntVector idxHi = patch->getSFCXFORTHighIndex();

      //fortran call
#ifdef WONT_COMPILE_YET
      FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		     idxLo.get_pointer(), idxHi.get_pointer(),
		     velocity.getPointer(),
		     velCoeff[StencilMatrix::AP].getPointer(), 
		     velCoeff[StencilMatrix::AE].getPointer(), 
		     velCoeff[StencilMatrix::AW].getPointer(), 
		     velCoeff[StencilMatrix::AN].getPointer(), 
		     velCoeff[StencilMatrix::AS].getPointer(), 
		     velCoeff[StencilMatrix::AT].getPointer(), 
		     velCoeff[StencilMatrix::AB].getPointer(), 
		     velNonLinSrc.getPointer(), 
		     d_underrelax);
#endif

      // Write the velocity Coefficients and nonlinear source terms into new DW
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->put(velCoeff[ii], d_uVelCoefMSLabel, matlIndex, patch);
      }
      new_dw->put(velNonLinSrc, d_uVelNonLinSrcMSLabel, matlIndex, patch);
    }
    break;
  case 1:
    {
      // Patch based variables
      SFCYVariable<double> velocity;
      StencilMatrix<SFCYVariable<double> > velCoeff;
      SFCYVariable<double> velNonLinSrc;

      // Get the velocity from the old DW and velocity coefficients and non-linear
      // source terms from the new DW
      old_dw->get(velocity, d_vVelocityCPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->get(velCoeff[ii], d_vVelCoefMBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      }
      new_dw->get(velNonLinSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);

      // Get the patch bounds and the variable bounds
      IntVector domLo = velocity.getFortLowIndex();
      IntVector domHi = velocity.getFortHighIndex();
      IntVector idxLo = patch->getSFCYFORTLowIndex();
      IntVector idxHi = patch->getSFCYFORTHighIndex();

      //fortran call
#ifdef WONT_COMPILE_YET
      FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		     idxLo.get_pointer(), idxHi.get_pointer(),
		     velocity.getPointer(),
		     velCoeff[StencilMatrix::AP].getPointer(), 
		     velCoeff[StencilMatrix::AE].getPointer(), 
		     velCoeff[StencilMatrix::AW].getPointer(), 
		     velCoeff[StencilMatrix::AN].getPointer(), 
		     velCoeff[StencilMatrix::AS].getPointer(), 
		     velCoeff[StencilMatrix::AT].getPointer(), 
		     velCoeff[StencilMatrix::AB].getPointer(), 
		     velNonLinSrc.getPointer(), 
		     d_underrelax);
#endif

      // Write the velocity Coefficients and nonlinear source terms into new DW
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->put(velCoeff[ii], d_vVelCoefMSLabel, matlIndex, patch);
      }
      new_dw->put(velNonLinSrc, d_vVelNonLinSrcMSLabel, matlIndex, patch);
    }
    break;
  case 2:
    {
      // Patch based variables
      SFCZVariable<double> velocity;
      StencilMatrix<SFCZVariable<double> > velCoeff;
      SFCZVariable<double> velNonLinSrc;
      
      // Get the velocity from the old DW and velocity coefficients and non-linear
      // source terms from the new DW
      old_dw->get(velocity, d_wVelocityCPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->get(velCoeff[ii], d_wVelCoefMBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      }
      new_dw->get(velNonLinSrc, d_wVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);

      // Get the patch bounds and the variable bounds
      IntVector domLo = velocity.getFortLowIndex();
      IntVector domHi = velocity.getFortHighIndex();
      IntVector idxLo = patch->getSFCZFORTLowIndex();
      IntVector idxHi = patch->getSFCZFORTHighIndex();

      //fortran call
#ifdef WONT_COMPILE_YET
      FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		     idxLo.get_pointer(), idxHi.get_pointer(),
		     velocity.getPointer(),
		     velCoeff[StencilMatrix::AP].getPointer(), 
		     velCoeff[StencilMatrix::AE].getPointer(), 
		     velCoeff[StencilMatrix::AW].getPointer(), 
		     velCoeff[StencilMatrix::AN].getPointer(), 
		     velCoeff[StencilMatrix::AS].getPointer(), 
		     velCoeff[StencilMatrix::AT].getPointer(), 
		     velCoeff[StencilMatrix::AB].getPointer(), 
		     velNonLinSrc.getPointer(), 
		     d_underrelax);
#endif

      // Write the velocity Coefficients and nonlinear source terms into new DW
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->put(velCoeff[ii], d_wVelCoefMSLabel, matlIndex, patch);
      }
      new_dw->put(velNonLinSrc, d_wVelNonLinSrcMSLabel, matlIndex, patch);
    }
    break;
  default:
    throw InvalidValue("Valid velocity index = 0 or 1 or 2");
  }
}

//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
RBGSSolver::vel_lisolve(const ProcessorGroup* ,
			const Patch* patch,
			DataWarehouseP& old_dw ,
			DataWarehouseP& new_dw, 
			int index)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  switch(index) {
  case 0:
    {
      // Variables
      SFCXVariable<double> velocity;
      StencilMatrix<SFCXVariable<double> > velCoeff;
      SFCXVariable<double> velNonLinSrc;

      // Get the velocity from the old DW and velocity coefficients and non-linear
      // source terms from the new DW
      old_dw->get(velocity, d_uVelocityCPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->get(velCoeff[ii], d_uVelCoefMSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      }
      new_dw->get(velNonLinSrc, d_uVelNonLinSrcMSLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);

      // Get the patch bounds and the variable bounds
      IntVector domLo = velocity.getFortLowIndex();
      IntVector domHi = velocity.getFortHighIndex();
      IntVector idxLo = patch->getSFCXFORTLowIndex();
      IntVector idxHi = patch->getSFCXFORTHighIndex();
      
#ifdef WONT_COMPILE_YET
      //fortran call for red-black GS solver
      FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      velocity.getPointer(),
		      velCoeff[StencilMatrix::AP].getPointer(), 
		      velCoeff[StencilMatrix::AE].getPointer(), 
		      velCoeff[StencilMatrix::AW].getPointer(), 
		      velCoeff[StencilMatrix::AN].getPointer(), 
		      velCoeff[StencilMatrix::AS].getPointer(), 
		      velCoeff[StencilMatrix::AT].getPointer(), 
		      velCoeff[StencilMatrix::AB].getPointer(), 
		      velNonLinSrc.getPointer());
#endif

      new_dw->put(velocity, d_uVelocityMSLabel, matlIndex, patch);
    }
    break;
  case 1:
    {
      // Variables
      SFCYVariable<double> velocity;
      StencilMatrix<SFCYVariable<double> > velCoeff;
      SFCYVariable<double> velNonLinSrc;

      // Get the velocity from the old DW and velocity coefficients and non-linear
      // source terms from the new DW
      old_dw->get(velocity, d_vVelocityCPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->get(velCoeff[ii], d_vVelCoefMSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      }
      new_dw->get(velNonLinSrc, d_vVelNonLinSrcMSLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);

      // Get the patch bounds and the variable bounds
      IntVector domLo = velocity.getFortLowIndex();
      IntVector domHi = velocity.getFortHighIndex();
      IntVector idxLo = patch->getSFCYFORTLowIndex();
      IntVector idxHi = patch->getSFCYFORTHighIndex();
      
#ifdef WONT_COMPILE_YET
      //fortran call for red-black GS solver
      FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      velocity.getPointer(),
		      velCoeff[StencilMatrix::AP].getPointer(), 
		      velCoeff[StencilMatrix::AE].getPointer(), 
		      velCoeff[StencilMatrix::AW].getPointer(), 
		      velCoeff[StencilMatrix::AN].getPointer(), 
		      velCoeff[StencilMatrix::AS].getPointer(), 
		      velCoeff[StencilMatrix::AT].getPointer(), 
		      velCoeff[StencilMatrix::AB].getPointer(), 
		      velNonLinSrc.getPointer());
#endif

      new_dw->put(velocity, d_vVelocityMSLabel, matlIndex, patch);
    }
    break;
  case 2:
    {
      // Variables
      SFCZVariable<double> velocity;
      StencilMatrix<SFCZVariable<double> > velCoeff;
      SFCZVariable<double> velNonLinSrc;
      
      // Get the velocity from the old DW and velocity coefficients and non-linear
      // source terms from the new DW
      old_dw->get(velocity, d_wVelocityCPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) {
	new_dw->get(velCoeff[ii], d_wVelCoefMSLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      }
      new_dw->get(velNonLinSrc, d_wVelNonLinSrcMSLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      
      // Get the patch bounds and the variable bounds
      IntVector domLo = velocity.getFortLowIndex();
      IntVector domHi = velocity.getFortHighIndex();
      IntVector idxLo = patch->getSFCZFORTLowIndex();
      IntVector idxHi = patch->getSFCZFORTHighIndex();
      
#ifdef WONT_COMPILE_YET
      //fortran call for red-black GS solver
      FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      velocity.getPointer(),
		      velCoeff[StencilMatrix::AP].getPointer(), 
		      velCoeff[StencilMatrix::AE].getPointer(), 
		      velCoeff[StencilMatrix::AW].getPointer(), 
		      velCoeff[StencilMatrix::AN].getPointer(), 
		      velCoeff[StencilMatrix::AS].getPointer(), 
		      velCoeff[StencilMatrix::AT].getPointer(), 
		      velCoeff[StencilMatrix::AB].getPointer(), 
		      velNonLinSrc.getPointer());
#endif
      
      new_dw->put(velocity, d_wVelocityMSLabel, matlIndex, patch);
    }
    break;
  default:
    throw InvalidValue("Valid velocity index = 0 or 1 or 2");
  }
}

//****************************************************************************
// Calculate Velocity residuals
//****************************************************************************
void 
RBGSSolver::vel_residCalculation(const ProcessorGroup* ,
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
RBGSSolver::scalar_underrelax(const ProcessorGroup* ,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw, 
			      int index)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  // Patch based variables
  CCVariable<double> scalar;
  StencilMatrix<CCVariable<double> > scalarCoeff;
  CCVariable<double> scalarNonLinSrc;

  // Get the scalar from the old DW and scalar coefficients and non-linear
  // source terms from the new DW
  old_dw->get(scalar, d_scalarSPLabel, index, patch, Ghost::None,
	      numGhostCells);

  // ** WARNING ** scalarCoeff is not being read in corrctly .. may
  //               have to create new type instead of StencilMatrix
  for (int ii = 0; ii < nofStencils; ii++) {
     new_dw->get(scalarCoeff[ii], d_scalCoefSBLMLabel, index, patch, 
		 Ghost::None, numGhostCells);
  }
  new_dw->get(scalarNonLinSrc, d_scalNonLinSrcSBLMLabel, index, patch, 
	      Ghost::None, numGhostCells);
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = scalar.getFortLowIndex();
  IntVector domHi = scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
#ifdef WONT_COMPILE_YET
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 scalar.getPointer(),
		 scalarCoeff[StencilMatrix::AP].getPointer(), 
		 scalarCoeff[StencilMatrix::AE].getPointer(), 
		 scalarCoeff[StencilMatrix::AW].getPointer(), 
		 scalarCoeff[StencilMatrix::AN].getPointer(), 
		 scalarCoeff[StencilMatrix::AS].getPointer(), 
		 scalarCoeff[StencilMatrix::AT].getPointer(), 
		 scalarCoeff[StencilMatrix::AB].getPointer(), 
		 scalarNonLinSrc.getPointer(), 
		 d_underrelax);
#endif

  // Write the scalar Coefficients and nonlinear source terms into new DW
  // ** WARNING ** scalarCoeff is not being read in corrctly .. may
  //               have to create new type instead of StencilMatrix
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarCoeff[ii], d_scalCoefSSLabel, index, patch);
  }
  new_dw->put(scalarNonLinSrc, d_scalNonLinSrcSSLabel, index, patch);
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RBGSSolver::scalar_lisolve(const ProcessorGroup* ,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw, 
			   int index)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  // Variables
  CCVariable<double> scalar;
  StencilMatrix<CCVariable<double> > scalarCoeff;
  CCVariable<double> scalarNonLinSrc;

  // Get the scalar from the old DW and scalar coefficients and non-linear
  // source terms from the new DW
  old_dw->get(scalar, d_scalarSPLabel, index, patch, Ghost::None,
	      numGhostCells);

  // ** WARNING ** scalarCoeff is not being read in corrctly .. may
  //               have to create new type instead of StencilMatrix
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(scalarCoeff[ii], d_scalCoefSSLabel, index, patch, 
		Ghost::None, numGhostCells);
  }

  new_dw->get(scalarNonLinSrc, d_scalNonLinSrcSSLabel, index, patch, 
	      Ghost::None, numGhostCells);
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = scalar.getFortLowIndex();
  IntVector domHi = scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  scalar.getPointer(),
		  scalarCoeff[StencilMatrix::AP].getPointer(), 
		  scalarCoeff[StencilMatrix::AE].getPointer(), 
		  scalarCoeff[StencilMatrix::AW].getPointer(), 
		  scalarCoeff[StencilMatrix::AN].getPointer(), 
		  scalarCoeff[StencilMatrix::AS].getPointer(), 
		  scalarCoeff[StencilMatrix::AT].getPointer(), 
		  scalarCoeff[StencilMatrix::AB].getPointer(), 
		  scalarNonLinSrc.getPointer());
		  d_underrelax);
#endif

  new_dw->put(scalar, d_scalarSSLabel, index, patch);
}

//****************************************************************************
// Calculate Scalar residuals
//****************************************************************************
void 
RBGSSolver::scalar_residCalculation(const ProcessorGroup* ,
				    const Patch* ,
				    DataWarehouseP& ,
				    DataWarehouseP& , 
				    int index)
{
  index = 0;
}

//
// $Log$
// Revision 1.11  2000/06/29 22:56:43  bbanerje
// Changed FCVars to SFC[X,Y,Z]Vars, and added the neceesary getIndex calls.
//
// Revision 1.10  2000/06/22 23:06:37  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.9  2000/06/21 07:51:01  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.8  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.7  2000/06/17 07:06:26  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.6  2000/06/12 21:29:59  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
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

