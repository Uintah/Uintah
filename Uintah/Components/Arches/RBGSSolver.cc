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
#include <Uintah/Components/Arches/ArchesVariables.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/ReductionVariable.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Default constructor for RBGSSolver
//****************************************************************************
RBGSSolver::RBGSSolver()
{

  // Momentum Solve requires (inputs)

  // Scalar Solve Requires/Computes
  d_scalarINLabel = scinew VarLabel("scalarIN",
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
  d_scalarSPLabel = scinew VarLabel("scalarSP",
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
      tsk->requires(old_dw, d_scalarINLabel, index, patch, Ghost::None,
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
      tsk->requires(old_dw, d_scalarINLabel, index, patch, Ghost::None,
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
      tsk->requires(old_dw, d_scalarINLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalCoefSSLabel, index, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_scalNonLinSrcSSLabel, index, patch, 
		    Ghost::None, numGhostCells);

      // Computes
      tsk->computes(new_dw, d_scalarSPLabel, index, patch);

      sched->addTask(tsk);
    }
    // add another task that computes the linear residual
  }    
}

//****************************************************************************
// Actual compute of pressure underrelaxation
//****************************************************************************
void 
RBGSSolver::computePressResidual(const ProcessorGroup*,
				 const Patch* patch,
				 DataWarehouseP&,
				 DataWarehouseP&,
				 ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
#ifdef WONT_COMPILE_YET
  FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->pressure.getPointer(),
		    vars->pressCoeff[Arches::AP].getPointer(), 
		    vars->pressCoeff[Arches::AE].getPointer(), 
		    vars->pressCoeff[Arches::AW].getPointer(), 
		    vars->pressCoeff[Arches::AN].getPointer(), 
		    vars->pressCoeff[Arches::AS].getPointer(), 
		    vars->pressCoeff[Arches::AT].getPointer(), 
		    vars->pressCoeff[Arches::AB].getPointer(), 
		    vars->pressNonLinSrc.getPointer(),
		    &vars->residPress, &vars->truncPress);

); 
#endif

}

void 
RBGSSolver::computePressUnderrelax(const ProcessorGroup*,
				   const Patch* patch,
				   DataWarehouseP&,
				   DataWarehouseP&, 
				   ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
#ifdef WONT_COMPILE_YET
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->pressure.getPointer(),
		 vars->pressCoeff[Arches::AP].getPointer(), 
		 vars->pressCoeff[Arches::AE].getPointer(), 
		 vars->pressCoeff[Arches::AW].getPointer(), 
		 vars->pressCoeff[Arches::AN].getPointer(), 
		 vars->pressCoeff[Arches::AS].getPointer(), 
		 vars->pressCoeff[Arches::AT].getPointer(), 
		 vars->pressCoeff[Arches::AB].getPointer(), 
		 vars->pressNonLinSrc.getPointer(), 
		 &d_underrelax);
#endif

}

//****************************************************************************
// Actual linear solve
//****************************************************************************
void 
RBGSSolver::pressLisolve(const ProcessorGroup*,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw,
			 ArchesVariables* vars)
{
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->pressure.getPointer(),
		  vars->pressCoeff[Arches::AP].getPointer(), 
		  vars->pressCoeff[Arches::AE].getPointer(), 
		  vars->pressCoeff[Arches::AW].getPointer(), 
		  vars->pressCoeff[Arches::AN].getPointer(), 
		  vars->pressCoeff[Arches::AS].getPointer(), 
		  vars->pressCoeff[Arches::AT].getPointer(), 
		  vars->pressCoeff[Arches::AB].getPointer(), 
		  vars->pressNonLinSrc.getPointer());
#endif

}

void 
RBGSSolver::computeVelResidual(const ProcessorGroup* ,
			       const Patch* patch,
			       DataWarehouseP& old_dw ,
			       DataWarehouseP& new_dw, 
			       int index, ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->uVelocity.getPointer(),
		      vars->uVelocityCoeff[Arches::AP].getPointer(), 
		      vars->uVelocityCoeff[Arches::AE].getPointer(), 
		      vars->uVelocityCoeff[Arches::AW].getPointer(), 
		      vars->uVelocityCoeff[Arches::AN].getPointer(), 
		      vars->uVelocityCoeff[Arches::AS].getPointer(), 
		      vars->uVelocityCoeff[Arches::AT].getPointer(), 
		      vars->uVelocityCoeff[Arches::AB].getPointer(), 
		      vars->uVelNonLinSrc.getPointer(),
		      &vars->residUVel, &vars->truncUVel);


#endif
    break;
    case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->vVelocity.getPointer(),
		      vars->vVelocityCoeff[Arches::AP].getPointer(), 
		      vars->vVelocityCoeff[Arches::AE].getPointer(), 
		      vars->vVelocityCoeff[Arches::AW].getPointer(), 
		      vars->vVelocityCoeff[Arches::AN].getPointer(), 
		      vars->vVelocityCoeff[Arches::AS].getPointer(), 
		      vars->vVelocityCoeff[Arches::AT].getPointer(), 
		      vars->vVelocityCoeff[Arches::AB].getPointer(), 
		      vars->vVelNonLinSrc.getPointer(),
		      &vars->residVVel, &vars->truncVVel);


#endif
    break;
    case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->wVelocity.getPointer(),
		      vars->wVelocityCoeff[Arches::AP].getPointer(), 
		      vars->wVelocityCoeff[Arches::AE].getPointer(), 
		      vars->wVelocityCoeff[Arches::AW].getPointer(), 
		      vars->wVelocityCoeff[Arches::AN].getPointer(), 
		      vars->wVelocityCoeff[Arches::AS].getPointer(), 
		      vars->wVelocityCoeff[Arches::AT].getPointer(), 
		      vars->wVelocityCoeff[Arches::AB].getPointer(), 
		      vars->wVelNonLinSrc.getPointer(),
		      &vars->residWVel, &vars->truncWVel);


#endif
    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}

//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
RBGSSolver::computeVelUnderrelax(const ProcessorGroup* ,
				 const Patch* patch,
				 DataWarehouseP& old_dw ,
				 DataWarehouseP& new_dw, 
				 int index, ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonLinSrc.getPointer(),
		   &d_underrelax);


#endif
    break;
    case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonLinSrc.getPointer(),
		   &d_underrelax);


#endif
    break;
    case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonLinSrc.getPointer(),
		   &d_underrelax);

#endif
    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}


//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
RBGSSolver::velocityLisolve(const ProcessorGroup* ,
			    const Patch* patch,
			    DataWarehouseP& old_dw ,
			    DataWarehouseP& new_dw, 
			    int index, ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->uVelocity.getPointer(),
		    vars->uVelocityCoeff[Arches::AP].getPointer(), 
		    vars->uVelocityCoeff[Arches::AE].getPointer(), 
		    vars->uVelocityCoeff[Arches::AW].getPointer(), 
		    vars->uVelocityCoeff[Arches::AN].getPointer(), 
		    vars->uVelocityCoeff[Arches::AS].getPointer(), 
		    vars->uVelocityCoeff[Arches::AT].getPointer(), 
		    vars->uVelocityCoeff[Arches::AB].getPointer(), 
		    vars->uVelNonLinSrc.getPointer());


#endif
    break;
    case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->vVelocity.getPointer(),
		    vars->vVelocityCoeff[Arches::AP].getPointer(), 
		    vars->vVelocityCoeff[Arches::AE].getPointer(), 
		    vars->vVelocityCoeff[Arches::AW].getPointer(), 
		    vars->vVelocityCoeff[Arches::AN].getPointer(), 
		    vars->vVelocityCoeff[Arches::AS].getPointer(), 
		    vars->vVelocityCoeff[Arches::AT].getPointer(), 
		    vars->vVelocityCoeff[Arches::AB].getPointer(), 
		    vars->vVelNonLinSrc.getPointer());


#endif
    break;
    case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->wVelocity.getPointer(),
		    vars->wVelocityCoeff[Arches::AP].getPointer(), 
		    vars->wVelocityCoeff[Arches::AE].getPointer(), 
		    vars->wVelocityCoeff[Arches::AW].getPointer(), 
		    vars->wVelocityCoeff[Arches::AN].getPointer(), 
		    vars->wVelocityCoeff[Arches::AS].getPointer(), 
		    vars->wVelocityCoeff[Arches::AT].getPointer(), 
		    vars->wVelocityCoeff[Arches::AB].getPointer(), 
		    vars->wVelNonLinSrc.getPointer());



#endif
    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
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
		 scalarCoeff[Arches::AP].getPointer(), 
		 scalarCoeff[Arches::AE].getPointer(), 
		 scalarCoeff[Arches::AW].getPointer(), 
		 scalarCoeff[Arches::AN].getPointer(), 
		 scalarCoeff[Arches::AS].getPointer(), 
		 scalarCoeff[Arches::AT].getPointer(), 
		 scalarCoeff[Arches::AB].getPointer(), 
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
  old_dw->get(scalar, d_scalarINLabel, index, patch, Ghost::None,
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
		  scalarCoeff[Arches::AP].getPointer(), 
		  scalarCoeff[Arches::AE].getPointer(), 
		  scalarCoeff[Arches::AW].getPointer(), 
		  scalarCoeff[Arches::AN].getPointer(), 
		  scalarCoeff[Arches::AS].getPointer(), 
		  scalarCoeff[Arches::AT].getPointer(), 
		  scalarCoeff[Arches::AB].getPointer(), 
		  scalarNonLinSrc.getPointer());
		  d_underrelax);
#endif

  new_dw->put(scalar, d_scalarSPLabel, index, patch);
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
// Revision 1.14  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.13  2000/07/08 23:42:55  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.12  2000/07/08 08:03:34  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
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

