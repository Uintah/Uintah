//----- ScalarSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/ScalarSolver.h>
#include <Uintah/Components/Arches/CellInformationP.h>
#include <Uintah/Components/Arches/PetscSolver.h>
#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
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
// Default constructor for PressureSolver
//****************************************************************************
ScalarSolver::ScalarSolver(const ArchesLabel* label,
			   TurbulenceModel* turb_model,
			   BoundaryCondition* bndry_cond,
			   PhysicalConstants* physConst) :
                                 d_lab(label),
                                 d_turbModel(turb_model), 
                                 d_boundaryCondition(bndry_cond),
				 d_physicalConsts(physConst)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
ScalarSolver::~ScalarSolver()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MixtureFractionSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "second") 
    d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
    //throw InvalidValue("Finite Differencing scheme "
	//	       "not supported: " + finite_diff, db);
  }
  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs")
    d_linearSolver = scinew RBGSSolver();
  else if (linear_sol == "petsc")
     d_linearSolver = scinew PetscSolver(0); // CHEAT - steve d_myworld);
  else {
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol);
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
  }
  d_linearSolver->problemSetup(db);
}

//****************************************************************************
// Schedule solve of linearized scalar equation
//****************************************************************************
void 
ScalarSolver::solve(const LevelP& level,
		    SchedulerP& sched,
		    DataWarehouseP& old_dw,
		    DataWarehouseP& new_dw,
		    double /*time*/, double delta_t, int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
   //DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);

  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(level, sched, old_dw, new_dw, delta_t, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_scalarLinearSolve(level, sched, old_dw, new_dw, delta_t, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(const LevelP& level,
				      SchedulerP& sched,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw,
				      double delta_t, int index)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      // steve: requires two arguments
      //Task* tsk = scinew Task("ScalarSolver::BuildCoeff",
	//		      patch, old_dw, new_dw, this,
	//		      Discretization::buildLinearMatrix,
	//		      delta_t, index);
      Task* tsk = scinew Task("ScalarSolver::BuildCoeff",
			      patch, old_dw, new_dw, this,
			      &ScalarSolver::buildLinearMatrix,
			      delta_t, index);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;

      // This task requires scalar and density from old time step for transient
      // calculation
      //DataWarehouseP old_dw = new_dw->getTop();
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(old_dw, d_lab->d_scalarSPLabel, index, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_densityCPLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(new_dw, d_lab->d_scalarCPBCLabel, index, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);

      // added one more argument of index to specify scalar component
      for (int ii = 0; ii < nofStencils; ii++) {
	tsk->computes(new_dw, d_lab->d_scalCoefSBLMLabel, ii, patch);
      }
      tsk->computes(new_dw, d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch);

      sched->addTask(tsk);
    }

  }
}

//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolve(const LevelP& level,
				      SchedulerP& sched,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw,
				      double delta_t,
				      int index)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("ScalarSolver::scalarLinearSolve",
			   patch, old_dw, new_dw, this,
			   &ScalarSolver::scalarLinearSolve, delta_t, index);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;

      // coefficient for the variable for which solve is invoked
      tsk->requires(old_dw, d_lab->d_densityCPLabel, index, patch,
		    Ghost::None);
      tsk->requires(new_dw, d_lab->d_scalarCPBCLabel, index, patch, 
		    Ghost::AroundCells, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++) 
	tsk->requires(new_dw, d_lab->d_scalCoefSBLMLabel, 
		      ii, patch, Ghost::None, zeroGhostCells);
      tsk->requires(new_dw, d_lab->d_scalNonLinSrcSBLMLabel, 
		    matlIndex, patch, Ghost::None, zeroGhostCells);

      tsk->computes(new_dw, d_lab->d_scalarSPLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}
      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t, int index)
{
  ArchesVariables scalarVars;
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;

  // get old_dw from getTop function
  //DataWarehouseP old_dw = new_dw->getTop();

  // Get the PerPatch CellInformation data
  PerPatch<CellInformationP> cellInfoP;
  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  CellInformation* cellinfo = cellInfoP.get().get_rep();

  // from old_dw get PCELL, DENO, FO(index)
  old_dw->get(scalarVars.cellType, d_lab->d_cellTypeLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  old_dw->get(scalarVars.old_density, d_lab->d_densityCPLabel, 
	      matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(scalarVars.old_scalar, d_lab->d_scalarSPLabel, 
	      index, patch, Ghost::None, zeroGhostCells);

  // from new_dw get DEN, VIS, F(index), U, V, W
  new_dw->get(scalarVars.density, d_lab->d_densityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(scalarVars.viscosity, d_lab->d_viscosityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(scalarVars.scalar, d_lab->d_scalarCPBCLabel, 
	      index, patch, Ghost::None, zeroGhostCells);
  // for explicit get old values
  new_dw->get(scalarVars.uVelocity, d_lab->d_uVelocityCPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(scalarVars.vVelocity, d_lab->d_vVelocityCPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(scalarVars.wVelocity, d_lab->d_wVelocityCPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(scalarVars.scalarCoeff[ii], 
			d_lab->d_scalCoefSBLMLabel, ii, patch);
    new_dw->allocate(scalarVars.scalarConvectCoeff[ii],
			d_lab->d_scalConvCoefSBLMLabel, ii, patch);
  }
  new_dw->allocate(scalarVars.scalarLinearSrc, 
		      d_lab->d_scalLinSrcSBLMLabel, matlIndex, patch);
  new_dw->allocate(scalarVars.scalarNonlinearSrc, 
		      d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch);
 
  // compute ith component of scalar stencil coefficients
  // inputs : scalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
  d_discretize->calculateScalarCoeff(pc, patch, old_dw, new_dw, 
				     delta_t, index, cellinfo, 
				     &scalarVars);

  // Calculate scalar source terms
  // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
  // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
  d_source->calculateScalarSource(pc, patch, old_dw, new_dw, 
				  delta_t, index, cellinfo, 
				  &scalarVars );

  // Calculate the scalar boundary conditions
  // inputs : scalarSP, scalCoefSBLM
  // outputs: scalCoefSBLM
  d_boundaryCondition->scalarBC(pc, patch, old_dw, new_dw, index, cellinfo, 
				  &scalarVars);

  // similar to mascal
  // inputs :
  // outputs:
  d_source->modifyScalarMassSource(pc, patch, old_dw,
				   new_dw, delta_t, index, &scalarVars);

  // Calculate the scalar diagonal terms
  // inputs : scalCoefSBLM, scalLinSrcSBLM
  // outputs: scalCoefSBLM
  d_discretize->calculateScalarDiagonal(pc, patch, old_dw,
				     new_dw, index, &scalarVars);
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarVars.scalarCoeff[ii], 
		   d_lab->d_scalCoefSBLMLabel, ii, patch);
  }
  new_dw->put(scalarVars.scalarNonlinearSrc, 
		 d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch);

}

//****************************************************************************
// Actual scalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ScalarSolver::scalarLinearSolve(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t,
				int index)
{
  ArchesVariables scalarVars;
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;
  //DataWarehouseP old_dw = new_dw->getTop();
  // Get the PerPatch CellInformation data
  PerPatch<CellInformationP> cellInfoP;
  // get old_dw from getTop function
  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP.get().get_rep();
  old_dw->get(scalarVars.old_density, d_lab->d_densityCPLabel, 
	      matlIndex, patch, Ghost::None, zeroGhostCells);
  // for explicit calculation
#if 0
  new_dw->get(scalarVars.old_scalar, d_lab->d_scalarCPBCLabel, 
	      index, patch, Ghost::AroundCells, numGhostCells);
#endif
  new_dw->get(scalarVars.scalar, d_lab->d_scalarCPBCLabel, 
	      index, patch, Ghost::AroundCells, numGhostCells);
  scalarVars.old_scalar.allocate(scalarVars.scalar.getLowIndex(),
				    scalarVars.scalar.getHighIndex());
  scalarVars.old_scalar.copy(scalarVars.scalar);

  for (int ii = 0; ii < nofStencils; ii++)
    new_dw->get(scalarVars.scalarCoeff[ii], d_lab->d_scalCoefSBLMLabel, 
		   ii, patch, Ghost::None, zeroGhostCells);
  new_dw->get(scalarVars.scalarNonlinearSrc, d_lab->d_scalNonLinSrcSBLMLabel,
		 matlIndex, patch, Ghost::None, zeroGhostCells);
  new_dw->allocate(scalarVars.residualScalar, d_lab->d_scalarRes,
			  matlIndex, patch);

#if 0  
  // compute eqn residual
  d_linearSolver->computeScalarResidual(pc, patch, new_dw, new_dw, index, 
					&scalarVars);
  new_dw->put(sum_vartype(scalarVars.residScalar), d_lab->d_scalarResidLabel);
  new_dw->put(sum_vartype(scalarVars.truncScalar), d_lab->d_scalarTruncLabel);
#endif
  // apply underelax to eqn
  d_linearSolver->computeScalarUnderrelax(pc, patch, new_dw, new_dw, index, 
					  &scalarVars);
#if 0
  new_dw->allocate(scalarVars.old_scalar, d_lab->d_old_scalarGuess,
		   index, patch);
  scalarVars.old_scalar = scalarVars.scalar;
#endif
  // make it a separate task later
  d_linearSolver->scalarLisolve(pc, patch, new_dw, new_dw, index, delta_t, 
				&scalarVars, cellinfo, d_lab);
  // put back the results
  new_dw->put(scalarVars.scalar, d_lab->d_scalarSPLabel, 
	      index, patch);
}

//
// $Log$
// Revision 1.27.2.1  2000/10/26 10:05:16  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.30  2000/10/14 17:11:05  sparker
// Changed PerPatch<CellInformation*> to PerPatch<CellInformationP>
// to get rid of memory leak
//
// Revision 1.29  2000/10/10 19:30:57  rawat
// added scalarsolver
//
// Revision 1.28  2000/10/09 18:48:12  sparker
// Remove matrix_dw
// Fixed ghost cell specification
//
// Revision 1.27  2000/09/20 18:05:34  sparker
// Adding support for Petsc and per-processor tasks
//
// Revision 1.26  2000/09/14 17:04:54  rawat
// converting arches to multipatch
//
// Revision 1.25  2000/09/12 15:47:24  sparker
// Use petsc solver if available
//
// Revision 1.24  2000/09/07 23:07:17  rawat
// fixed some bugs in bc and added pressure solver using petsc
//
// Revision 1.23  2000/08/15 00:23:33  rawat
// added explicit solve for momentum and scalar eqns
//
// Revision 1.22  2000/08/14 02:34:57  bbanerje
// Removed a small buf in sum_vars for residual in MomentumSolver and ScalarSolver
//
// Revision 1.21  2000/08/12 23:53:19  bbanerje
// Added Linegs part to the solver.
//
// Revision 1.20  2000/08/10 00:56:33  rawat
// added pressure bc for scalar and changed discretization option for velocity
//
// Revision 1.19  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.18  2000/08/01 06:18:38  bbanerje
// Made ScalarSolver similar to PressureSolver and MomentumSolver.
//
// Revision 1.17  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.15  2000/07/03 05:30:16  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.14  2000/06/29 23:37:11  bbanerje
// Changed FCVarsto SFC[X,Y,Z]Vars and added relevant getIndex() calls.
//
// Revision 1.13  2000/06/28 08:14:53  bbanerje
// Changed the init routines a bit.
//
// Revision 1.12  2000/06/22 23:06:37  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.11  2000/06/21 07:51:01  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.10  2000/06/21 06:12:12  bbanerje
// Added missing VarLabel* mallocs .
//
// Revision 1.9  2000/06/18 01:20:17  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.8  2000/06/17 07:06:26  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/06/16 04:25:40  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.6  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.5  2000/06/12 21:30:00  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.4  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.3  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

