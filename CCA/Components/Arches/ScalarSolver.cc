//----- ScalarSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ScalarSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>

using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
ScalarSolver::ScalarSolver(const ArchesLabel* label,
			   const MPMArchesLabel* MAlb,
			   TurbulenceModel* turb_model,
			   BoundaryCondition* bndry_cond,
			   PhysicalConstants* physConst) :
                                 d_lab(label), d_MAlab(MAlb),
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
ScalarSolver::solve(SchedulerP& sched,
		    const PatchSet* patches,
		    const MaterialSet* matls,
		    double /*time*/, double delta_t, int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
   //DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);

  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, delta_t, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_scalarLinearSolve(sched, patches, matls, delta_t, index);
#if 0
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
#endif
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls,
				      double delta_t, int index)
{
  Task* tsk = scinew Task("ScalarSolver::BuildCoeff",
			  this,
			  &ScalarSolver::buildLinearMatrix,
			  delta_t, index);

  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarCPBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityCPBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityCPBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityCPBCLabel,
		Ghost::AroundCells, numGhostCells);

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl);
  tsk->computes(d_lab->d_scalNonLinSrcSBLMLabel);

  sched->addTask(tsk, patches, matls);
#if 0
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
      tsk->requires(new_dw, d_lab->d_scalarINLabel, index, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(new_dw, d_lab->d_scalarCPBCLabel, index, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
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
#endif
}

//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolve(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls,
				      double delta_t,
				      int index)
{
  Task* tsk = scinew Task("ScalarSolver::scalarLinearSolve",
			  this,
			  &ScalarSolver::scalarLinearSolve, delta_t, index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_scalarCPBCLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalCoefSBLMLabel, 
		d_lab->d_stencilMatl, Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcSBLMLabel, 
		Ghost::None, zeroGhostCells);
  tsk->computes(d_lab->d_scalarSPLabel);
  
  sched->addTask(tsk, patches, matls);
#if 0
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
      tsk->requires(new_dw, d_lab->d_densityINLabel, index, patch,
		    Ghost::AroundCells, numGhostCells+1);
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
#endif
}
      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw,
				     double delta_t, int index)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO(index)
    old_dw->get(scalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(scalarVars.old_scalar, d_lab->d_scalarINLabel, 
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
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
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
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &scalarVars);

    // Calculate scalar source terms
    // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &scalarVars );

#ifdef multimaterialform
    if (d_mmInterface) {
      MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
      d_mmSGSModel->calculateScalarSource(patch, index, cellinfo,
					  mmVars, &scalarVars);
    }
#endif
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &scalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &scalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index, &scalarVars);
    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &scalarVars);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(scalarVars.scalarCoeff[ii], 
		  d_lab->d_scalCoefSBLMLabel, ii, patch);
    }
    new_dw->put(scalarVars.scalarNonlinearSrc, 
		d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch);

  }
}
//****************************************************************************
// Actual scalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ScalarSolver::scalarLinearSolve(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw,
				double delta_t,
				int index)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    //DataWarehouseP old_dw = new_dw->getTop();
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // get old_dw from getTop function
    // checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->get(scalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit calculation
    new_dw->get(scalarVars.scalar, d_lab->d_scalarCPBCLabel, 
		index, patch, Ghost::AroundCells, numGhostCells);
    scalarVars.old_scalar.allocate(scalarVars.scalar.getLowIndex(),
				   scalarVars.scalar.getHighIndex());
    scalarVars.old_scalar.copy(scalarVars.scalar);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
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
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &scalarVars);
#if 0
    new_dw->allocate(scalarVars.old_scalar, d_lab->d_old_scalarGuess,
		     index, patch);
    scalarVars.old_scalar = scalarVars.scalar;
#endif
    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
				  &scalarVars, cellinfo, d_lab);
  // put back the results
    new_dw->put(scalarVars.scalar, d_lab->d_scalarSPLabel, 
		index, patch);
  }
}
