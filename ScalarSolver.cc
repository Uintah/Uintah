//----- ScalarSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ScalarSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

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
		    int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
   //DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);

  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_scalarLinearSolve(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls,
				      int index)
{
  Task* tsk = scinew Task("ScalarSolver::BuildCoeff",
			  this,
			  &ScalarSolver::buildLinearMatrix,
			  index);

  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarOUTBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::AroundCells, numGhostCells);

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_scalNonLinSrcSBLMLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolve(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls,
				      int index)
{
  Task* tsk = scinew Task("ScalarSolver::scalarLinearSolve",
			  this,
			  &ScalarSolver::scalarLinearSolve, index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_scalarOUTBCLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcSBLMLabel, 
		Ghost::None, zeroGhostCells);
  tsk->computes(d_lab->d_scalarSPLabel);
  
  sched->addTask(tsk, patches, matls);
}
      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw,
				     int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

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
    new_dw->get(scalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(scalarVars.old_scalar, d_lab->d_scalarINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(scalarVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.scalar, d_lab->d_scalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit get old values
    new_dw->get(scalarVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
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
				int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
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
    new_dw->get(scalarVars.scalar, d_lab->d_scalarOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
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
    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
				  &scalarVars, cellinfo, d_lab);
  // put back the results
    new_dw->put(scalarVars.scalar, d_lab->d_scalarSPLabel, 
		matlIndex, patch);
  }
}

//****************************************************************************
// Schedule solve of linearized scalar equation
//****************************************************************************
void 
ScalarSolver::solvePred(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls,
			int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixPred(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_scalarLinearSolvePred(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrixPred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ScalarSolver::BuildCoeffPred",
			  this,
			  &ScalarSolver::buildLinearMatrixPred,
			  index);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarOUTBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::AroundCells, numGhostCells);

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_scalCoefPredLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_scalNonLinSrcPredLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrixPred(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw,
					 int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
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
    new_dw->get(scalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    //    new_dw->get(scalarVars.old_scalar, d_lab->d_scalarINLabel, 
    //		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(scalarVars.old_scalar, d_lab->d_scalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(scalarVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.scalar, d_lab->d_scalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit get old values
    new_dw->get(scalarVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocate(scalarVars.scalarCoeff[ii], 
		       d_lab->d_scalCoefPredLabel, ii, patch);
      new_dw->allocate(scalarVars.scalarConvectCoeff[ii],
		       d_lab->d_scalConvCoefPredLabel, ii, patch);
    }
    new_dw->allocate(scalarVars.scalarLinearSrc, 
		     d_lab->d_scalLinSrcPredLabel, matlIndex, patch);
    new_dw->allocate(scalarVars.scalarNonlinearSrc, 
		     d_lab->d_scalNonLinSrcPredLabel, matlIndex, patch);
 
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
		  d_lab->d_scalCoefPredLabel, ii, patch);
    }
    new_dw->put(scalarVars.scalarNonlinearSrc, 
		d_lab->d_scalNonLinSrcPredLabel, matlIndex, patch);

  }
}


//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolvePred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ScalarSolver::scalarLinearSolvePred",
			  this,
			  &ScalarSolver::scalarLinearSolvePred,
			  index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_scalarOUTBCLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalCoefPredLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcPredLabel, 
		Ghost::None, zeroGhostCells);
#ifdef correctorstep
  tsk->computes(d_lab->d_scalarPredLabel);
#else
  tsk->computes(d_lab->d_scalarSPLabel);
#endif
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual scalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ScalarSolver::scalarLinearSolvePred(const ProcessorGroup* pc,
                                const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw,
				int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
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
    new_dw->get(scalarVars.scalar, d_lab->d_scalarOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    scalarVars.old_scalar.allocate(scalarVars.scalar.getLowIndex(),
				   scalarVars.scalar.getHighIndex());
    scalarVars.old_scalar.copy(scalarVars.scalar);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(scalarVars.scalarCoeff[ii], d_lab->d_scalCoefPredLabel, 
		  ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(scalarVars.scalarNonlinearSrc, d_lab->d_scalNonLinSrcPredLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(scalarVars.residualScalar, d_lab->d_scalarRes,
		     matlIndex, patch);

  // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &scalarVars);
    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
    &scalarVars, cellinfo, d_lab);
				  // put back the results
#if 0
    cerr << "print scalar solve after predict" << endl;
    scalarVars.scalar.print(cerr);
#endif
#ifdef correctorstep
    new_dw->put(scalarVars.scalar, d_lab->d_scalarPredLabel, 
                matlIndex, patch);
#else
    new_dw->put(scalarVars.scalar, d_lab->d_scalarSPLabel, 
                matlIndex, patch);
#endif
  }
}

//****************************************************************************
// Schedule solve of linearized scalar equation, corrector step
//****************************************************************************
void 
ScalarSolver::solveCorr(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls,
			int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixCorr(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_scalarLinearSolveCorr(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrixCorr(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ScalarSolver::BuildCoeffCorr",
			  this,
			  &ScalarSolver::buildLinearMatrixCorr,
			  index);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundCells, numGhostCells);

      // added one more argument of index to specify scalar component
  tsk->computes(d_lab->d_scalCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_scalNonLinSrcCorrLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrixCorr(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw,
					 int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
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
    new_dw->get(scalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    // ***warning* 21st July changed from IN to Pred
    new_dw->get(scalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(scalarVars.old_scalar, d_lab->d_scalarINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(scalarVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.scalar, d_lab->d_scalarPredLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit get old values
    new_dw->get(scalarVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(scalarVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocate(scalarVars.scalarCoeff[ii], 
		       d_lab->d_scalCoefCorrLabel, ii, patch);
      new_dw->allocate(scalarVars.scalarConvectCoeff[ii],
		       d_lab->d_scalConvCoefCorrLabel, ii, patch);
    }
    new_dw->allocate(scalarVars.scalarLinearSrc, 
		     d_lab->d_scalLinSrcCorrLabel, matlIndex, patch);
    new_dw->allocate(scalarVars.scalarNonlinearSrc, 
		     d_lab->d_scalNonLinSrcCorrLabel, matlIndex, patch);
 
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
		  d_lab->d_scalCoefCorrLabel, ii, patch);
    }
    new_dw->put(scalarVars.scalarNonlinearSrc, 
		d_lab->d_scalNonLinSrcCorrLabel, matlIndex, patch);

  }
}


//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolveCorr(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ScalarSolver::scalarLinearSolveCorr",
			  this,
			  &ScalarSolver::scalarLinearSolveCorr,
			  index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  //***warning changed in to pred  
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalCoefCorrLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcCorrLabel, 
		Ghost::None, zeroGhostCells);
  tsk->computes(d_lab->d_scalarSPLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual scalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ScalarSolver::scalarLinearSolveCorr(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw,
				int index)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    int numGhostCells = 1;
    int zeroGhostCells = 0;
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
    // ***warning* 21st July changed from IN to Pred
    new_dw->get(scalarVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit calculation
    new_dw->get(scalarVars.scalar, d_lab->d_scalarPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    scalarVars.old_scalar.allocate(scalarVars.scalar.getLowIndex(),
				   scalarVars.scalar.getHighIndex());
    scalarVars.old_scalar.copy(scalarVars.scalar);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(scalarVars.scalarCoeff[ii], d_lab->d_scalCoefCorrLabel, 
		  ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(scalarVars.scalarNonlinearSrc, d_lab->d_scalNonLinSrcCorrLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(scalarVars.residualScalar, d_lab->d_scalarRes,
		     matlIndex, patch);

  // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &scalarVars);
    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
				  &scalarVars, cellinfo, d_lab);
  // put back the results
    new_dw->put(scalarVars.scalar, d_lab->d_scalarSPLabel, 
		matlIndex, patch);
  }
}
