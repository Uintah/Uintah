//----- EnthalpySolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/EnthalpySolver.h>
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
EnthalpySolver::EnthalpySolver(const ArchesLabel* label,
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
EnthalpySolver::~EnthalpySolver()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
EnthalpySolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("EnthalpySolver");
  db->require("radiation",d_radiationCalc);
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
EnthalpySolver::solve(SchedulerP& sched,
		    const PatchSet* patches,
		    const MaterialSet* matls)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
   //DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);

  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_enthalpyLinearSolve(sched, patches, matls);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
EnthalpySolver::sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::BuildCoeff",
			  this,
			  &EnthalpySolver::buildLinearMatrix);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
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

      // added one more argument of index to specify enthalpy component
  tsk->computes(d_lab->d_enthCoefSBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthNonLinSrcSBLMLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Schedule linear solve of enthalpy
//****************************************************************************
void
EnthalpySolver::sched_enthalpyLinearSolve(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::enthalpyLinearSolve",
			  this,
			  &EnthalpySolver::enthalpyLinearSolve);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcSBLMLabel, 
		Ghost::None, zeroGhostCells);
  tsk->computes(d_lab->d_enthalpySPLabel);
  
  sched->addTask(tsk, patches, matls);
}
      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void EnthalpySolver::buildLinearMatrix(const ProcessorGroup* pc,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->get(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(enthalpyVars.old_enthalpy, d_lab->d_enthalpyINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(enthalpyVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit get old values
    new_dw->get(enthalpyVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocate(enthalpyVars.scalarCoeff[ii], 
		       d_lab->d_enthCoefSBLMLabel, ii, patch);
      new_dw->allocate(enthalpyVars.scalarConvectCoeff[ii],
		       d_lab->d_enthConvCoefSBLMLabel, ii, patch);
    }
    new_dw->allocate(enthalpyVars.scalarLinearSrc, 
		     d_lab->d_enthLinSrcSBLMLabel, matlIndex, patch);
    new_dw->allocate(enthalpyVars.scalarNonlinearSrc, 
		     d_lab->d_enthNonLinSrcSBLMLabel, matlIndex, patch);
 
  // compute ith component of enthalpy stencil coefficients
  // inputs : enthalpySP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
    // coeffs calculation is same as that of scalar
    int index = 0;
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &enthalpyVars);

    // Calculate enthalpy source terms
    // inputs : [u,v,w]VelocityMS, enthalpySP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateEnthalpySource(pc, patch,
				    delta_t, cellinfo, 
				    &enthalpyVars );

    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->enthalpyBC(pc, patch, cellinfo, 
				  &enthalpyVars);
  // apply multimaterial intrusion wallbc
#if 0
    if (d_MAlab)
      d_boundaryCondition->mmenthalpyWallBC(pc, patch, cellinfo,
					  &enthalpyVars);
#endif
    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t, &enthalpyVars);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefSBLMLabel, ii, patch);
    }
    new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcSBLMLabel, matlIndex, patch);

  }
}
//****************************************************************************
// Actual enthalpy solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
EnthalpySolver::enthalpyLinearSolve(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->get(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit calculation
    new_dw->get(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefSBLMLabel, 
		  ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcSBLMLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(enthalpyVars.residualEnthalpy, d_lab->d_enthalpyRes,
		     matlIndex, patch);

#if 0  
  // compute eqn residual
    d_linearSolver->computeEnthalpyResidual(pc, patch, new_dw, new_dw, index, 
					  &enthalpyVars);
    new_dw->put(sum_vartype(enthalpyVars.residEnthalpy), d_lab->d_enthalpyResidLabel);
    new_dw->put(sum_vartype(enthalpyVars.truncEnthalpy), d_lab->d_enthalpyTruncLabel);
#endif
  // apply underelax to eqn
    d_linearSolver->computeEnthalpyUnderrelax(pc, patch,  
					    &enthalpyVars);
    // make it a separate task later
    d_linearSolver->enthalpyLisolve(pc, patch, delta_t, 
				  &enthalpyVars, cellinfo, d_lab);
  // put back the results
    new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch);
  }
}

//****************************************************************************
// Schedule solve of linearized enthalpy equation
//****************************************************************************
void 
EnthalpySolver::solvePred(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls)
{
  //computes stencil coefficients and source terms
  // requires : enthalpyIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixPred(sched, patches, matls);
  
  // Schedule the enthalpy solve
  // require : enthalpyIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, enthalpySP
  //d_linearSolver->sched_enthalpySolve(level, sched, new_dw, matrix_dw);
  sched_enthalpyLinearSolvePred(sched, patches, matls);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
EnthalpySolver::sched_buildLinearMatrixPred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::BuildCoeffPred",
			  this,
			  &EnthalpySolver::buildLinearMatrixPred);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
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
  if (d_radiationCalc) {
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::OldDW, d_lab->d_absorpINLabel,
		  Ghost::None, zeroGhostCells);
  }      // added one more argument of index to specify enthalpy component
  tsk->computes(d_lab->d_enthCoefPredLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthNonLinSrcPredLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void EnthalpySolver::buildLinearMatrixPred(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->get(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    //    new_dw->get(enthalpyVars.old_enthalpy, d_lab->d_enthalpyINLabel, 
    //		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(enthalpyVars.old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(enthalpyVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit get old values
    new_dw->get(enthalpyVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocate(enthalpyVars.scalarCoeff[ii], 
		       d_lab->d_enthCoefPredLabel, ii, patch);
      new_dw->allocate(enthalpyVars.scalarConvectCoeff[ii],
		       d_lab->d_enthConvCoefPredLabel, ii, patch);
    }
    new_dw->allocate(enthalpyVars.scalarLinearSrc, 
		     d_lab->d_enthLinSrcPredLabel, matlIndex, patch);
    new_dw->allocate(enthalpyVars.scalarNonlinearSrc, 
		     d_lab->d_enthNonLinSrcPredLabel, matlIndex, patch);
    enthalpyVars.scalarNonlinearSrc.initialize(0.0);
    if (d_radiationCalc) {
      enthalpyVars.qfluxe.allocate(patch->getCellLowIndex(),
				   patch->getCellHighIndex());
      enthalpyVars.qfluxe.initialize(0.0);
      enthalpyVars.qfluxw.allocate(patch->getCellLowIndex(),
				   patch->getCellHighIndex());
      enthalpyVars.qfluxw.initialize(0.0);
      enthalpyVars.qfluxn.allocate(patch->getCellLowIndex(),
				   patch->getCellHighIndex());
      enthalpyVars.qfluxn.initialize(0.0);
      enthalpyVars.qfluxs.allocate(patch->getCellLowIndex(),
				   patch->getCellHighIndex());
      enthalpyVars.qfluxs.initialize(0.0);
      enthalpyVars.qfluxt.allocate(patch->getCellLowIndex(),
				   patch->getCellHighIndex());
      enthalpyVars.qfluxt.initialize(0.0);
      enthalpyVars.qfluxb.allocate(patch->getCellLowIndex(),
				   patch->getCellHighIndex());
      enthalpyVars.qfluxb.initialize(0.0);
      old_dw->get(enthalpyVars.temperature, d_lab->d_tempINLabel, 
		  matlIndex, patch, Ghost::AroundCells, numGhostCells);
      old_dw->get(enthalpyVars.absorption, d_lab->d_absorpINLabel, 
		  matlIndex, patch, Ghost::None, zeroGhostCells);

    }
  // compute ith component of enthalpy stencil coefficients
  // inputs : enthalpySP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
    int index = 0;
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &enthalpyVars);

    // Calculate enthalpy source terms
    // inputs : [u,v,w]VelocityMS, enthalpySP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateEnthalpySource(pc, patch,
				    delta_t, cellinfo, 
				    &enthalpyVars );

    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->enthalpyBC(pc, patch,  cellinfo, 
				  &enthalpyVars);
  // apply multimaterial intrusion wallbc
#if 0
    if (d_MAlab)
      d_boundaryCondition->mmenthalpyWallBC(pc, patch, cellinfo,
					  &enthalpyVars);
#endif
    if (d_radiationCalc) {
#ifdef opticallythick
      d_source->computeEnthalpyRadFluxes(pc, patch,
					 cellinfo, 
					 &enthalpyVars );
      d_boundaryCondition->enthalpyRadWallBC(pc, patch,
					     cellinfo,
					     &enthalpyVars);
#if 0
      if (d_MAlab)
	d_boundaryCondition->mmenthalpyRadWallBC(pc, patch,
						 cellinfo,
						 &enthalpyVars);
#endif
      d_source->computeEnthalpyRadSrc(pc, patch,
				      cellinfo, &enthalpyVars);
#else
      d_source->computeEnthalpyRadThinSrc(pc, patch,
					  cellinfo, &enthalpyVars);
#endif
    }
      
    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t,  &enthalpyVars);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefPredLabel, ii, patch);
    }
    new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcPredLabel, matlIndex, patch);

  }
}


//****************************************************************************
// Schedule linear solve of enthalpy
//****************************************************************************
void
EnthalpySolver::sched_enthalpyLinearSolvePred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::enthalpyLinearSolvePred",
			  this,
			  &EnthalpySolver::enthalpyLinearSolvePred);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthCoefPredLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcPredLabel, 
		Ghost::None, zeroGhostCells);
#ifdef correctorstep
  tsk->computes(d_lab->d_enthalpyPredLabel);
#else
  tsk->computes(d_lab->d_enthalpySPLabel);
#endif
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual enthalpy solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
EnthalpySolver::enthalpyLinearSolvePred(const ProcessorGroup* pc,
                                const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->get(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit calculation
    new_dw->get(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefPredLabel, 
		  ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcPredLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(enthalpyVars.residualEnthalpy, d_lab->d_enthalpyRes,
		     matlIndex, patch);

  // apply underelax to eqn
    d_linearSolver->computeEnthalpyUnderrelax(pc, patch,
					    &enthalpyVars);
    // make it a separate task later
    d_linearSolver->enthalpyLisolve(pc, patch, delta_t, 
    &enthalpyVars, cellinfo, d_lab);
				  // put back the results
#if 0
    cerr << "print enthalpy solve after predict" << endl;
    enthalpyVars.enthalpy.print(cerr);
#endif
#ifdef correctorstep
    new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
                matlIndex, patch);
#else
    new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
                matlIndex, patch);
#endif
  }
}

//****************************************************************************
// Schedule solve of linearized enthalpy equation, corrector step
//****************************************************************************
void 
EnthalpySolver::solveCorr(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls)
{
  //computes stencil coefficients and source terms
  // requires : enthalpyIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixCorr(sched, patches, matls);
  
  // Schedule the enthalpy solve
  // require : enthalpyIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, enthalpySP
  //d_linearSolver->sched_enthalpySolve(level, sched, new_dw, matrix_dw, index);
  sched_enthalpyLinearSolveCorr(sched, patches, matls);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
EnthalpySolver::sched_buildLinearMatrixCorr(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::BuildCoeffCorr",
			  this,
			  &EnthalpySolver::buildLinearMatrixCorr);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel,
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

      // added one more argument of index to specify enthalpy component
  tsk->computes(d_lab->d_enthCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthNonLinSrcCorrLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void EnthalpySolver::buildLinearMatrixCorr(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->get(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    // ***warning* 21st July changed from IN to Pred
    new_dw->get(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(enthalpyVars.old_enthalpy, d_lab->d_enthalpyINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(enthalpyVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit get old values
    new_dw->get(enthalpyVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(enthalpyVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocate(enthalpyVars.scalarCoeff[ii], 
		       d_lab->d_enthCoefCorrLabel, ii, patch);
      new_dw->allocate(enthalpyVars.scalarConvectCoeff[ii],
		       d_lab->d_enthConvCoefCorrLabel, ii, patch);
    }
    new_dw->allocate(enthalpyVars.scalarLinearSrc, 
		     d_lab->d_enthLinSrcCorrLabel, matlIndex, patch);
    new_dw->allocate(enthalpyVars.scalarNonlinearSrc, 
		     d_lab->d_enthNonLinSrcCorrLabel, matlIndex, patch);
 
  // compute ith component of enthalpy stencil coefficients
  // inputs : enthalpySP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
    int index = 0;
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &enthalpyVars);

    // Calculate enthalpy source terms
    // inputs : [u,v,w]VelocityMS, enthalpySP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateEnthalpySource(pc, patch,
				    delta_t, cellinfo, 
				    &enthalpyVars );

    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->enthalpyBC(pc, patch,  cellinfo, 
				  &enthalpyVars);
  // apply multimaterial intrusion wallbc
#if 0
    if (d_MAlab)
      d_boundaryCondition->mmenthalpyWallBC(pc, patch, cellinfo,
					  &enthalpyVars);

#endif
    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t,  &enthalpyVars);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefCorrLabel, ii, patch);
    }
    new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcCorrLabel, matlIndex, patch);

  }
}


//****************************************************************************
// Schedule linear solve of enthalpy
//****************************************************************************
void
EnthalpySolver::sched_enthalpyLinearSolveCorr(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::enthalpyLinearSolveCorr",
			  this,
			  &EnthalpySolver::enthalpyLinearSolveCorr);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  //***warning changed in to pred
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthCoefCorrLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcCorrLabel, 
		Ghost::None, zeroGhostCells);
  tsk->computes(d_lab->d_enthalpySPLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual enthalpy solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
EnthalpySolver::enthalpyLinearSolveCorr(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->get(enthalpyVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit calculation
    new_dw->get(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefCorrLabel, 
		  ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcCorrLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(enthalpyVars.residualEnthalpy, d_lab->d_enthalpyRes,
		     matlIndex, patch);

  // apply underelax to eqn
    d_linearSolver->computeEnthalpyUnderrelax(pc, patch,
					    &enthalpyVars);
    // make it a separate task later
    d_linearSolver->enthalpyLisolve(pc, patch, delta_t, 
				  &enthalpyVars, cellinfo, d_lab);
  // put back the results
    new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch);
  }
}
