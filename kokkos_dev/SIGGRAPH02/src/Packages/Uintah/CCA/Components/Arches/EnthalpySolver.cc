//----- EnthalpySolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/EnthalpySolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
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
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
EnthalpySolver::~EnthalpySolver()
{
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
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


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (d_MAlab) {
    tsk->requires(Task::OldDW, d_MAlab->d_enth_mmLinSrc_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::OldDW, d_MAlab->d_enth_mmNonLinSrc_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  
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
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcSBLMLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
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
    new_dw->getCopy(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.old_enthalpy, d_lab->d_enthalpyINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->getCopy(enthalpyVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit get old values
    new_dw->getCopy(enthalpyVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefSBLMLabel, ii, patch);
      new_dw->allocateTemporary(enthalpyVars.scalarConvectCoeff[ii],  patch);
    }
    new_dw->allocateTemporary(enthalpyVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcSBLMLabel, matlIndex, patch);
 
    if (d_MAlab) {

      new_dw->getCopy(enthalpyVars.mmEnthSu, d_MAlab->d_enth_mmNonLinSrc_CCLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->getCopy(enthalpyVars.mmEnthSp, d_MAlab->d_enth_mmLinSrc_CCLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

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

    if (d_MAlab) {

      d_source->addMMEnthalpySource(pc, patch, cellinfo,
				    &enthalpyVars);
    }

    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->enthalpyBC(pc, patch, cellinfo, 
				  &enthalpyVars);
  // apply multimaterial intrusion wallbc

    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &enthalpyVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t, &enthalpyVars);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefSBLMLabel, ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcSBLMLabel, matlIndex, patch); */;

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
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit calculation
    {
    new_dw->allocateAndPut(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->copyOut(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefSBLMLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcSBLMLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(enthalpyVars.residualEnthalpy,  patch);

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
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch); */;
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


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef Scalar_ENO
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#else
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
  
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  if (d_radiationCalc) {
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::OldDW, d_lab->d_absorpINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }      // added one more argument of index to specify enthalpy component

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_MAlab->d_enth_mmLinSrc_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::NewDW, d_MAlab->d_enth_mmNonLinSrc_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

#ifdef Scalar_ENO
    tsk->requires(Task::OldDW, d_lab->d_maxAbsU_label);
    tsk->requires(Task::OldDW, d_lab->d_maxAbsV_label);
    tsk->requires(Task::OldDW, d_lab->d_maxAbsW_label);
#endif

  tsk->computes(d_lab->d_enthCoefPredLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthDiffCoefPredLabel, d_lab->d_stencilMatl,
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

#ifdef correctorstep
#ifndef Runge_Kutta_2nd
#ifndef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  delta_t /= 2.0;
#endif
#endif
#endif
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_1 = 8.0/15.0;
  delta_t *= gamma_1; 
#endif
#endif

#ifdef Scalar_ENO
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
    old_dw->get(mxAbsU, d_lab->d_maxAbsU_label);
    old_dw->get(mxAbsV, d_lab->d_maxAbsV_label);
    old_dw->get(mxAbsW, d_lab->d_maxAbsW_label);
    double maxAbsU = mxAbsU;
    double maxAbsV = mxAbsW;
    double maxAbsW = mxAbsW;
#endif

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
    
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
    new_dw->getCopy(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    //    new_dw->get(enthalpyVars.old_enthalpy, d_lab->d_enthalpyINLabel, 
    //		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->getCopy(enthalpyVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit get old values
    new_dw->getCopy(enthalpyVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefPredLabel, ii, patch);
      new_dw->allocateTemporary(enthalpyVars.scalarConvectCoeff[ii],  patch);
      new_dw->allocateAndPut(enthalpyVars.scalarDiffusionCoeff[ii], d_lab->d_enthDiffCoefPredLabel, ii, patch);
    }
    new_dw->allocateTemporary(enthalpyVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcPredLabel, matlIndex, patch);
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
      old_dw->getCopy(enthalpyVars.temperature, d_lab->d_tempINLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
      old_dw->getCopy(enthalpyVars.absorption, d_lab->d_absorpINLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    if (d_MAlab) {

      new_dw->getCopy(enthalpyVars.mmEnthSu, d_MAlab->d_enth_mmNonLinSrc_CCLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->getCopy(enthalpyVars.mmEnthSp, d_MAlab->d_enth_mmLinSrc_CCLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

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

    // Add enthalpy source terms due to multimaterial intrusions
    if (d_MAlab) {

      d_source->addMMEnthalpySource(pc, patch, cellinfo,
      				    &enthalpyVars);

    }

    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    d_boundaryCondition->enthalpyBC(pc, patch,  cellinfo, 
				  &enthalpyVars);

    // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &enthalpyVars);

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

#ifdef Scalar_ENO
    new_dw->getCopy(enthalpyVars.scalar, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef Scalar_WENO
    d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo, 
					   maxAbsU, maxAbsV, maxAbsW, 
				  	   &enthalpyVars);
#else
    d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo, 
					   maxAbsU, maxAbsV, maxAbsW, 
				  	   &enthalpyVars);
#endif
#endif
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefPredLabel, ii, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarDiffusionCoeff[ii],
		  d_lab->d_enthDiffCoefPredLabel, ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcPredLabel, matlIndex, patch); */;

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
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthCoefPredLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#ifdef correctorstep
  tsk->computes(d_lab->d_enthalpyPredLabel);
#else
  tsk->computes(d_lab->d_enthalpySPLabel);
#endif
  
#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  tsk->computes(d_lab->d_enthalpyTempLabel);
#endif
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

#ifdef correctorstep
#ifndef Runge_Kutta_2nd
#ifndef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  delta_t /= 2.0;
#endif
#endif
#endif
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_1 = 8.0/15.0;
  delta_t *= gamma_1; 
#endif
#endif

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit calculation
#ifdef correctorstep  
    new_dw->allocateAndPut(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
                matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->allocateAndPut(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
                matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif    
    new_dw->copyOut(enthalpyVars.enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefPredLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcPredLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(enthalpyVars.residualEnthalpy,  patch);

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

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
    CCVariable<double> temp_enthalpy;
    constCCVariable<double> old_density;

    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(temp_enthalpy, d_lab->d_enthalpyTempLabel, 
		matlIndex, patch);
    temp_enthalpy.initialize(0.0);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            temp_enthalpy[currCell] = old_density[currCell]*
	    (enthalpyVars.enthalpy[currCell]-
	    enthalpyVars.old_enthalpy[currCell])/gamma_1;
        }
      }
    }
    // allocateAndPut instead:
    /* new_dw->put(temp_enthalpy, d_lab->d_enthalpyTempLabel, matlIndex, patch); */;
#endif
#endif

#ifdef correctorstep
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
                matlIndex, patch); */;
#else
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
                matlIndex, patch); */;
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


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
//  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel,
//		Ghost::None, Arches::ZEROGHOSTCELLS);
//  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
//		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityIntermLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyIntermLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #else
  #ifndef Runge_Kutta_2nd
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #endif

  if (d_radiationCalc) {
  #ifdef Runge_Kutta_3d
    tsk->requires(Task::NewDW, d_lab->d_tempINIntermLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_absorpINIntermLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    tsk->requires(Task::NewDW, d_lab->d_tempINPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_absorpINPredLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  }      // added one more argument of index to specify enthalpy component

#ifdef Scalar_ENO
  #ifdef Runge_Kutta_3d
    tsk->requires(Task::NewDW, d_lab->d_maxAbsUInterm_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsVInterm_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsWInterm_label);
  #else
    tsk->requires(Task::NewDW, d_lab->d_maxAbsUPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsVPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsWPred_label);
  #endif
#endif

      // added one more argument of index to specify enthalpy component
  tsk->computes(d_lab->d_enthCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthDiffCoefCorrLabel, d_lab->d_stencilMatl,
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

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_3 = 3.0/4.0;
  delta_t *= gamma_3;
#endif
#endif
  
#ifdef Scalar_ENO
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
#ifdef Runge_Kutta_3d
    new_dw->get(mxAbsU, d_lab->d_maxAbsUInterm_label);
    new_dw->get(mxAbsV, d_lab->d_maxAbsVInterm_label);
    new_dw->get(mxAbsW, d_lab->d_maxAbsWInterm_label);
#else
    new_dw->get(mxAbsU, d_lab->d_maxAbsUPred_label);
    new_dw->get(mxAbsV, d_lab->d_maxAbsVPred_label);
    new_dw->get(mxAbsW, d_lab->d_maxAbsWPred_label);
#endif
    double maxAbsU = mxAbsU;
    double maxAbsV = mxAbsW;
    double maxAbsW = mxAbsW;
#endif

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
    
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
    new_dw->getCopy(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // from new_dw get DEN, VIS, F(index), U, V, W
  #ifdef Runge_Kutta_3d
    // old_density and old_enthalpy for Runge-Kutta are NOT from initial 
    // timestep but from previous (Interm) time step
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.old_enthalpy, d_lab->d_enthalpyIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.viscosity, d_lab->d_viscosityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.enthalpy, d_lab->d_enthalpyIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
  #ifdef Runge_Kutta_2nd
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.old_enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    // ***warning* 21st July changed from IN to Pred
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif

    new_dw->getCopy(enthalpyVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
    // for explicit get old values
  #ifdef Runge_Kutta_3d
    new_dw->getCopy(enthalpyVars.uVelocity, d_lab->d_uVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.vVelocity, d_lab->d_vVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.wVelocity, d_lab->d_wVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #else
    new_dw->getCopy(enthalpyVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #endif

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefCorrLabel, ii, patch);
      new_dw->allocateTemporary(enthalpyVars.scalarConvectCoeff[ii],  patch);
      new_dw->allocateAndPut(enthalpyVars.scalarDiffusionCoeff[ii], d_lab->d_enthDiffCoefCorrLabel, ii, patch);

    }
    new_dw->allocateTemporary(enthalpyVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcCorrLabel, matlIndex, patch);
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
  #ifdef Runge_Kutta_3d
      new_dw->getCopy(enthalpyVars.temperature, d_lab->d_tempINIntermLabel, 
		      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
      new_dw->getCopy(enthalpyVars.absorption, d_lab->d_absorpINIntermLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
      new_dw->getCopy(enthalpyVars.temperature, d_lab->d_tempINPredLabel, 
		      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
      new_dw->getCopy(enthalpyVars.absorption, d_lab->d_absorpINPredLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif

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

    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &enthalpyVars);


    if (d_radiationCalc) {
      d_source->computeEnthalpyRadThinSrc(pc, patch,
					  cellinfo, &enthalpyVars);
    }

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t,  &enthalpyVars);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);

#ifdef Scalar_ENO
  #ifdef Runge_Kutta_3d
    new_dw->getCopy(enthalpyVars.scalar, d_lab->d_enthalpyIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  #else
    new_dw->getCopy(enthalpyVars.scalar, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  #endif
#ifdef Scalar_WENO
    d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo, 
					   maxAbsU, maxAbsV, maxAbsW, 
				  	   &enthalpyVars);
#else
    d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo, 
					   maxAbsU, maxAbsV, maxAbsW, 
				  	   &enthalpyVars);
#endif
#endif
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefCorrLabel, ii, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarDiffusionCoeff[ii],
		  d_lab->d_enthDiffCoefCorrLabel, ii, patch); */;


    }
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcCorrLabel, matlIndex, patch); */;

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
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  //***warning changed in to pred
  #ifdef Runge_Kutta_3d
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyIntermLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifndef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_enthalpyTempLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  #else
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif 
  tsk->requires(Task::NewDW, d_lab->d_enthCoefCorrLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcCorrLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #ifdef Runge_Kutta_2nd
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  #ifdef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
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

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_3 = 3.0/4.0;
  double zeta_2 = -5.0/12.0;
  delta_t *= gamma_3;
#endif
#endif
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
  #ifdef Runge_Kutta_3d
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
    // for explicit calculation
    {
    new_dw->allocateAndPut(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
    new_dw->copyOut(enthalpyVars.enthalpy, d_lab->d_enthalpyIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
    new_dw->copyOut(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
    }
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefCorrLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcCorrLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(enthalpyVars.residualEnthalpy,  patch);

  // apply underelax to eqn
    d_linearSolver->computeEnthalpyUnderrelax(pc, patch,
					    &enthalpyVars);
    // make it a separate task later
    d_linearSolver->enthalpyLisolve(pc, patch, delta_t, 
				  &enthalpyVars, cellinfo, d_lab);
  #ifdef Runge_Kutta_3d
  #ifndef Runge_Kutta_3d_ssp
    constCCVariable<double> temp_enthalpy;
    constCCVariable<double> old_density;

    new_dw->get(old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(temp_enthalpy, d_lab->d_enthalpyTempLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            enthalpyVars.enthalpy[currCell] += zeta_2*temp_enthalpy[currCell]/
            old_density[currCell];
        }
      }
    }
  #endif
  #endif
  #ifdef Runge_Kutta_2nd
    constCCVariable<double> old_enthalpy;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;

    new_dw->get(old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            enthalpyVars.enthalpy[currCell] = (enthalpyVars.enthalpy[currCell]+
            old_density[currCell]/new_density[currCell]*
	    old_enthalpy[currCell])/2.0;
        }
      }
    }
  #endif
  #ifdef Runge_Kutta_3d_ssp
    constCCVariable<double> old_enthalpy;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;

    new_dw->get(old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(new_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            enthalpyVars.enthalpy[currCell] = 
	    (2.0*enthalpyVars.enthalpy[currCell]+
            old_density[currCell]/new_density[currCell]*
	    old_enthalpy[currCell])/3.0;
        }
      }
    }
  #endif

  // put back the results
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch); */;
  }
}

//****************************************************************************
// Schedule solve of linearized enthalpy equation, intermediate step
//****************************************************************************
void 
EnthalpySolver::solveInterm(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls)
{
  //computes stencil coefficients and source terms
  // requires : enthalpyIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixInterm(sched, patches, matls);
  
  // Schedule the enthalpy solve
  // require : enthalpyIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, enthalpySP
  //d_linearSolver->sched_enthalpySolve(level, sched, new_dw, matrix_dw, index);
  sched_enthalpyLinearSolveInterm(sched, patches, matls);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
EnthalpySolver::sched_buildLinearMatrixInterm(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::BuildCoeffInterm",
			  this,
			  &EnthalpySolver::buildLinearMatrixInterm);


  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
//  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
//		Ghost::None, Arches::ZEROGHOSTCELLS);
//  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
//		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (d_radiationCalc) {
    tsk->requires(Task::NewDW, d_lab->d_tempINPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_absorpINPredLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }      // added one more argument of index to specify enthalpy component

#ifdef Scalar_ENO
    tsk->requires(Task::NewDW, d_lab->d_maxAbsUPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsVPred_label);
    tsk->requires(Task::NewDW, d_lab->d_maxAbsWPred_label);
#endif

      // added one more argument of index to specify enthalpy component
  tsk->computes(d_lab->d_enthCoefIntermLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthDiffCoefIntermLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_enthNonLinSrcIntermLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void EnthalpySolver::buildLinearMatrixInterm(const ProcessorGroup* pc,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  delta_t *= gamma_2; 
#endif

#ifdef Scalar_ENO
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
    new_dw->get(mxAbsU, d_lab->d_maxAbsUPred_label);
    new_dw->get(mxAbsV, d_lab->d_maxAbsVPred_label);
    new_dw->get(mxAbsW, d_lab->d_maxAbsWPred_label);
    double maxAbsU = mxAbsU;
    double maxAbsV = mxAbsW;
    double maxAbsW = mxAbsW;
#endif
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
    
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
    new_dw->getCopy(enthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // ***warning* 21st July changed from IN to Pred
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.old_enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->getCopy(enthalpyVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit get old values
    new_dw->getCopy(enthalpyVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(enthalpyVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefIntermLabel, ii, patch);
      new_dw->allocateTemporary(enthalpyVars.scalarConvectCoeff[ii],  patch);
      new_dw->allocateAndPut(enthalpyVars.scalarDiffusionCoeff[ii], d_lab->d_enthDiffCoefIntermLabel, ii, patch);
    }
    new_dw->allocateTemporary(enthalpyVars.scalarLinearSrc,  patch);
    new_dw->allocateAndPut(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcIntermLabel, matlIndex, patch);
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
      new_dw->getCopy(enthalpyVars.temperature, d_lab->d_tempINPredLabel, 
		      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
      new_dw->getCopy(enthalpyVars.absorption, d_lab->d_absorpINPredLabel, 
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

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
      d_source->computeEnthalpyRadThinSrc(pc, patch,
					  cellinfo, &enthalpyVars);
    }

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t,  &enthalpyVars);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);

#ifdef Scalar_ENO
    new_dw->getCopy(enthalpyVars.scalar, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef Scalar_WENO
    d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo, 
					   maxAbsU, maxAbsV, maxAbsW, 
				  	   &enthalpyVars);
#else
    d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo, 
					   maxAbsU, maxAbsV, maxAbsW, 
				  	   &enthalpyVars);
#endif
#endif
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarCoeff[ii], 
		  d_lab->d_enthCoefIntermLabel, ii, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpyVars.scalarDiffusionCoeff[ii],
		  d_lab->d_enthDiffCoefIntermLabel, ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.scalarNonlinearSrc, 
		d_lab->d_enthNonLinSrcIntermLabel, matlIndex, patch); */;

  }
}


//****************************************************************************
// Schedule linear solve of enthalpy
//****************************************************************************
void
EnthalpySolver::sched_enthalpyLinearSolveInterm(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("EnthalpySolver::enthalpyLinearSolveInterm",
			  this,
			  &EnthalpySolver::enthalpyLinearSolveInterm);
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  //***warning changed in to pred
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_enthCoefIntermLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcIntermLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #ifndef Runge_Kutta_3d_ssp
  tsk->modifies(d_lab->d_enthalpyTempLabel);
  #endif
  #ifdef Runge_Kutta_3d_ssp
  tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  tsk->computes(d_lab->d_enthalpyIntermLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual enthalpy solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
EnthalpySolver::enthalpyLinearSolveInterm(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  double zeta_1 = -17.0/60.0;
  delta_t *= gamma_2; 
#endif
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
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
    new_dw->getCopy(enthalpyVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    // for explicit calculation
    {
    new_dw->allocateAndPut(enthalpyVars.enthalpy, d_lab->d_enthalpyIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->copyOut(enthalpyVars.enthalpy, d_lab->d_enthalpyPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    enthalpyVars.old_enthalpy.allocate(enthalpyVars.enthalpy.getLowIndex(),
				   enthalpyVars.enthalpy.getHighIndex());
    enthalpyVars.old_enthalpy.copy(enthalpyVars.enthalpy);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(enthalpyVars.scalarCoeff[ii], d_lab->d_enthCoefIntermLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(enthalpyVars.scalarNonlinearSrc, d_lab->d_enthNonLinSrcIntermLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(enthalpyVars.residualEnthalpy,  patch);

  // apply underelax to eqn
    d_linearSolver->computeEnthalpyUnderrelax(pc, patch,
					    &enthalpyVars);
    // make it a separate task later
    d_linearSolver->enthalpyLisolve(pc, patch, delta_t, 
				  &enthalpyVars, cellinfo, d_lab);

  #ifndef Runge_Kutta_3d_ssp
    CCVariable<double> temp_enthalpy;
    constCCVariable<double> old_density;

    new_dw->get(old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(temp_enthalpy, d_lab->d_enthalpyTempLabel,
                matlIndex, patch);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            enthalpyVars.enthalpy[currCell] += zeta_1*temp_enthalpy[currCell]/
            old_density[currCell];
            temp_enthalpy[currCell] = old_density[currCell]*
	    (enthalpyVars.enthalpy[currCell]-
	    enthalpyVars.old_enthalpy[currCell])/
            gamma_2-zeta_1*temp_enthalpy[currCell]/gamma_2;
        }
      }
    }
//    new_dw->put(temp_enthalpy, d_lab->d_enthalpyTempLabel, matlIndex, patch);
  #endif
  #ifdef Runge_Kutta_3d_ssp
    constCCVariable<double> old_enthalpy;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;

    new_dw->get(old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            enthalpyVars.enthalpy[currCell] = (enthalpyVars.enthalpy[currCell]+
            old_density[currCell]/new_density[currCell]*
	    3.0*old_enthalpy[currCell])/4.0;
        }
      }
    }
  #endif
  
  // put back the results
    // allocateAndPut instead:
    /* new_dw->put(enthalpyVars.enthalpy, d_lab->d_enthalpyIntermLabel, 
		matlIndex, patch); */;
  }
}
