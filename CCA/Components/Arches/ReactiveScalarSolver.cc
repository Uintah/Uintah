//----- ReactiveScalarSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ReactiveScalarSolver.h>
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
ReactiveScalarSolver::ReactiveScalarSolver(const ArchesLabel* label,
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
ReactiveScalarSolver::~ReactiveScalarSolver()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ReactiveScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("ReactiveScalarSolver");
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
// Schedule solve of linearized reactscalar equation
//****************************************************************************
void 
ReactiveScalarSolver::solvePred(SchedulerP& sched,
			const PatchSet* patches,
			const MaterialSet* matls,
			int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : reactscalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrixPred(sched, patches, matls, index);
  
  // Schedule the scalar solve
  // require : scalarIN, reactscalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, reactscalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_reactscalarLinearSolvePred(sched, patches, matls, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrixPred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::BuildCoeffPred",
			  this,
			  &ReactiveScalarSolver::buildLinearMatrixPred,
			  index);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarINLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel,
		Ghost::AroundCells, numGhostCells);

  tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel,
		Ghost::None, zeroGhostCells);

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
  tsk->computes(d_lab->d_reactscalCoefPredLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_reactscalNonLinSrcPredLabel);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ReactiveScalarSolver::buildLinearMatrixPred(const ProcessorGroup* pc,
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
    ArchesVariables reactscalarVars;
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
    new_dw->get(reactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(reactscalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    //    new_dw->get(scalarVars.old_scalar, d_lab->d_scalarINLabel, 
    //		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(reactscalarVars.old_scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(reactscalarVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(reactscalarVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(reactscalarVars.scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // computes reaction scalar source term in properties
    old_dw->get(reactscalarVars.reactscalarSRC, d_lab->d_reactscalarSRCINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);


    // for explicit get old values
    new_dw->get(reactscalarVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(reactscalarVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(reactscalarVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

  // allocate matrix coeffs
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocate(reactscalarVars.scalarCoeff[ii], 
		       d_lab->d_reactscalCoefPredLabel, ii, patch);
      new_dw->allocate(reactscalarVars.scalarConvectCoeff[ii],
		       d_lab->d_reactscalConvCoefPredLabel, ii, patch);
    }
    new_dw->allocate(reactscalarVars.scalarLinearSrc, 
		     d_lab->d_reactscalLinSrcPredLabel, matlIndex, patch);
    new_dw->allocate(reactscalarVars.scalarNonlinearSrc, 
		     d_lab->d_reactscalNonLinSrcPredLabel, matlIndex, patch);
 
  // compute ith component of reactscalar stencil coefficients
  // inputs : reactscalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &reactscalarVars);

    // Calculate reactscalar source terms
    // inputs : [u,v,w]VelocityMS, reactscalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars );
    d_source->addReactiveScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars );
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, reactscalCoefSBLM
    // outputs: reactscalCoefSBLM
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &reactscalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &reactscalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index, &reactscalarVars);
    
    // Calculate the reactscalar diagonal terms
    // inputs : reactscalCoefSBLM, scalLinSrcSBLM
    // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &reactscalarVars);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(reactscalarVars.scalarCoeff[ii], 
		  d_lab->d_reactscalCoefPredLabel, ii, patch);
    }
    new_dw->put(reactscalarVars.scalarNonlinearSrc, 
		d_lab->d_reactscalNonLinSrcPredLabel, matlIndex, patch);

  }
}


//****************************************************************************
// Schedule linear solve of reactscalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolvePred(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  int index)
{
  Task* tsk = scinew Task("ReactiveScalarSolver::reactscalarLinearSolvePred",
			  this,
			  &ReactiveScalarSolver::reactscalarLinearSolvePred,
			  index);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

  // coefficient for the variable for which solve is invoked
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		Ghost::AroundCells, numGhostCells+1);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_reactscalCoefPredLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_reactscalNonLinSrcPredLabel, 
		Ghost::None, zeroGhostCells);
#ifdef correctorstep
  tsk->computes(d_lab->d_reactscalarPredLabel);
#else
  tsk->computes(d_lab->d_reactscalarSPLabel);
#endif
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual reactscalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ReactiveScalarSolver::reactscalarLinearSolvePred(const ProcessorGroup* pc,
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
    ArchesVariables reactscalarVars;
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
    new_dw->get(reactscalarVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // for explicit calculation
    new_dw->get(reactscalarVars.scalar, d_lab->d_reactscalarOUTBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    reactscalarVars.old_scalar.allocate(reactscalarVars.scalar.getLowIndex(),
				   reactscalarVars.scalar.getHighIndex());
    reactscalarVars.old_scalar.copy(reactscalarVars.scalar);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(reactscalarVars.scalarCoeff[ii], d_lab->d_reactscalCoefPredLabel, 
		  ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(reactscalarVars.scalarNonlinearSrc, d_lab->d_reactscalNonLinSrcPredLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(reactscalarVars.residualReactivescalar, d_lab->d_reactscalarRes,
		     matlIndex, patch);

  // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &reactscalarVars);
    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
    &reactscalarVars, cellinfo, d_lab);
				  // put back the results
#if 0
    cerr << "print reactscalar solve after predict" << endl;
    reactscalarVars.reactscalar.print(cerr);
#endif
#ifdef correctorstep
    new_dw->put(reactscalarVars.scalar, d_lab->d_reactscalarPredLabel, 
                matlIndex, patch);
#else
    new_dw->put(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
                matlIndex, patch);
#endif
  }
}

