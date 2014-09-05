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
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
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
// Default constructor for ReactiveScalarSolver
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
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
ReactiveScalarSolver::~ReactiveScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
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
  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"l2up");
//  if (db->findBlock("convection_scheme")) {
//    db->require("convection_scheme",conv_scheme);
    if (conv_scheme == "l2up") d_conv_scheme = 0;
    else if (conv_scheme == "eno") d_conv_scheme = 1;
         else if (conv_scheme == "weno") d_conv_scheme = 2;
	      else throw InvalidValue("Convection scheme "
		       "not supported: " + conv_scheme);
//  } else
//    d_conv_scheme = 0;
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
ReactiveScalarSolver::solve(SchedulerP& sched,
			    const PatchSet* patches,
			    const MaterialSet* matls,
			    const TimeIntegratorLabel* timelabels,
			    int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : reactscalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels, index);
  
  // Schedule the scalar solve
  // require : scalarIN, reactscalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, reactscalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_reactscalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_reactscalarLinearSolve(sched, patches, matls, timelabels, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls,
				      	  const TimeIntegratorLabel* timelabels,
					      int index)
{
  string taskname =  "ReactiveScalarSolver::BuildCoeff" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ReactiveScalarSolver::buildLinearMatrix,
			  timelabels, index);


  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  // This task requires reactscalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) old_values_dw = parent_old_dw;
  else old_values_dw = Task::NewDW;

  tsk->requires(old_values_dw, d_lab->d_reactscalarSPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  else
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSRCINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_conv_scheme > 0) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->requires(Task::OldDW, timelabels->maxabsu_in);
      tsk->requires(Task::OldDW, timelabels->maxabsv_in);
      tsk->requires(Task::OldDW, timelabels->maxabsw_in);
    }
    else {
      tsk->requires(Task::NewDW, timelabels->maxabsu_in);
      tsk->requires(Task::NewDW, timelabels->maxabsv_in);
      tsk->requires(Task::NewDW, timelabels->maxabsw_in);
    }
  }

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_reactscalCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_reactscalDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_reactscalNonLinSrcSBLMLabel);
  }
  else {
    tsk->modifies(d_lab->d_reactscalCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_reactscalDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_reactscalNonLinSrcSBLMLabel);
  }

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ReactiveScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw,
					  const TimeIntegratorLabel* timelabels,
					     int index)
{

  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;
  
  double maxAbsU = 0.0;
  double maxAbsV = 0.0;
  double maxAbsW = 0.0;
  if (d_conv_scheme > 0) {
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      old_dw->get(mxAbsU, timelabels->maxabsu_in);
      old_dw->get(mxAbsV, timelabels->maxabsv_in);
      old_dw->get(mxAbsW, timelabels->maxabsw_in);
    }
    else {
      new_dw->get(mxAbsU, timelabels->maxabsu_in);
      new_dw->get(mxAbsV, timelabels->maxabsv_in);
      new_dw->get(mxAbsW, timelabels->maxabsw_in);
    }
    maxAbsU = mxAbsU;
    maxAbsV = mxAbsV;
    maxAbsW = mxAbsW;
  }

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO(index)
    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constReactscalarVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(constReactscalarVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constReactscalarVars.viscosity, d_lab->d_viscosityCTSLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    // for explicit get old values
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // computes reaction scalar source term in properties
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      old_dw->get(constReactscalarVars.reactscalarSRC,
                  d_lab->d_reactscalarSRCINLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    else
      new_dw->get(constReactscalarVars.reactscalarSRC,
                  d_lab->d_reactscalarSRCINLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);


  // allocate matrix coeffs
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(reactscalarVars.scalarCoeff[ii],
			     d_lab->d_reactscalCoefSBLMLabel, ii, patch);
      reactscalarVars.scalarCoeff[ii].initialize(0.0);
      new_dw->allocateAndPut(reactscalarVars.scalarDiffusionCoeff[ii],
			     d_lab->d_reactscalDiffCoefLabel, ii, patch);
      reactscalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->allocateAndPut(reactscalarVars.scalarNonlinearSrc,
			   d_lab->d_reactscalNonLinSrcSBLMLabel,
			   matlIndex, patch);
    reactscalarVars.scalarNonlinearSrc.initialize(0.0);
  }
  else {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getModifiable(reactscalarVars.scalarCoeff[ii],
			    d_lab->d_reactscalCoefSBLMLabel, ii, patch);
      reactscalarVars.scalarCoeff[ii].initialize(0.0);
      new_dw->getModifiable(reactscalarVars.scalarDiffusionCoeff[ii],
			    d_lab->d_reactscalDiffCoefLabel, ii, patch);
      reactscalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->getModifiable(reactscalarVars.scalarNonlinearSrc,
			  d_lab->d_reactscalNonLinSrcSBLMLabel,
			  matlIndex, patch);
    reactscalarVars.scalarNonlinearSrc.initialize(0.0);
  }

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateTemporary(reactscalarVars.scalarConvectCoeff[ii],  patch);
      reactscalarVars.scalarConvectCoeff[ii].initialize(0.0);
    }
    new_dw->allocateTemporary(reactscalarVars.scalarLinearSrc,  patch);
    reactscalarVars.scalarLinearSrc.initialize(0.0);
 
  // compute ith component of reactscalar stencil coefficients
  // inputs : reactscalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &reactscalarVars, &constReactscalarVars,
				       d_conv_scheme);

    // Calculate reactscalar source terms
    // inputs : [u,v,w]VelocityMS, reactscalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    d_source->addReactiveScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      if (d_conv_scheme == 2)
        d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo,
					        maxAbsU, maxAbsV, maxAbsW, 
				  	        &reactscalarVars,
						&constReactscalarVars, wall_celltypeval);
      else
        d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo,
					       maxAbsU, maxAbsV, maxAbsW, 
				  	       &reactscalarVars,
					       &constReactscalarVars, wall_celltypeval);
    }
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, reactscalCoefSBLM
    // outputs: reactscalCoefSBLM
    if (d_boundaryCondition->anyArchesPhysicalBC())
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &reactscalarVars, &constReactscalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
				&reactscalarVars, &constReactscalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index,
				     &reactscalarVars, &constReactscalarVars,
				     d_conv_scheme);
    
    // Calculate the reactscalar diagonal terms
    // inputs : reactscalCoefSBLM, scalLinSrcSBLM
    // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &reactscalarVars);

    // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &reactscalarVars,
					    &constReactscalarVars);

  }
}


//****************************************************************************
// Schedule linear solve of reactscalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolve(SchedulerP& sched,
						   const PatchSet* patches,
						   const MaterialSet* matls,
					const TimeIntegratorLabel* timelabels,
						   int index)
{
  string taskname =  "ReactiveScalarSolver::ScalarLinearSolve" + 
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ReactiveScalarSolver::reactscalarLinearSolve,
			  timelabels, index);
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (timelabels->multiple_steps)
    tsk->requires(Task::NewDW, d_lab->d_reactscalarTempLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) old_values_dw = parent_old_dw;
  else old_values_dw = Task::NewDW;

  tsk->requires(old_values_dw, d_lab->d_reactscalarSPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_reactscalCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_reactscalNonLinSrcSBLMLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->requires(Task::OldDW, timelabels->maxabsu_in);
    tsk->requires(Task::OldDW, timelabels->maxabsv_in);
    tsk->requires(Task::OldDW, timelabels->maxabsw_in);
    tsk->requires(Task::OldDW, timelabels->maxuxplus_in);
  }
  else {
    tsk->requires(Task::NewDW, timelabels->maxabsu_in);
    tsk->requires(Task::NewDW, timelabels->maxabsv_in);
    tsk->requires(Task::NewDW, timelabels->maxabsw_in);
    tsk->requires(Task::NewDW, timelabels->maxuxplus_in);
  }

  tsk->modifies(d_lab->d_reactscalarSPLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual reactscalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ReactiveScalarSolver::reactscalarLinearSolve(const ProcessorGroup* pc,
                                	     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw,
					  const TimeIntegratorLabel* timelabels,
					     int index)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  double maxAbsU;
  double maxAbsV;
  double maxAbsW;
  double maxUxplus;
  max_vartype mxAbsU;
  max_vartype mxAbsV;
  max_vartype mxAbsW;
  max_vartype mxUxp;
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    old_dw->get(mxAbsU, timelabels->maxabsu_in);
    old_dw->get(mxAbsV, timelabels->maxabsv_in);
    old_dw->get(mxAbsW, timelabels->maxabsw_in);
    old_dw->get(mxUxp, timelabels->maxuxplus_in);
  }
  else {
    new_dw->get(mxAbsU, timelabels->maxabsu_in);
    new_dw->get(mxAbsV, timelabels->maxabsv_in);
    new_dw->get(mxAbsW, timelabels->maxabsw_in);
    new_dw->get(mxUxp, timelabels->maxuxplus_in);
  }
  maxAbsU = mxAbsU;
  maxAbsV = mxAbsV;
  maxAbsW = mxAbsW;
  maxUxplus = mxUxp;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->get(constReactscalarVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.density_guess, d_lab->d_densityGuessLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    if (timelabels->multiple_steps)
      new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarTempLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else
      old_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constReactscalarVars.old_old_scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constReactscalarVars.old_old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // for explicit calculation
    new_dw->getModifiable(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel, 
                matlIndex, patch);

    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constReactscalarVars.scalarCoeff[ii],
		  d_lab->d_reactscalCoefSBLMLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constReactscalarVars.scalarNonlinearSrc,
		d_lab->d_reactscalNonLinSrcSBLMLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(reactscalarVars.residualReactivescalar,  patch);

    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // make it a separate task later
    d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
                                  &reactscalarVars, &constReactscalarVars,
				  cellinfo);

// Outlet bc is done here not to change old scalar
    if (d_boundaryCondition->getOutletBC())
    d_boundaryCondition->scalarOutletBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars, &constReactscalarVars,
					delta_t, maxUxplus, maxAbsV, maxAbsW);

    if (d_boundaryCondition->getPressureBC())
    d_boundaryCondition->scalarPressureBC(pc, patch,  index, cellinfo, 
				        &reactscalarVars,&constReactscalarVars, delta_t);

  }
}

