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
#include <Packages/Uintah/CCA/Components/Arches/RHSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
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
  d_rhsSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
ReactiveScalarSolver::~ReactiveScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ReactiveScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("ReactiveScalarSolver");

  d_discretize = scinew Discretization();

  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"central-upwind");
    if (conv_scheme == "central-upwind") d_conv_scheme = 0;
      else if (conv_scheme == "flux_limited") d_conv_scheme = 1;
	else throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  string limiter_type;
  if (d_conv_scheme == 1) {
    db->getWithDefault("limiter_type",limiter_type,"superbee");
    if (limiter_type == "superbee") d_limiter_type = 0;
      else if (limiter_type == "vanLeer") d_limiter_type = 1;
        else if (limiter_type == "none") {
	  d_limiter_type = 2;
	  cout << "WARNING! Running central scheme for scalar," << endl;
	  cout << "which can be unstable." << endl;
	}
          else if (limiter_type == "central-upwind") d_limiter_type = 3;
            else if (limiter_type == "upwind") d_limiter_type = 4;
	      else throw InvalidValue("Flux limiter type "
		                           "not supported: " + limiter_type, __FILE__, __LINE__);
  string boundary_limiter_type;
  d_boundary_limiter_type = 3;
  if (d_limiter_type < 3) {
    db->getWithDefault("boundary_limiter_type",boundary_limiter_type,"central-upwind");
    if (boundary_limiter_type == "none") {
	  d_boundary_limiter_type = 2;
	  cout << "WARNING! Running central scheme for scalar on the boundaries," << endl;
	  cout << "which can be unstable." << endl;
    }
      else if (boundary_limiter_type == "central-upwind") d_boundary_limiter_type = 3;
        else if (boundary_limiter_type == "upwind") d_boundary_limiter_type = 4;
	  else throw InvalidValue("Flux limiter type on the boundary"
		                  "not supported: " + boundary_limiter_type, __FILE__, __LINE__);
    d_central_limiter = false;
    if (d_limiter_type < 2)
      db->getWithDefault("central_limiter",d_central_limiter,false);
  }
  }

  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);
  
  if (d_doMMS)
	  d_source->problemSetup(db);
  
  d_rhsSolver = scinew RHSSolver();

  d_turbPrNo = d_turbModel->getTurbulentPrandtlNumber();
  d_discretize->setTurbulentPrandtlNumber(d_turbPrNo);
  d_dynScalarModel = d_turbModel->getDynScalarModel();
}

//****************************************************************************
// Schedule solve of linearized reactscalar equation
//****************************************************************************
void 
ReactiveScalarSolver::solve(SchedulerP& sched,
			    const PatchSet* patches,
			    const MaterialSet* matls,
			    const TimeIntegratorLabel* timelabels)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : reactscalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels);
  
  // Schedule the scalar solve
  // require : scalarIN, reactscalCoefSBLM, scalNonLinSrcSBLM
  sched_reactscalarLinearSolve(sched, patches, matls, timelabels);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls,
				      	  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ReactiveScalarSolver::BuildCoeff" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ReactiveScalarSolver::buildLinearMatrix,
			  timelabels);


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

  if (d_dynScalarModel)
    tsk->requires(Task::NewDW, d_lab->d_reactScalarDiffusivityLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  else
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);

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
					  const TimeIntegratorLabel* timelabels)
{

  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;
  
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

    // from old_dw get PCELL, DENO, FO
    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarSPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constReactscalarVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F, U, V, W
    new_dw->get(constReactscalarVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    if (d_dynScalarModel)
      new_dw->get(constReactscalarVars.viscosity,
		  d_lab->d_reactScalarDiffusivityLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    else
      new_dw->get(constReactscalarVars.viscosity, d_lab->d_viscosityCTSLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);

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
				       delta_t, cellinfo, 
				       &reactscalarVars, &constReactscalarVars,
				       d_conv_scheme);

    // Calculate reactscalar source terms
    // inputs : [u,v,w]VelocityMS, reactscalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    d_source->addReactiveScalarSource(pc, patch,
				    delta_t, cellinfo, 
				    &reactscalarVars, &constReactscalarVars);
    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      d_discretize->calculateScalarFluxLimitedConvection
		                                  (pc, patch,  cellinfo,
				  	          &reactscalarVars,
						  &constReactscalarVars,
					          wall_celltypeval, 
						  d_limiter_type, 
						  d_boundary_limiter_type,
						  d_central_limiter); 
    }
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, reactscalCoefSBLM
    // outputs: reactscalCoefSBLM
    if (d_boundaryCondition->anyArchesPhysicalBC())
    d_boundaryCondition->scalarBC(pc, patch, 
				  &reactscalarVars, &constReactscalarVars);
  // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
				&reactscalarVars, &constReactscalarVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t,
				     &reactscalarVars, &constReactscalarVars,
				     d_conv_scheme);
    
    // Calculate the reactscalar diagonal terms
    // inputs : reactscalCoefSBLM, scalLinSrcSBLM
    // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, &reactscalarVars);

  }
}


//****************************************************************************
// Schedule linear solve of reactscalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolve(SchedulerP& sched,
						   const PatchSet* patches,
						   const MaterialSet* matls,
					const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ReactiveScalarSolver::ScalarLinearSolve" + 
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ReactiveScalarSolver::reactscalarLinearSolve,
			  timelabels);
  
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


  tsk->modifies(d_lab->d_reactscalarSPLabel);
  if (timelabels->recursion)
    tsk->computes(d_lab->d_ReactScalarClippedLabel);
  
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
					  const TimeIntegratorLabel* timelabels)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

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

    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // make it a separate task later
    d_rhsSolver->scalarLisolve(pc, patch, delta_t, 
                                  &reactscalarVars, &constReactscalarVars,
				  cellinfo);

  double reactscalar_clipped = 0.0;
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
	IntVector currCell(ii,jj,kk);
	if (reactscalarVars.scalar[currCell] > 1.0) {
	  reactscalarVars.scalar[currCell] = 1.0;
	  reactscalar_clipped = 1.0;
	  cout << "reactscalar got clipped to 1 at " << currCell << endl;
	}  
	else if (reactscalarVars.scalar[currCell] < 0.0) {
	  reactscalarVars.scalar[currCell] = 0.0;
	  reactscalar_clipped = 1.0;
	  cout << "reactscalar got clipped to 0 at " << currCell << endl;
	}
      }
    }
  }
  if (timelabels->recursion)
    new_dw->put(max_vartype(reactscalar_clipped), d_lab->d_ReactScalarClippedLabel);

// Outlet bc is done here not to change old scalar
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC()))
    d_boundaryCondition->scalarOutletPressureBC(pc, patch,
				       &reactscalarVars, &constReactscalarVars);

  }
}

