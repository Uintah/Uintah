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
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadLinearSolver.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
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
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
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
			       PhysicalConstants* physConst,
			       const ProcessorGroup* myworld) :
                                 d_lab(label), d_MAlab(MAlb),
                                 d_turbModel(turb_model), 
                                 d_boundaryCondition(bndry_cond),
				 d_physicalConsts(physConst),
				 d_myworld(myworld)

{
  d_perproc_patches = 0;
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0;
  d_DORadiation = 0;
  d_radCounter = -1; //to decide how often radiation calc is done
  d_radCalcFreq = 0; 

}

//****************************************************************************
// Destructor
//****************************************************************************
EnthalpySolver::~EnthalpySolver()
{
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
  delete d_DORadiation;
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
EnthalpySolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("EnthalpySolver");
  db->require("radiation",d_radiationCalc);
  if (d_radiationCalc) {
    db->getWithDefault("radiationCalcFreq",d_radCalcFreq,3);
//    if (db->findBlock("radiationCalcFreq"))
//      db->require("radiationCalcFreq",d_radCalcFreq);
//    else
//      d_radCalcFreq = 3; // default: radiation is computed every third time step
    db->getWithDefault("discrete_ordinates",d_DORadiationCalc,true);
//    if (db->findBlock("discrete_ordinates"))
//      db->require("discrete_ordinates", d_DORadiationCalc);
//    else
//      d_DORadiationCalc = true;
    if (d_DORadiationCalc) {
      d_DORadiation = scinew DORadiationModel(d_boundaryCondition, d_myworld);
      d_DORadiation->problemSetup(db);
    }
  }
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
// Schedule solve of linearized enthalpy equation
//****************************************************************************
void 
EnthalpySolver::solve(const LevelP& level,
		      SchedulerP& sched,
		      const PatchSet* patches,
		      const MaterialSet* matls,
		      const TimeIntegratorLabel* timelabels)
{
  //computes stencil coefficients and source terms
  // requires : enthalpyIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(level, sched, patches, matls, timelabels);
  
  // Schedule the enthalpy solve
  // require : enthalpyIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, enthalpySP
  //d_linearSolver->sched_enthalpySolve(level, sched, new_dw, matrix_dw);
  sched_enthalpyLinearSolve(sched, patches, matls, timelabels);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
EnthalpySolver::sched_buildLinearMatrix(const LevelP& level,
					SchedulerP& sched,
					const PatchSet*,
					const MaterialSet* matls,
		    			const TimeIntegratorLabel* timelabels)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  //  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();


  string taskname =  "EnthalpySolver::BuildCoeff" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &EnthalpySolver::buildLinearMatrix,
			  timelabels);


  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) old_values_dw = parent_old_dw;
  else old_values_dw = Task::NewDW;

  tsk->requires(old_values_dw, d_lab->d_enthalpySPLabel,
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

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::OldDW, d_lab->d_cpINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    if (d_radiationCalc) {
      tsk->requires(Task::OldDW, d_lab->d_absorpINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      if (d_DORadiationCalc) {
        tsk->requires(Task::OldDW, d_lab->d_co2INLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_h2oINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_sootFVINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationSRCINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxEINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxWINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxNINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxSINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxTINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxBINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
      }
    } 
  }
  else {
    tsk->requires(Task::NewDW, d_lab->d_tempINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_cpINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    if (d_radiationCalc) {
      tsk->requires(Task::NewDW, d_lab->d_absorpINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      if (d_DORadiationCalc) {
        tsk->requires(Task::NewDW, d_lab->d_co2INLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_h2oINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_sootFVINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationSRCINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationFluxEINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationFluxWINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationFluxNINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationFluxSINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationFluxTINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
        tsk->requires(Task::NewDW, d_lab->d_radiationFluxBINLabel,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
      }
    } 
  }

  if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {
    tsk->requires(Task::NewDW, d_MAlab->d_enth_mmLinSrc_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::NewDW, d_MAlab->d_enth_mmNonLinSrc_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

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
    tsk->computes(d_lab->d_enthCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_enthDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_enthNonLinSrcSBLMLabel);
  }
  else {
    tsk->modifies(d_lab->d_enthCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_enthDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_enthNonLinSrcSBLMLabel);
  }

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    if (d_DORadiationCalc) {
      tsk->computes(d_lab->d_abskgINLabel);
      tsk->computes(d_lab->d_radiationSRCINLabel);
      tsk->computes(d_lab->d_radiationFluxEINLabel);
      tsk->computes(d_lab->d_radiationFluxWINLabel);
      tsk->computes(d_lab->d_radiationFluxNINLabel);
      tsk->computes(d_lab->d_radiationFluxSINLabel);
      tsk->computes(d_lab->d_radiationFluxTINLabel);
      tsk->computes(d_lab->d_radiationFluxBINLabel);
    }
  }
  else {
    if (d_DORadiationCalc) {
      tsk->modifies(d_lab->d_abskgINLabel);
      tsk->modifies(d_lab->d_radiationSRCINLabel);
      tsk->modifies(d_lab->d_radiationFluxEINLabel);
      tsk->modifies(d_lab->d_radiationFluxWINLabel);
      tsk->modifies(d_lab->d_radiationFluxNINLabel);
      tsk->modifies(d_lab->d_radiationFluxSINLabel);
      tsk->modifies(d_lab->d_radiationFluxTINLabel);
      tsk->modifies(d_lab->d_radiationFluxBINLabel);
    }
  }

  //  sched->addTask(tsk, patches, matls);
  sched->addTask(tsk, d_perproc_patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void EnthalpySolver::buildLinearMatrix(const ProcessorGroup* pc,
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

  if (d_radiationCalc) {
    if (d_DORadiationCalc){
      d_radCounter = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    }
  }

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
    if (d_radiationCalc)
      if (d_DORadiationCalc)
        d_DORadiation->d_linearSolver->matrixCreate(d_perproc_patches, patches);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
    ArchesConstVariables constEnthalpyVars;
    
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
    new_dw->get(constEnthalpyVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constEnthalpyVars.old_enthalpy, d_lab->d_enthalpySPLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constEnthalpyVars.old_density, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(constEnthalpyVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constEnthalpyVars.viscosity, d_lab->d_viscosityCTSLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constEnthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_conv_scheme > 0) {
      new_dw->get(constEnthalpyVars.scalar, d_lab->d_enthalpySPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    }
    // for explicit get old values
    new_dw->get(constEnthalpyVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constEnthalpyVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constEnthalpyVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    {
    old_dw->get(constEnthalpyVars.temperature, d_lab->d_tempINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    old_dw->get(constEnthalpyVars.cp, d_lab->d_cpINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    else {
    new_dw->get(constEnthalpyVars.temperature, d_lab->d_tempINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constEnthalpyVars.cp, d_lab->d_cpINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }

    // allocate matrix coeffs
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(enthalpyVars.scalarCoeff[ii],
			     d_lab->d_enthCoefSBLMLabel, ii, patch);
      enthalpyVars.scalarCoeff[ii].initialize(0.0);
      new_dw->allocateAndPut(enthalpyVars.scalarDiffusionCoeff[ii],
			     d_lab->d_enthDiffCoefLabel, ii, patch);
      enthalpyVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->allocateAndPut(enthalpyVars.scalarNonlinearSrc,
			   d_lab->d_enthNonLinSrcSBLMLabel, matlIndex, patch);
    enthalpyVars.scalarNonlinearSrc.initialize(0.0);
  }
  else {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getModifiable(enthalpyVars.scalarCoeff[ii],
			     d_lab->d_enthCoefSBLMLabel, ii, patch);
      enthalpyVars.scalarCoeff[ii].initialize(0.0);
      new_dw->getModifiable(enthalpyVars.scalarDiffusionCoeff[ii],
			     d_lab->d_enthDiffCoefLabel, ii, patch);
      enthalpyVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->getModifiable(enthalpyVars.scalarNonlinearSrc,
			   d_lab->d_enthNonLinSrcSBLMLabel, matlIndex, patch);
    enthalpyVars.scalarNonlinearSrc.initialize(0.0);
  }

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateTemporary(enthalpyVars.scalarConvectCoeff[ii],  patch);
      enthalpyVars.scalarConvectCoeff[ii].initialize(0.0);
    }
    new_dw->allocateTemporary(enthalpyVars.scalarLinearSrc,  patch);
    enthalpyVars.scalarLinearSrc.initialize(0.0);

    if (d_radiationCalc) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        old_dw->get(constEnthalpyVars.absorption, d_lab->d_absorpINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      else
        new_dw->get(constEnthalpyVars.absorption, d_lab->d_absorpINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      if (d_DORadiationCalc) {
        if (timelabels->integrator_step_number ==
			TimeIntegratorStepNumber::First)
        {
        old_dw->get(constEnthalpyVars.co2, d_lab->d_co2INLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
        old_dw->get(constEnthalpyVars.h2o, d_lab->d_h2oINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
        old_dw->get(constEnthalpyVars.sootFV, d_lab->d_sootFVINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

	new_dw->allocateAndPut(enthalpyVars.qfluxe,
			       d_lab->d_radiationFluxEINLabel,matlIndex, patch);
	old_dw->copyOut(enthalpyVars.qfluxe, d_lab->d_radiationFluxEINLabel,
			matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->allocateAndPut(enthalpyVars.qfluxw,
			       d_lab->d_radiationFluxWINLabel,matlIndex, patch);
	old_dw->copyOut(enthalpyVars.qfluxw, d_lab->d_radiationFluxWINLabel,
			matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->allocateAndPut(enthalpyVars.qfluxn,
			       d_lab->d_radiationFluxNINLabel,matlIndex, patch);
	old_dw->copyOut(enthalpyVars.qfluxn, d_lab->d_radiationFluxNINLabel,
			matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->allocateAndPut(enthalpyVars.qfluxs,
			       d_lab->d_radiationFluxSINLabel,matlIndex, patch);
	old_dw->copyOut(enthalpyVars.qfluxs, d_lab->d_radiationFluxSINLabel,
			matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->allocateAndPut(enthalpyVars.qfluxt,
			       d_lab->d_radiationFluxTINLabel,matlIndex, patch);
	old_dw->copyOut(enthalpyVars.qfluxt, d_lab->d_radiationFluxTINLabel,
			matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	new_dw->allocateAndPut(enthalpyVars.qfluxb,
			       d_lab->d_radiationFluxBINLabel,matlIndex, patch);
	old_dw->copyOut(enthalpyVars.qfluxb, d_lab->d_radiationFluxBINLabel,
			matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

        new_dw->allocateAndPut(enthalpyVars.src, d_lab->d_radiationSRCINLabel,
			       matlIndex, patch);
        old_dw->copyOut(enthalpyVars.src, d_lab->d_radiationSRCINLabel,
		        matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
        }
        else {
        new_dw->get(constEnthalpyVars.co2, d_lab->d_co2INLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
        new_dw->get(constEnthalpyVars.h2o, d_lab->d_h2oINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
        new_dw->get(constEnthalpyVars.sootFV, d_lab->d_sootFVINLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

	new_dw->getModifiable(enthalpyVars.qfluxe,
			       d_lab->d_radiationFluxEINLabel,matlIndex, patch);
	new_dw->getModifiable(enthalpyVars.qfluxw,
			       d_lab->d_radiationFluxWINLabel,matlIndex, patch);
	new_dw->getModifiable(enthalpyVars.qfluxn,
			       d_lab->d_radiationFluxNINLabel,matlIndex, patch);
	new_dw->getModifiable(enthalpyVars.qfluxs,
			       d_lab->d_radiationFluxSINLabel,matlIndex, patch);
	new_dw->getModifiable(enthalpyVars.qfluxt,
			       d_lab->d_radiationFluxTINLabel,matlIndex, patch);
	new_dw->getModifiable(enthalpyVars.qfluxb,
			       d_lab->d_radiationFluxBINLabel,matlIndex, patch);

        new_dw->getModifiable(enthalpyVars.src, d_lab->d_radiationSRCINLabel,
			       matlIndex, patch);
        }
      }
      else {
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
      }
    }

    if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {
  
      new_dw->get(constEnthalpyVars.mmEnthSu, d_MAlab->d_enth_mmNonLinSrc_CCLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->get(constEnthalpyVars.mmEnthSp, d_MAlab->d_enth_mmLinSrc_CCLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    // compute ith component of enthalpy stencil coefficients
    // inputs : enthalpySP, [u,v,w]VelocityMS, densityCP, viscosityCTS
    // outputs: scalCoefSBLM
    int index = 0;
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, index, cellinfo, 
				       &enthalpyVars, &constEnthalpyVars,
				       d_conv_scheme);

    // Calculate enthalpy source terms
    // inputs : [u,v,w]VelocityMS, enthalpySP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateEnthalpySource(pc, patch,
				    delta_t, cellinfo, 
				    &enthalpyVars, &constEnthalpyVars);

    // Add enthalpy source terms due to multimaterial intrusions
    if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {

      d_source->addMMEnthalpySource(pc, patch, cellinfo,
      				    &enthalpyVars, &constEnthalpyVars);

    }
    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      if (d_conv_scheme == 2)
        d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo,
					        maxAbsU, maxAbsV, maxAbsW, 
				  	        &enthalpyVars,
						&constEnthalpyVars, wall_celltypeval);
      else
        d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo,
					       maxAbsU, maxAbsV, maxAbsW, 
				  	       &enthalpyVars,
					       &constEnthalpyVars, wall_celltypeval);
    }

    if (d_radiationCalc) {
      if (d_DORadiationCalc){
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
	new_dw->allocateAndPut(enthalpyVars.ABSKG, d_lab->d_abskgINLabel,
			       matlIndex, patch);
      else
	new_dw->getModifiable(enthalpyVars.ABSKG, d_lab->d_abskgINLabel,
			       matlIndex, patch);
        enthalpyVars.ESRCG.allocate(patch->getCellLowIndex(),
				    patch->getCellHighIndex());
	
	
	enthalpyVars.ABSKG.initialize(0.0);
	enthalpyVars.ESRCG.initialize(0.0);

	if (d_radCounter%d_radCalcFreq == 0) {
	  enthalpyVars.src.initialize(0.0);
	  enthalpyVars.qfluxe.initialize(0.0);
	  enthalpyVars.qfluxw.initialize(0.0);
	  enthalpyVars.qfluxn.initialize(0.0);
	  enthalpyVars.qfluxs.initialize(0.0);
	  enthalpyVars.qfluxt.initialize(0.0);
	  enthalpyVars.qfluxb.initialize(0.0);

	  d_DORadiation->computeRadiationProps(pc, patch, cellinfo,
					     &enthalpyVars, &constEnthalpyVars);
	  d_DORadiation->boundarycondition(pc, patch, cellinfo,
					   &enthalpyVars, &constEnthalpyVars);
	  d_DORadiation->intensitysolve(pc, patch, cellinfo,
					&enthalpyVars, &constEnthalpyVars);
	}
	IntVector indexLow = patch->getCellFORTLowIndex();
	IntVector indexHigh = patch->getCellFORTHighIndex();
	for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	  for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	    for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	      IntVector currCell(colX, colY, colZ);
              double vol=cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	      enthalpyVars.scalarNonlinearSrc[currCell] += vol*enthalpyVars.src[currCell];
	    }
	  }
	}
#if 0
	d_DORadiation->d_linearSolver->destroyMatrix();
#endif

      }
      else
	d_source->computeEnthalpyRadThinSrc(pc, patch, cellinfo,
					    &enthalpyVars, &constEnthalpyVars);
    }

    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    if (d_boundaryCondition->anyArchesPhysicalBC()) {
      d_boundaryCondition->enthalpyBC(pc, patch,  cellinfo, 
				      &enthalpyVars, &constEnthalpyVars);

      if (d_boundaryCondition->getIntrusionBC()) {
        d_boundaryCondition->intrusionEnergyExBC(pc, patch, cellinfo,
					      &enthalpyVars,&constEnthalpyVars);
        d_boundaryCondition->intrusionEnthalpyBC(pc, patch, delta_t, cellinfo,
					      &enthalpyVars,&constEnthalpyVars);
      }
    }

    // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &enthalpyVars, &constEnthalpyVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t,
				       &enthalpyVars, &constEnthalpyVars,
				       d_conv_scheme);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &enthalpyVars);

    // apply underelax to eqn
    d_linearSolver->computeEnthalpyUnderrelax(pc, patch,
					      &enthalpyVars,&constEnthalpyVars);

  }
}


//****************************************************************************
// Schedule linear solve of enthalpy
//****************************************************************************
void
EnthalpySolver::sched_enthalpyLinearSolve(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
				          const TimeIntegratorLabel* timelabels)
{
  string taskname =  "EnthalpySolver::enthalpyLinearSolve" + 
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &EnthalpySolver::enthalpyLinearSolve,
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
    tsk->requires(Task::NewDW, d_lab->d_enthalpyTempLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) old_values_dw = parent_old_dw;
  else old_values_dw = Task::NewDW;

  tsk->requires(old_values_dw, d_lab->d_enthalpySPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_enthCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_enthNonLinSrcSBLMLabel, 
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

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  } 

  tsk->modifies(d_lab->d_enthalpySPLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual enthalpy solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
EnthalpySolver::enthalpyLinearSolve(const ProcessorGroup* pc,
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
    ArchesVariables enthalpyVars;
    ArchesConstVariables constEnthalpyVars;
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constEnthalpyVars.cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    new_dw->get(constEnthalpyVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constEnthalpyVars.density_guess, d_lab->d_densityGuessLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    if (timelabels->multiple_steps)
      new_dw->get(constEnthalpyVars.old_enthalpy, d_lab->d_enthalpyTempLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else
      old_dw->get(constEnthalpyVars.old_enthalpy, d_lab->d_enthalpySPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constEnthalpyVars.old_old_enthalpy, d_lab->d_enthalpySPLabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constEnthalpyVars.old_old_density, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constEnthalpyVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constEnthalpyVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constEnthalpyVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // for explicit calculation
    new_dw->getModifiable(enthalpyVars.enthalpy, d_lab->d_enthalpySPLabel, 
                matlIndex, patch);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constEnthalpyVars.scalarCoeff[ii],
		  d_lab->d_enthCoefSBLMLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constEnthalpyVars.scalarNonlinearSrc,
		d_lab->d_enthNonLinSrcSBLMLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(enthalpyVars.residualEnthalpy,  patch);


    if (d_MAlab) {
      new_dw->get(constEnthalpyVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		      matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    // make it a separate task later
    if (d_MAlab) 
      d_boundaryCondition->enthalpyLisolve_mm(pc, patch, delta_t, 
					      &enthalpyVars, 
					      &constEnthalpyVars, 
					      cellinfo);
    else
      d_linearSolver->enthalpyLisolve(pc, patch, delta_t, 
				      &enthalpyVars, &constEnthalpyVars,
				      cellinfo);


// Outlet bc is done here not to change old enthalpy
    if (d_boundaryCondition->getOutletBC())
    d_boundaryCondition->enthalpyOutletBC(pc, patch,  cellinfo, 
					  &enthalpyVars, &constEnthalpyVars,
					  delta_t, maxUxplus, maxAbsV, maxAbsW);

    if (d_boundaryCondition->getPressureBC())
    d_boundaryCondition->enthalpyPressureBC(pc, patch,  cellinfo, 
				  	    &enthalpyVars,&constEnthalpyVars);

  }
}
