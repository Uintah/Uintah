//----- ExplicitSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ExplicitSolver.h>
#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/EnthalpySolver.h>
#include <Packages/Uintah/CCA/Components/Arches/MomentumSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/ScalarSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ReactiveScalarSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#ifdef PetscFilter
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#endif

#include <math.h>

using namespace Uintah;

// ****************************************************************************
// Default constructor for ExplicitSolver
// ****************************************************************************
ExplicitSolver::
ExplicitSolver(const ArchesLabel* label, 
	       const MPMArchesLabel* MAlb,
	       Properties* props, 
	       BoundaryCondition* bc,
	       TurbulenceModel* turbModel,
	       PhysicalConstants* physConst,
	       bool calc_reactingScalar,
	       bool calc_enthalpy,
	       const ProcessorGroup* myworld): 
               NonlinearSolver(myworld),
	       d_lab(label), d_MAlab(MAlb), d_props(props), 
	       d_boundaryCondition(bc), d_turbModel(turbModel),
	       d_reactingScalarSolve(calc_reactingScalar),
	       d_enthalpySolve(calc_enthalpy),
	       d_physicalConsts(physConst)
{
  d_pressSolver = 0;
  d_momSolver = 0;
  d_scalarSolver = 0;
  d_reactingScalarSolver = 0;
  d_enthalpySolver = 0;
  nosolve_timelabels_allocated = false;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
ExplicitSolver::~ExplicitSolver()
{
  delete d_pressSolver;
  delete d_momSolver;
  delete d_scalarSolver;
  delete d_reactingScalarSolver;
  delete d_enthalpySolver;
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
    delete d_timeIntegratorLabels[curr_level];
  if (nosolve_timelabels_allocated)
    delete nosolve_timelabels;
}

// ****************************************************************************
// Problem Setup 
// ****************************************************************************
void 
ExplicitSolver::problemSetup(const ProblemSpecP& params)
  // MultiMaterialInterface* mmInterface
{
  ProblemSpecP db = params->findBlock("ExplicitSolver");
  db->require("probe_data", d_probe_data);
  if (d_probe_data) {
    IntVector prbPoint;
    for (ProblemSpecP probe_db = db->findBlock("ProbePoints");
	 probe_db;
	 probe_db = probe_db->findNextBlock("ProbePoints")) {
      probe_db->require("probe_point", prbPoint);
      d_probePoints.push_back(prbPoint);
    }
  }
  bool calPress;
  db->require("cal_pressure", calPress);
  if (calPress) {
    d_pressSolver = scinew PressureSolver(d_lab, d_MAlab,
					  d_turbModel, d_boundaryCondition,
					  d_physicalConsts, d_myworld);
    d_pressSolver->problemSetup(db); // d_mmInterface
  }
  bool calMom;
  db->require("cal_momentum", calMom);
  if (calMom) {
    d_momSolver = scinew MomentumSolver(d_lab, d_MAlab,
					d_turbModel, d_boundaryCondition,
					d_physicalConsts);
    d_momSolver->problemSetup(db); // d_mmInterface
  }
  if ((calPress)&&(calMom)) {
  d_pressure_correction = d_momSolver->getPressureCorrectionFlag();
  d_pressSolver->setPressureCorrectionFlag(d_pressure_correction);
  }
  bool calScalar;
  db->require("cal_mixturescalar", calScalar);
  if (calScalar) {
    d_scalarSolver = scinew ScalarSolver(d_lab, d_MAlab,
					 d_turbModel, d_boundaryCondition,
					 d_physicalConsts);
    d_scalarSolver->problemSetup(db);
  }
  if (d_reactingScalarSolve) {
    d_reactingScalarSolver = scinew ReactiveScalarSolver(d_lab, d_MAlab,
					     d_turbModel, d_boundaryCondition,
					     d_physicalConsts);
    d_reactingScalarSolver->problemSetup(db);
  }
  if (d_enthalpySolve) {
    d_enthalpySolver = scinew EnthalpySolver(d_lab, d_MAlab,
					     d_turbModel, d_boundaryCondition,
					     d_physicalConsts, d_myworld);
    d_enthalpySolver->problemSetup(db);
  }
    db->getWithDefault("timeIntegratorType",d_timeIntegratorType,"FE");
    
    if (d_timeIntegratorType == "FE") {
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::FE));
    }
    else if (d_timeIntegratorType == "RK2") {
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::OldPredictor));
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::OldCorrector));
    }
    else if (d_timeIntegratorType == "RK2SSP") {
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::Predictor));
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::Corrector));
    }
    else if (d_timeIntegratorType == "RK3SSP") {
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::Predictor));
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::Intermediate));
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::CorrectorRK3));
    }
    else if (d_timeIntegratorType == "BEEmulation") {
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::BEEmulation1));
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::BEEmulation2));
      d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
					TimeIntegratorStepType::BEEmulation3));
    }
    else {
      throw ProblemSetupException("Integrator type is not defined "+d_timeIntegratorType);
    }
}

// ****************************************************************************
// Schedule non linear solve and carry out some actual operations
// ****************************************************************************
int ExplicitSolver::nonlinearSolve(const LevelP& level,
					  SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP, 
  // densityCP, viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN

  sched_setInitialGuess(sched, patches, matls);

  // Start the iterations

  int nofScalars = d_props->getNumMixVars();
  int nofScalarVars = d_props->getNumMixStatVars();

  // check if filter is defined...only required if using dynamic
  // or scalesimilarity models
#ifdef PetscFilter
  if (d_turbModel->getFilter()) {
    // if the matrix is not initialized
    if (!d_turbModel->getFilter()->isInitialized()) 
      d_turbModel->sched_initFilterMatrix(level, sched, patches, matls);
    d_props->setFilter(d_turbModel->getFilter());
#ifdef divergenceconstraint
    d_momSolver->setDiscretizationFilter(d_turbModel->getFilter());
#endif
  }
#endif

  if (d_timeIntegratorType == "FE")
    numTimeIntegratorLevels = 1;
  else if ((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "RK2SSP"))
    numTimeIntegratorLevels = 2;
  else if ((d_timeIntegratorType == "RK3SSP")||(d_timeIntegratorType == "BEEmulation"))
    numTimeIntegratorLevels = 3;

  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
  {

    if (curr_level > 0)
       sched_saveTempCopies(sched, patches, matls,
			   	      d_timeIntegratorLabels[curr_level]);
    
    for (int index = 0;index < nofScalars; index ++) {
    // in this case we're only solving for one scalar...but
    // the same subroutine can be used to solve multiple scalars
      d_scalarSolver->solve(sched, patches, matls, 
			    d_timeIntegratorLabels[curr_level], index);
    }

    if (d_reactingScalarSolve) {
      int index = 0;
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve multiple scalars
      d_reactingScalarSolver->solve(sched, patches, matls,
				    d_timeIntegratorLabels[curr_level], index);
    }

    if (d_enthalpySolve)
      d_enthalpySolver->solve(level, sched, patches, matls,
			      d_timeIntegratorLabels[curr_level]);

    if (nofScalarVars > 0) {
      for (int index = 0;index < nofScalarVars; index ++) {
        // in this case we're only solving for one scalarVar...but
        // the same subroutine can be used to solve multiple scalarVars
        d_turbModel->sched_computeScalarVariance(sched, patches, matls,
					   d_timeIntegratorLabels[curr_level]);
      }
    d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
					   d_timeIntegratorLabels[curr_level]);
    }

    d_props->sched_reComputeProps(sched, patches, matls,
				  d_timeIntegratorLabels[curr_level], true);
    d_props->sched_computeDenRefArray(sched, patches, matls,
				      d_timeIntegratorLabels[curr_level]);

    // linearizes and solves pressure eqn
    // first computes, hatted velocities and then computes
    // the pressure poisson equation
    d_momSolver->solveVelHat(level, sched, d_timeIntegratorLabels[curr_level]);

    // averaging for RKSSP
    if ((curr_level>0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {
      d_props->sched_averageRKProps(sched, patches, matls,
			   	    d_timeIntegratorLabels[curr_level]);
      d_props->sched_saveTempDensity(sched, patches, matls,
			   	     d_timeIntegratorLabels[curr_level]);
      d_props->sched_reComputeProps(sched, patches, matls,
				    d_timeIntegratorLabels[curr_level], false);
      if (nofScalarVars > 0) {
        for (int index = 0;index < nofScalarVars; index ++) {
        // in this case we're only solving for one scalarVar...but
        // the same subroutine can be used to solve multiple scalarVars
          d_turbModel->sched_computeScalarVariance(sched, patches, matls,
					    d_timeIntegratorLabels[curr_level]);
        }
        d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
					    d_timeIntegratorLabels[curr_level]);
      }
      d_momSolver->sched_averageRKHatVelocities(sched, patches, matls,
					    d_timeIntegratorLabels[curr_level]);
    } 

    d_props->sched_computeDrhodt(sched, patches, matls,
				 d_timeIntegratorLabels[curr_level]);

    d_pressSolver->solve(level, sched, d_timeIntegratorLabels[curr_level]);
  
    // project velocities using the projection step
    for (int index = 1; index <= Arches::NDIM; ++index) {
      d_momSolver->solve(sched, patches, matls,
			 d_timeIntegratorLabels[curr_level], index);
    }
    if (d_pressure_correction)
    sched_updatePressure(sched, patches, matls,
				 d_timeIntegratorLabels[curr_level]);

    //if (curr_level == numTimeIntegratorLevels - 1) {
    d_boundaryCondition->sched_getFlowINOUT(sched, patches, matls,
					    d_timeIntegratorLabels[curr_level]);
    d_boundaryCondition->sched_correctVelocityOutletBC(sched, patches, matls,
					    d_timeIntegratorLabels[curr_level]);
    //}

    // Schedule an interpolation of the face centered velocity data 
    sched_interpolateFromFCToCC(sched, patches, matls,
				d_timeIntegratorLabels[curr_level]);
    d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
					    d_timeIntegratorLabels[curr_level]);

    sched_printTotalKE(sched, patches, matls,
		       d_timeIntegratorLabels[curr_level]);
  }

  // print information at probes provided in input file
  if (d_probe_data)
    sched_probeData(sched, patches, matls);


  return(0);
}

// ****************************************************************************
// No Solve option (used to skip first time step calculation
// so that further time steps will have correct initial condition)
// ****************************************************************************

int ExplicitSolver::noSolve(const LevelP& level,
					  SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  // use FE timelabels for nosolve
  nosolve_timelabels = scinew TimeIntegratorLabel(d_lab,
					    TimeIntegratorStepType::FE);
  nosolve_timelabels_allocated = true;

  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP, 
  // densityCP, viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN

  sched_setInitialGuess(sched, patches, matls);

  d_props->sched_computePropsFirst_mm(sched, patches, matls);

  d_props->sched_computeDrhodt(sched, patches, matls,
				 nosolve_timelabels);

  sched_dummySolve(sched, patches, matls);

  sched_interpolateFromFCToCC(sched, patches, matls, nosolve_timelabels);
  
  d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
					   nosolve_timelabels);
  
  d_pressSolver->sched_addHydrostaticTermtoPressure(sched, patches, matls);
 
  // Schedule an interpolation of the face centered velocity data 
  // to a cell centered vector for used by the viz tools

  // print information at probes provided in input file

  if (d_probe_data)
    sched_probeData(sched, patches, matls);

  return(0);
}

// ****************************************************************************
// Schedule initialize 
// ****************************************************************************
void 
ExplicitSolver::sched_setInitialGuess(SchedulerP& sched, 
				      const PatchSet* patches,
				      const MaterialSet* matls)
{
  //copies old db to new_db and then uses non-linear
  //solver to compute new values
  Task* tsk = scinew Task( "ExplicitSolver::initialGuess",
			   this, &ExplicitSolver::setInitialGuess);
  if (d_MAlab) 
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  else
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_MAlab)
    tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  int nofScalars = d_props->getNumMixVars();
  // warning **only works for one scalar
  for (int ii = 0; ii < nofScalars; ii++) {
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_reactingScalarSolve) {
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_enthalpySolve)
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_viscosityCTSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->computes(d_lab->d_cellTypeLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);
  tsk->computes(d_lab->d_uVelRhoHatLabel);
  tsk->computes(d_lab->d_vVelRhoHatLabel);
  tsk->computes(d_lab->d_wVelRhoHatLabel);

  for (int ii = 0; ii < nofScalars; ii++) {
    tsk->computes(d_lab->d_scalarSPLabel);
    if (d_timeIntegratorLabels[0]->multiple_steps)
      tsk->computes(d_lab->d_scalarTempLabel);
  }

  if (d_reactingScalarSolve) {
    tsk->computes(d_lab->d_reactscalarSPLabel);
    if (d_timeIntegratorLabels[0]->multiple_steps)
      tsk->computes(d_lab->d_reactscalarTempLabel);
  }
  if (d_enthalpySolve) {
    tsk->computes(d_lab->d_enthalpySPLabel);
    if (d_timeIntegratorLabels[0]->multiple_steps)
    tsk->computes(d_lab->d_enthalpyTempLabel);
  }
  tsk->computes(d_lab->d_densityCPLabel);
  if (d_timeIntegratorLabels[0]->multiple_steps)
    tsk->computes(d_lab->d_densityTempLabel);
  tsk->computes(d_lab->d_viscosityCTSLabel);
  if (d_MAlab)
    tsk->computes(d_lab->d_densityMicroINLabel);
  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Schedule data copy for first time step of Multimaterial algorithm
// ****************************************************************************
void
ExplicitSolver::sched_dummySolve(SchedulerP& sched,
			       const PatchSet* patches,
			       const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::dataCopy",
			   this, &ExplicitSolver::dummySolve);

  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

  int nofScalarVars = d_props->getNumMixStatVars();
  // warning **only works for one scalarVar
  if (nofScalarVars > 0) {
    for (int ii = 0; ii < nofScalarVars; ii++) {
      tsk->requires(Task::OldDW, d_lab->d_scalarVarSPLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    }
  }


  tsk->computes(d_lab->d_pressurePSLabel);
  tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

  // warning **only works for one scalar

  tsk->computes(d_lab->d_uvwoutLabel);
  tsk->computes(d_lab->d_totalflowINLabel);
  tsk->computes(d_lab->d_totalflowOUTLabel);
  tsk->computes(d_lab->d_netflowOUTBCLabel);
  tsk->computes(d_lab->d_denAccumLabel);

  tsk->requires(Task::OldDW, d_lab->d_maxAbsU_label);
  tsk->requires(Task::OldDW, d_lab->d_maxAbsV_label);
  tsk->requires(Task::OldDW, d_lab->d_maxAbsW_label);

  if (nofScalarVars > 0) {
    for (int ii = 0; ii < nofScalarVars; ii++) {
      tsk->computes(d_lab->d_scalarVarSPLabel);
    }
  }

  tsk->computes(d_lab->d_maxAbsU_label);
  tsk->computes(d_lab->d_maxAbsV_label);
  tsk->computes(d_lab->d_maxAbsW_label);

  sched->addTask(tsk, patches, matls);  
  
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void 
ExplicitSolver::sched_interpolateFromFCToCC(SchedulerP& sched, 
					    const PatchSet* patches,
					    const MaterialSet* matls,
				         const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::interpFCToCC" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this, 
			 &ExplicitSolver::interpolateFromFCToCC, timelabels);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel, 
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
// hat velocities are only interpolated for first substep, since they are
// not really needed anyway
  tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->computes(d_lab->d_oldCCVelocityLabel);
  tsk->computes(d_lab->d_uVelRhoHat_CCLabel);
  tsk->computes(d_lab->d_vVelRhoHat_CCLabel);
  tsk->computes(d_lab->d_wVelRhoHat_CCLabel);
  }


  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
  tsk->computes(d_lab->d_newCCVelocityLabel);
  tsk->computes(d_lab->d_newCCVelMagLabel);
  tsk->computes(d_lab->d_newCCUVelocityLabel);
  tsk->computes(d_lab->d_newCCVVelocityLabel);
  tsk->computes(d_lab->d_newCCWVelocityLabel);
  tsk->computes(d_lab->d_kineticEnergyLabel);
  }
  else {
  tsk->modifies(d_lab->d_newCCVelocityLabel);
  tsk->modifies(d_lab->d_newCCVelMagLabel);
  tsk->modifies(d_lab->d_newCCUVelocityLabel);
  tsk->modifies(d_lab->d_newCCVVelocityLabel);
  tsk->modifies(d_lab->d_newCCWVelocityLabel);
  tsk->modifies(d_lab->d_kineticEnergyLabel);
  }
  tsk->computes(timelabels->tke_out);
      
  sched->addTask(tsk, patches, matls);  
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector 
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void 
ExplicitSolver::interpolateFromFCToCC(const ProcessorGroup* ,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constSFCXVariable<double> oldUVel;
    constSFCYVariable<double> oldVVel;
    constSFCZVariable<double> oldWVel;
    constSFCXVariable<double> uHatVel_FCX;
    constSFCYVariable<double> vHatVel_FCY;
    constSFCZVariable<double> wHatVel_FCZ;
    CCVariable<Vector> oldCCVel;
    CCVariable<double> uHatVel_CC;
    CCVariable<double> vHatVel_CC;
    CCVariable<double> wHatVel_CC;

    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    CCVariable<Vector> newCCVel;
    CCVariable<double> newCCVelMag;
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    CCVariable<double> kineticEnergy;
    
    IntVector idxLo = patch->getCellLowIndex();
    IntVector idxHi = patch->getCellHighIndex();

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    old_dw->get(oldUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    old_dw->get(oldVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    old_dw->get(oldWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(uHatVel_FCX, d_lab->d_uVelRhoHatLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vHatVel_FCY, d_lab->d_vVelRhoHatLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wHatVel_FCZ, d_lab->d_wVelRhoHatLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(oldCCVel, d_lab->d_oldCCVelocityLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(uHatVel_CC, d_lab->d_uVelRhoHat_CCLabel, 
			   matlIndex, patch);
    new_dw->allocateAndPut(vHatVel_CC, d_lab->d_vVelRhoHat_CCLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(wHatVel_CC, d_lab->d_wVelRhoHat_CCLabel,
			   matlIndex, patch);
    for (int kk = idxLo.z(); kk < idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj < idxHi.y(); ++jj) {
	for (int ii = idxLo.x(); ii < idxHi.x(); ++ii) {
	  
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double old_u = 0.5*(oldUVel[idx] + 
			      oldUVel[idxU]);
	  double uhat = 0.5*(uHatVel_FCX[idx] +
			     uHatVel_FCX[idxU]);
	  double old_v = 0.5*(oldVVel[idx] +
			      oldVVel[idxV]);
	  double vhat = 0.5*(vHatVel_FCY[idx] +
			     vHatVel_FCY[idxV]);
	  double old_w = 0.5*(oldWVel[idx] +
			      oldWVel[idxW]);
	  double what = 0.5*(wHatVel_FCZ[idx] +
			     wHatVel_FCZ[idxW]);
	  
	  oldCCVel[idx] = Vector(old_u,old_v,old_w);
	  uHatVel_CC[idx] = uhat;
	  vHatVel_CC[idx] = vhat;
	  wHatVel_CC[idx] = what;
	}
      }
    }
    } 

    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    new_dw->allocateAndPut(newCCVel, d_lab->d_newCCVelocityLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCVelMag, d_lab->d_newCCVelMagLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCUVel, d_lab->d_newCCUVelocityLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCVVel, d_lab->d_newCCVVelocityLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCWVel, d_lab->d_newCCWVelocityLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(kineticEnergy, d_lab->d_kineticEnergyLabel,
			   matlIndex, patch);
    }
    else {
    new_dw->getModifiable(newCCVel, d_lab->d_newCCVelocityLabel,
			   matlIndex, patch);
    new_dw->getModifiable(newCCVelMag, d_lab->d_newCCVelMagLabel,
			   matlIndex, patch);
    new_dw->getModifiable(newCCUVel, d_lab->d_newCCUVelocityLabel,
			   matlIndex, patch);
    new_dw->getModifiable(newCCVVel, d_lab->d_newCCVVelocityLabel,
			   matlIndex, patch);
    new_dw->getModifiable(newCCWVel, d_lab->d_newCCWVelocityLabel,
			   matlIndex, patch);
    new_dw->getModifiable(kineticEnergy, d_lab->d_kineticEnergyLabel,
			   matlIndex, patch);
    }
    newCCUVel.initialize(0.0);
    newCCVVel.initialize(0.0);
    newCCWVel.initialize(0.0);
    kineticEnergy.initialize(0.0);


    double total_kin_energy = 0.0;

    for (int kk = idxLo.z(); kk < idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj < idxHi.y(); ++jj) {
	for (int ii = idxLo.x(); ii < idxHi.x(); ++ii) {
	  
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  newCCVel[idx] = Vector(new_u,new_v,new_w);
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
	  total_kin_energy += kineticEnergy[idx];
	}
      }
    }
    new_dw->put(sum_vartype(total_kin_energy), timelabels->tke_out); 
  }
}

// ****************************************************************************
// Schedule probe data
// ****************************************************************************
void 
ExplicitSolver::sched_probeData(SchedulerP& sched, const PatchSet* patches,
				       const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::probeData",
			  this, &ExplicitSolver::probeData);
  
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_kineticEnergyLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  int nofScalarVars = d_props->getNumMixStatVars();
  if (nofScalarVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, 
    		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_tempINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    
    tsk->requires(Task::NewDW, d_MAlab->totHT_CCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::NewDW, d_MAlab->totHT_FCXLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCYLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCZLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::NewDW, d_MAlab->totHtFluxXLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->totHtFluxYLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->totHtFluxZLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  }

  sched->addTask(tsk, patches, matls);
  
}
// ****************************************************************************
// Actual probe data
// ****************************************************************************

void 
ExplicitSolver::probeData(const ProcessorGroup* ,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse*,
				 DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    double time = d_lab->d_sharedState->getElapsedTime();

  // Get the new velocity
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    constCCVariable<double> newintUVel;
    constCCVariable<double> newintVVel;
    constCCVariable<double> newintWVel;
    new_dw->get(newintUVel, d_lab->d_newCCUVelocityLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(newintVVel, d_lab->d_newCCVVelocityLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(newintWVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> pressure;
    constCCVariable<double> mixtureFraction;
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(pressure, d_lab->d_pressurePSLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(mixtureFraction, d_lab->d_scalarSPLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    constCCVariable<double> kineticEnergy;
    new_dw->get(kineticEnergy, d_lab->d_kineticEnergyLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    
    constCCVariable<double> mixFracVariance;
    if (d_props->getNumMixStatVars() > 0) {
      new_dw->get(mixFracVariance, d_lab->d_scalarVarSPLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    
    constCCVariable<double> gasfraction;
    constCCVariable<double> tempSolid;
    constCCVariable<double> totalHT;
    constSFCXVariable<double> totalHT_FCX;
    constSFCYVariable<double> totalHT_FCY;
    constSFCZVariable<double> totalHT_FCZ;
    constSFCXVariable<double> totHtFluxX;
    constSFCYVariable<double> totHtFluxY;
    constSFCZVariable<double> totHtFluxZ;
    if (d_MAlab) {
      new_dw->get(gasfraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(tempSolid, d_MAlab->integTemp_CCLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->get(totalHT, d_MAlab->totHT_CCLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->get(totalHT_FCX, d_MAlab->totHT_FCXLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(totalHT_FCY, d_MAlab->totHT_FCYLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(totalHT_FCZ, d_MAlab->totHT_FCZLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->get(totHtFluxX, d_MAlab->totHtFluxXLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(totHtFluxY, d_MAlab->totHtFluxYLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(totHtFluxZ, d_MAlab->totHtFluxZLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    constCCVariable<double> temperature;
    if (d_enthalpySolve) 
      new_dw->get(temperature, d_lab->d_tempINLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    for (vector<IntVector>::const_iterator iter = d_probePoints.begin();
	 iter != d_probePoints.end(); iter++) {

      if (patch->containsCell(*iter)) {
	cerr.precision(10);
	cerr << "for Intvector: " << *iter << endl;
	cerr << "Density: " << density[*iter] << endl;
	cerr << "Viscosity: " << viscosity[*iter] << endl;
	cerr << "Pressure: " << pressure[*iter] << endl;
	cerr << "MixtureFraction: " << mixtureFraction[*iter] << endl;
	if (d_enthalpySolve)
	  cerr<<"Gas Temperature: " << temperature[*iter] << endl;
	cerr << "UVelocity: " << newUVel[*iter] << endl;
	cerr << "VVelocity: " << newVVel[*iter] << endl;
	cerr << "WVelocity: " << newWVel[*iter] << endl;
	cerr << "CCUVelocity: " << newintUVel[*iter] << endl;
	cerr << "CCVVelocity: " << newintVVel[*iter] << endl;
	cerr << "CCWVelocity: " << newintWVel[*iter] << endl;
	cerr << "KineticEnergy: " << kineticEnergy[*iter] << endl;
	if (d_props->getNumMixStatVars() > 0) {
	  cerr << "MixFracVariance: " << mixFracVariance[*iter] << endl;
	}
	if (d_MAlab) {
	  cerr.precision(16);
	  cerr << "gas vol fraction: " << gasfraction[*iter] << endl;
	  cerr << " Solid Temperature at Location " << *iter << " At time " << time << ","<< tempSolid[*iter] << endl;
	  cerr << " Total Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT[*iter] << endl;
	  cerr << " Total X-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCX[*iter] << endl;
	  cerr << " Total Y-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCY[*iter] << endl;
	  cerr << " Total Z-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCZ[*iter] << endl;
	  cerr << " Total X-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxX[*iter] << endl;
	  cerr << " Total Y-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxY[*iter] << endl;
	  cerr << " Total Z-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxZ[*iter] << endl;
	}

      }
    }
  }
}

// ****************************************************************************
// Actual initialize 
// ****************************************************************************
void 
ExplicitSolver::setInitialGuess(const ProcessorGroup* ,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<double> denMicro;
    CCVariable<double> denMicro_new;
    if (d_MAlab) {
      old_dw->get(denMicro, d_lab->d_densityMicroLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(denMicro_new, d_lab->d_densityMicroINLabel, 
		       matlIndex, patch);
      denMicro_new.copyData(denMicro);
    }
    constCCVariable<int> cellType;
    if (d_MAlab)
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    else
      old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    constSFCXVariable<double> uVelocity;
    old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    constSFCYVariable<double> vVelocity;
    old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    constSFCZVariable<double> wVelocity;
    old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    int nofScalars = d_props->getNumMixVars();
    StaticArray< constCCVariable<double> > scalar (nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      old_dw->get(scalar[ii], d_lab->d_scalarSPLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    constCCVariable<double> enthalpy;
    if (d_enthalpySolve)
      old_dw->get(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    constCCVariable<double> density;
    old_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    constCCVariable<double> viscosity;
    old_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);


  // Create vars for new_dw ***warning changed new_dw to old_dw...check
    CCVariable<int> cellType_new;
    new_dw->allocateAndPut(cellType_new, d_lab->d_cellTypeLabel, matlIndex, patch);
    cellType_new.copyData(cellType);
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }

#if 0
    PerPatch<CellInformationP> cellInfoP;
    cellInfoP.setData(scinew CellInformation(patch));
    new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
#endif
    SFCXVariable<double> uVelocity_new;
    new_dw->allocateAndPut(uVelocity_new, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    uVelocity_new.copyData(uVelocity); // copy old into new
    SFCYVariable<double> vVelocity_new;
    new_dw->allocateAndPut(vVelocity_new, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    vVelocity_new.copyData(vVelocity); // copy old into new
    SFCZVariable<double> wVelocity_new;
    new_dw->allocateAndPut(wVelocity_new, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    wVelocity_new.copyData(wVelocity); // copy old into new
    SFCXVariable<double> uVelRhoHat_new;
    new_dw->allocateAndPut(uVelRhoHat_new, d_lab->d_uVelRhoHatLabel, matlIndex, patch);
    uVelRhoHat_new.initialize(0.0); // copy old into new
    SFCYVariable<double> vVelRhoHat_new;
    new_dw->allocateAndPut(vVelRhoHat_new, d_lab->d_vVelRhoHatLabel, matlIndex, patch);
    vVelRhoHat_new.initialize(0.0); // copy old into new
    SFCZVariable<double> wVelRhoHat_new;
    new_dw->allocateAndPut(wVelRhoHat_new, d_lab->d_wVelRhoHatLabel, matlIndex, patch);
    wVelRhoHat_new.initialize(0.0); // copy old into new

    StaticArray<CCVariable<double> > scalar_new(nofScalars);
    StaticArray<CCVariable<double> > scalar_temp(nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->allocateAndPut(scalar_new[ii], d_lab->d_scalarSPLabel, matlIndex, patch);
      scalar_new[ii].copyData(scalar[ii]); // copy old into new
      if (d_timeIntegratorLabels[0]->multiple_steps) {
      new_dw->allocateAndPut(scalar_temp[ii], d_lab->d_scalarTempLabel, matlIndex, patch);
      scalar_temp[ii].copyData(scalar[ii]); // copy old into new
      }
    }

    constCCVariable<double> reactscalar;
    CCVariable<double> new_reactscalar;
    CCVariable<double> temp_reactscalar;
    if (d_reactingScalarSolve) {
      old_dw->get(reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(new_reactscalar, d_lab->d_reactscalarSPLabel, matlIndex,
		       patch);
      new_reactscalar.copyData(reactscalar);
      if (d_timeIntegratorLabels[0]->multiple_steps) {
      new_dw->allocateAndPut(temp_reactscalar, d_lab->d_reactscalarTempLabel, matlIndex,
		       patch);
      temp_reactscalar.copyData(reactscalar);
      }
    }


    CCVariable<double> new_enthalpy;
    CCVariable<double> temp_enthalpy;
    if (d_enthalpySolve) {
      new_dw->allocateAndPut(new_enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);
      new_enthalpy.copyData(enthalpy);
      if (d_timeIntegratorLabels[0]->multiple_steps) {
      new_dw->allocateAndPut(temp_enthalpy, d_lab->d_enthalpyTempLabel, matlIndex, patch);
      temp_enthalpy.copyData(enthalpy);
      }
    }
    CCVariable<double> density_new;
    new_dw->allocateAndPut(density_new, d_lab->d_densityCPLabel, matlIndex, patch);
    density_new.copyData(density); // copy old into new
    if (d_timeIntegratorLabels[0]->multiple_steps) {
      CCVariable<double> density_temp;
      new_dw->allocateAndPut(density_temp, d_lab->d_densityTempLabel, matlIndex, patch);
      density_temp.copyData(density); // copy old into new
    }

    CCVariable<double> viscosity_new;
    new_dw->allocateAndPut(viscosity_new, d_lab->d_viscosityCTSLabel, matlIndex, patch);
    viscosity_new.copyData(viscosity); // copy old into new
  }
}


// ****************************************************************************
// Actual Data Copy for first time step of MPMArches
// ****************************************************************************

void 
ExplicitSolver::dummySolve(const ProcessorGroup* ,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw)
{
  max_vartype mxAbsU;
  max_vartype mxAbsV;
  max_vartype mxAbsW;
  old_dw->get(mxAbsU, d_lab->d_maxAbsU_label);
  old_dw->get(mxAbsV, d_lab->d_maxAbsV_label);
  old_dw->get(mxAbsW, d_lab->d_maxAbsW_label);
  new_dw->put(mxAbsU, d_lab->d_maxAbsU_label);
  new_dw->put(mxAbsV, d_lab->d_maxAbsV_label);
  new_dw->put(mxAbsW, d_lab->d_maxAbsW_label);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // gets for old dw variables

    int nofScalarVars = d_props->getNumMixStatVars();
    StaticArray< constCCVariable<double> > scalarVar (nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	old_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, matlIndex, patch, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      }
    }

    constCCVariable<double> pressure;
    old_dw->get(pressure, d_lab->d_pressurePSLabel, matlIndex, patch, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);


    // allocates and puts for new dw variables

    CCVariable<double> pressure_new;
    new_dw->allocateAndPut(pressure_new, d_lab->d_pressurePSLabel, 
			   matlIndex, patch);
    pressure_new.copyData(pressure); // copy old into new

    CCVariable<double> pressureNLSource;
    new_dw->allocateAndPut(pressureNLSource, d_lab->d_presNonLinSrcPBLMLabel, 
			   matlIndex, patch);
    pressureNLSource.initialize(0.0);

    StaticArray<CCVariable<double> > scalarVar_new(nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->allocateAndPut(scalarVar_new[ii], d_lab->d_scalarVarSPLabel, matlIndex, patch);
	scalarVar_new[ii].copyData(scalarVar[ii]); // copy old into new
      }
    }


    cout << "DOING DUMMY SOLVE " << endl;

    double uvwout = 0.0;
    double flowIN = 0.0;
    double flowOUT = 0.0;
    double flowOUToutbc = 0.0;
    double denAccum = 0.0;


    new_dw->put(delt_vartype(uvwout), d_lab->d_uvwoutLabel);
    new_dw->put(delt_vartype(flowIN), d_lab->d_totalflowINLabel);
    new_dw->put(delt_vartype(flowOUT), d_lab->d_totalflowOUTLabel);
    new_dw->put(delt_vartype(flowOUToutbc), d_lab->d_netflowOUTBCLabel);
    new_dw->put(delt_vartype(denAccum), d_lab->d_denAccumLabel);

  }
}

// ****************************************************************************
// compute the residual
// ****************************************************************************
double 
ExplicitSolver::computeResidual(const LevelP&,
				SchedulerP&,
				DataWarehouseP&,
				DataWarehouseP&)
{
  double nlresidual = 0.0;
#if 0
  SoleVariable<double> residual;
  SoleVariable<double> omg;
  // not sure of the syntax...this operation is supposed to get 
  // L1norm of the residual over the whole level
  new_dw->get(residual,"pressResidual");
  new_dw->get(omg,"pressomg");
  nlresidual = MACHINEPRECISSION + log(residual/omg);
  for (int index = 1; index <= Arches::NDIM; ++index) {
    new_dw->get(residual,"velocityResidual", index);
    new_dw->get(omg,"velocityomg", index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
  //for multiple scalars iterate
  for (int index = 1;index <= d_props->getNumMixVars(); index ++) {
    new_dw->get(residual,"scalarResidual", index);
    new_dw->get(omg,"scalaromg", index);
    nlresidual = max(nlresidual, MACHINEPRECISSION+log(residual/omg));
  }
#endif
  return nlresidual;
}

void 
ExplicitSolver::sched_printTotalKE(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::printTotalKE" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
			  this, &ExplicitSolver::printTotalKE,
			  timelabels);
  
  tsk->requires(Task::NewDW, timelabels->tke_out);

  sched->addTask(tsk, patches, matls);
  
}
void 
ExplicitSolver::printTotalKE(const ProcessorGroup* ,
			     const PatchSubset* ,
			     const MaterialSubset*,
			     DataWarehouse*,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels)
{

  sum_vartype tke;
  new_dw->get(tke, timelabels->tke_out);
  double total_kin_energy = tke;
  int me = d_myworld->myrank();
  if (me == 0)
     cerr << "Total kinetic energy " <<  total_kin_energy << endl;

}
void 
ExplicitSolver::sched_updatePressure(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::updatePressure" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
			  this, &ExplicitSolver::updatePressure,
			  timelabels);
  
  if ((timelabels->integrator_step_name == "BEEmulation2")||
      (timelabels->integrator_step_name == "BEEmulation3")) 
  tsk->requires(Task::NewDW, timelabels->pressure_guess, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  else
  tsk->requires(Task::OldDW, timelabels->pressure_guess, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->modifies(timelabels->pressure_out);

  sched->addTask(tsk, patches, matls);
  
}
void 
ExplicitSolver::updatePressure(const ProcessorGroup* ,
			     const PatchSubset* patches,
			     const MaterialSubset*,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch *patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

  
    constCCVariable<double> pressure_guess;
    CCVariable<double> pressure;
    new_dw->getModifiable(pressure, timelabels->pressure_out,
			  matlIndex, patch);
    if ((timelabels->integrator_step_name == "BEEmulation2")||
        (timelabels->integrator_step_name == "BEEmulation3")) 
    new_dw->get(pressure_guess, timelabels->pressure_guess, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    else
    old_dw->get(pressure_guess, timelabels->pressure_guess, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    for (int ColX = idxLo.x(); ColX <= idxHi.x(); ColX++) {
      for (int ColY = idxLo.y(); ColY <= idxHi.y(); ColY++) {
        for (int ColZ = idxLo.z(); ColZ <= idxHi.z(); ColZ++) {
	    IntVector currCell(ColX,ColY,ColZ);
	    pressure[currCell] += pressure_guess[currCell];
        }
      }
    }
  }
}
//****************************************************************************
// Schedule saving of temp copies of variables
//****************************************************************************
void 
ExplicitSolver::sched_saveTempCopies(SchedulerP& sched, const PatchSet* patches,
				  const MaterialSet* matls,
			   	  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::saveTempCopies" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ExplicitSolver::saveTempCopies,
			  timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_reactingScalarSolve)
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
 
  tsk->modifies(d_lab->d_densityTempLabel);
  tsk->modifies(d_lab->d_scalarTempLabel);
  if (d_reactingScalarSolve)
    tsk->modifies(d_lab->d_reactscalarTempLabel);
  if (d_enthalpySolve)
    tsk->modifies(d_lab->d_enthalpyTempLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp copies here
//****************************************************************************
void 
ExplicitSolver::saveTempCopies(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse*,
			   DataWarehouse* new_dw,
			   const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> temp_density;
    CCVariable<double> temp_scalar;
    CCVariable<double> temp_reactscalar;
    CCVariable<double> temp_enthalpy;

    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,
			  matlIndex, patch);
    new_dw->copyOut(temp_density, d_lab->d_densityCPLabel,
		     matlIndex, patch);
    new_dw->getModifiable(temp_scalar, d_lab->d_scalarTempLabel,
			  matlIndex, patch);
    new_dw->copyOut(temp_scalar, d_lab->d_scalarSPLabel,
		     matlIndex, patch);
    if (d_reactingScalarSolve) {
    new_dw->getModifiable(temp_reactscalar, d_lab->d_reactscalarTempLabel,
			  matlIndex, patch);
    new_dw->copyOut(temp_reactscalar, d_lab->d_reactscalarSPLabel,
		     matlIndex, patch);
    }
    if (d_enthalpySolve) {
    new_dw->getModifiable(temp_enthalpy, d_lab->d_enthalpyTempLabel,
			  matlIndex, patch);
    new_dw->copyOut(temp_enthalpy, d_lab->d_enthalpySPLabel,
		     matlIndex, patch);
    }
  }
}
