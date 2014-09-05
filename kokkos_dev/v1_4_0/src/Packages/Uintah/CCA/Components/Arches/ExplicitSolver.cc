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
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

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
					     d_physicalConsts);
    d_enthalpySolver->problemSetup(db);
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

  //correct inlet velocities to account for change in properties
  // require : densityIN, [u,v,w]VelocityIN (new_dw)
  // compute : [u,v,w]VelocitySIVBC
  d_boundaryCondition->sched_setInletVelocityBC(sched, patches, matls);
  d_boundaryCondition->sched_recomputePressureBC(sched, patches, matls);
  // compute total flowin, flow out and overall mass balance
  d_boundaryCondition->sched_computeFlowINOUT(sched, patches, matls);
  d_boundaryCondition->sched_computeOMB(sched, patches, matls);
  d_boundaryCondition->sched_transOutletBC(sched, patches, matls);
  // compute apo and drhodt, used in transport equations
  // put a logical to call computetranscoeff 
  // using a predictor corrector approach from Najm [1998]
  // compute df/dt|n using old values of u,v,w
  // use Adams-Bashforth time integration to compute predicted f*
  // using f* from equation of state compute den* (predicted value)
  // equation for scalars
  // require : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN (new_dw)
  //           scalarSP, densityCP (old_dw)
  // compute : scalarCoefSBLM, scalarLinSrcSBLM, scalarNonLinSrcSBLM
  //           scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSS
  // compute drhophidt, compute dphidt and using both of them compute
  // compute drhodt

  for (int index = 0;index < nofScalars; index ++) {
    // in this case we're only solving for one scalar...but
    // the same subroutine can be used to solve multiple scalars
    d_scalarSolver->solvePred(sched, patches, matls, index);
  }
  if (d_reactingScalarSolver) {
    int index = 0;
    // in this case we're only solving for one scalar...but
    // the same subroutine can be used to solve multiple scalars
    d_reactingScalarSolver->solvePred(sched, patches, matls, index);
  }

  if (nofScalarVars > 0) {
    for (int index = 0;index < nofScalarVars; index ++) {
      // in this case we're only solving for one scalarVar...but
      // the same subroutine can be used to solve multiple scalarVars
      d_turbModel->sched_computeScalarVariance(sched, patches, matls);
    }
  }

  if (d_enthalpySolve)
    d_enthalpySolver->solvePred(sched, patches, matls);
#ifdef correctorstep
  d_props->sched_computePropsPred(sched, patches, matls);
#else
  d_props->sched_reComputeProps(sched, patches, matls);
#endif
  // linearizes and solves pressure eqn
  // require : pressureIN, densityIN, viscosityIN,
  //           [u,v,w]VelocitySIVBC (new_dw)
  //           [u,v,w]VelocitySPBC, densityCP (old_dw)
  // compute : [u,v,w]VelConvCoefPBLM, [u,v,w]VelCoefPBLM, 
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM, (matrix_dw)
  //           presResidualPS, presCoefPBLM, presNonLinSrcPBLM,(matrix_dw)
  //           pressurePS (new_dw)
  // first computes, hatted velocities and then computes the pressure poisson equation
  d_props->sched_computeDenRefArray(sched, patches, matls);
  d_pressSolver->solvePred(level, sched);
  // Momentum solver
  // require : pressureSPBC, [u,v,w]VelocityCPBC, densityIN, 
  // viscosityIN (new_dw)
  //           [u,v,w]VelocitySPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  //           [u,v,w]VelResidualMS, [u,v,w]VelCoefMS, 
  //           [u,v,w]VelNonLinSrcMS, [u,v,w]VelLinSrcMS,
  //           [u,v,w]VelocitySPBC
  
  // project velocities using the projection step
  for (int index = 1; index <= Arches::NDIM; ++index) {
    d_momSolver->solvePred(sched, patches, matls, index);
  }
#ifdef correctorstep
  // corrected step
  for (int index = 0;index < nofScalars; index ++) {
    // in this case we're only solving for one scalar...but
    // the same subroutine can be used to solve multiple scalars
    d_scalarSolver->solveCorr(sched, patches, matls, index);
  }
  if (d_enthalpySolve)
    d_enthalpySolver->solveCorr(sched, patches, matls);
  // same as corrector
  d_props->sched_reComputeProps(sched, patches, matls);
  // linearizes and solves pressure eqn
  // require : pressureIN, densityIN, viscosityIN,
  //           [u,v,w]VelocitySIVBC (new_dw)
  //           [u,v,w]VelocitySPBC, densityCP (old_dw)
  // compute : [u,v,w]VelConvCoefPBLM, [u,v,w]VelCoefPBLM, 
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM, (matrix_dw)
  //           presResidualPS, presCoefPBLM, presNonLinSrcPBLM,(matrix_dw)
  //           pressurePS (new_dw)
  // first computes, hatted velocities and then computes the pressure 
  // poisson equation
  d_pressSolver->solveCorr(level, sched);
  // Momentum solver
  // require : pressureSPBC, [u,v,w]VelocityCPBC, densityIN, 
  // viscosityIN (new_dw)
  //           [u,v,w]VelocitySPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  //           [u,v,w]VelResidualMS, [u,v,w]VelCoefMS, 
  //           [u,v,w]VelNonLinSrcMS, [u,v,w]VelLinSrcMS,
  //           [u,v,w]VelocitySPBC
  
  // project velocities using the projection step
  for (int index = 1; index <= Arches::NDIM; ++index) {
    d_momSolver->solveCorr(sched, patches, matls, index);
  }
  // if external boundary then recompute velocities using new pressure
  // and puts them in nonlinear_dw
  // require : densityCP, pressurePS, [u,v,w]VelocitySIVBC
  // compute : [u,v,w]VelocityCPBC, pressureSPBC
  
#endif
  d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls);
 
  // Schedule an interpolation of the face centered velocity data 
  // to a cell centered vector for used by the viz tools

  sched_interpolateFromFCToCC(sched, patches, matls);
  
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
  int numGhostCells = 0;
  if (d_MAlab) 
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, 
		  Ghost::None, numGhostCells);
  else
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, 
		  Ghost::None, numGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_pressureSPBCLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, numGhostCells);
  if (d_MAlab)
    tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel, 
		  Ghost::None, numGhostCells);
  int nofScalars = d_props->getNumMixVars();
  // warning **only works for one scalar
  for (int ii = 0; ii < nofScalars; ii++) {
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel, 
		  Ghost::None, numGhostCells);
  }

  int nofScalarVars = d_props->getNumMixStatVars();
  // warning **only works for one scalarVar
  if (nofScalarVars > 0) {
    for (int ii = 0; ii < nofScalarVars; ii++) {
      tsk->requires(Task::OldDW, d_lab->d_scalarVarSPLabel, 
		    Ghost::None, numGhostCells);
    }
  }
  if (d_reactingScalarSolve) {
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel, 
		  Ghost::None, numGhostCells);
  }

  if (d_enthalpySolve)
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel, 
		  Ghost::None, numGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_viscosityCTSLabel,
		Ghost::None, numGhostCells);
  tsk->computes(d_lab->d_cellTypeLabel);
  tsk->computes(d_lab->d_pressureINLabel);
  tsk->computes(d_lab->d_uVelocityINLabel);
  tsk->computes(d_lab->d_vVelocityINLabel);
  tsk->computes(d_lab->d_wVelocityINLabel);

  for (int ii = 0; ii < nofScalars; ii++) {
    tsk->computes(d_lab->d_scalarINLabel);
  }

  if (nofScalarVars > 0) {
    for (int ii = 0; ii < nofScalarVars; ii++) {
      tsk->computes(d_lab->d_scalarVarINLabel);
    }
  }
  if (d_reactingScalarSolver)
    tsk->computes(d_lab->d_reactscalarINLabel);
  if (d_enthalpySolve)
    tsk->computes(d_lab->d_enthalpyINLabel);
  tsk->computes(d_lab->d_densityINLabel);
  tsk->computes(d_lab->d_viscosityINLabel);
  if (d_MAlab)
    tsk->computes(d_lab->d_densityMicroINLabel);
  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void 
ExplicitSolver::sched_interpolateFromFCToCC(SchedulerP& sched, 
						   const PatchSet* patches,
						   const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::interpFCToCC",
			   this, &ExplicitSolver::interpolateFromFCToCC);
  int numGhostCells = 1;
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel,
		Ghost::AroundFaces, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel, 
                Ghost::AroundFaces, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, 
		Ghost::AroundFaces, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
                Ghost::AroundFaces, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
                Ghost::AroundFaces, numGhostCells);

  tsk->computes(d_lab->d_oldCCVelocityLabel);
  tsk->computes(d_lab->d_newCCVelocityLabel);
  tsk->computes(d_lab->d_newCCUVelocityLabel);
  tsk->computes(d_lab->d_newCCVVelocityLabel);
  tsk->computes(d_lab->d_newCCWVelocityLabel);
      
  sched->addTask(tsk, patches, matls);

  
}

void 
ExplicitSolver::sched_probeData(SchedulerP& sched, const PatchSet* patches,
				       const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::probeData",
			  this, &ExplicitSolver::probeData);
  //int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::None, zeroGhostCells);

  int nofScalarVars = d_props->getNumMixStatVars();
  if (nofScalarVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, 
    		  Ghost::None, zeroGhostCells);
  }

  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_tempINLabel, 
		  Ghost::None, zeroGhostCells);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, zeroGhostCells);

  sched->addTask(tsk, patches, matls);
  
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
    int nofGhostCells = 0;
    constCCVariable<double> denMicro;
    CCVariable<double> denMicro_new;
    if (d_MAlab) {
      old_dw->get(denMicro, d_lab->d_densityMicroLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);
      new_dw->allocate(denMicro_new, d_lab->d_densityMicroINLabel, 
		       matlIndex, patch);
      denMicro_new.copyData(denMicro);
    }
    constCCVariable<int> cellType;
    if (d_MAlab)
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch,
		  Ghost::None, nofGhostCells);
    else
      old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::None, nofGhostCells);
    constCCVariable<double> pressure;
    old_dw->get(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);

    constSFCXVariable<double> uVelocity;
    old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    constSFCYVariable<double> vVelocity;
    old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    constSFCZVariable<double> wVelocity;
    old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);

    int nofScalars = d_props->getNumMixVars();
    StaticArray< constCCVariable<double> > scalar (nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      old_dw->get(scalar[ii], d_lab->d_scalarSPLabel, matlIndex, patch, 
		  Ghost::None, nofGhostCells);
    }

    int nofScalarVars = d_props->getNumMixStatVars();
    StaticArray< constCCVariable<double> > scalarVar (nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	old_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, matlIndex, patch, 
		    Ghost::None, nofGhostCells);
      }
    }

    constCCVariable<double> enthalpy;
    if (d_enthalpySolve)
      old_dw->get(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch, 
		  Ghost::None, nofGhostCells);

    constCCVariable<double> density;
    old_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);

    constCCVariable<double> viscosity;
    old_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);


  // Create vars for new_dw ***warning changed new_dw to old_dw...check
    CCVariable<int> cellType_new;
    new_dw->allocate(cellType_new, d_lab->d_cellTypeLabel, matlIndex, patch);
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
    CCVariable<double> pressure_new;
    new_dw->allocate(pressure_new, d_lab->d_pressureINLabel, matlIndex, patch);
    pressure_new.copyData(pressure); // copy old into new

    SFCXVariable<double> uVelocity_new;
    new_dw->allocate(uVelocity_new, d_lab->d_uVelocityINLabel, matlIndex, patch);
    uVelocity_new.copyData(uVelocity); // copy old into new
    SFCYVariable<double> vVelocity_new;
    new_dw->allocate(vVelocity_new, d_lab->d_vVelocityINLabel, matlIndex, patch);
    vVelocity_new.copyData(vVelocity); // copy old into new
    SFCZVariable<double> wVelocity_new;
    new_dw->allocate(wVelocity_new, d_lab->d_wVelocityINLabel, matlIndex, patch);
    wVelocity_new.copyData(wVelocity); // copy old into new

    StaticArray<CCVariable<double> > scalar_new(nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->allocate(scalar_new[ii], d_lab->d_scalarINLabel, matlIndex, patch);
      scalar_new[ii].copyData(scalar[ii]); // copy old into new
    }

    StaticArray<CCVariable<double> > scalarVar_new(nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->allocate(scalarVar_new[ii], d_lab->d_scalarVarINLabel, matlIndex, patch);
	scalarVar_new[ii].copyData(scalarVar[ii]); // copy old into new
      }
    }

    constCCVariable<double> reactscalar;
    CCVariable<double> new_reactscalar;
    if (d_reactingScalarSolve) {
      old_dw->get(reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch, 
		  Ghost::None, nofGhostCells);
      new_dw->allocate(new_reactscalar, d_lab->d_reactscalarINLabel, matlIndex,
		       patch);
      new_reactscalar.copyData(reactscalar);
    }


    CCVariable<double> new_enthalpy;
    if (d_enthalpySolve) {
      new_dw->allocate(new_enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch);
      new_enthalpy.copyData(enthalpy);
    }
    CCVariable<double> density_new;
    new_dw->allocate(density_new, d_lab->d_densityINLabel, matlIndex, patch);
    density_new.copyData(density); // copy old into new

    CCVariable<double> viscosity_new;
    new_dw->allocate(viscosity_new, d_lab->d_viscosityINLabel, matlIndex, patch);
    viscosity_new.copyData(viscosity); // copy old into new

    // Copy the variables into the new datawarehouse
    new_dw->put(cellType_new, d_lab->d_cellTypeLabel, matlIndex, patch);
    new_dw->put(pressure_new, d_lab->d_pressureINLabel, matlIndex, patch);
    new_dw->put(uVelocity_new, d_lab->d_uVelocityINLabel, matlIndex, patch);
    new_dw->put(vVelocity_new, d_lab->d_vVelocityINLabel, matlIndex, patch);
    new_dw->put(wVelocity_new, d_lab->d_wVelocityINLabel, matlIndex, patch);

    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->put(scalar_new[ii], d_lab->d_scalarINLabel, matlIndex, patch);
    }

    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->put(scalarVar_new[ii], d_lab->d_scalarVarINLabel, matlIndex, patch);
      }
    }
    if (d_reactingScalarSolve)
      new_dw->put(new_reactscalar, d_lab->d_reactscalarINLabel, matlIndex, patch);
    if (d_enthalpySolve)
      new_dw->put(new_enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch);
    new_dw->put(density_new, d_lab->d_densityINLabel, matlIndex, patch);
    new_dw->put(viscosity_new, d_lab->d_viscosityINLabel, matlIndex, patch);
    if (d_MAlab)
      new_dw->put(denMicro_new, d_lab->d_densityMicroINLabel, matlIndex, patch);
  }
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
					     DataWarehouse*,
					     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int nofGhostCells = 1;

    // Get the old velocity
    constSFCXVariable<double> oldUVel;
    constSFCYVariable<double> oldVVel;
    constSFCZVariable<double> oldWVel;
    new_dw->get(oldUVel, d_lab->d_uVelocityINLabel, matlIndex, patch, 
		Ghost::AroundFaces, nofGhostCells);
    new_dw->get(oldVVel, d_lab->d_vVelocityINLabel, matlIndex, patch, 
		Ghost::AroundFaces, nofGhostCells);
    new_dw->get(oldWVel, d_lab->d_wVelocityINLabel, matlIndex, patch, 
		Ghost::AroundFaces, nofGhostCells);

    // Get the new velocity
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, nofGhostCells);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, nofGhostCells);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, nofGhostCells);
    
    // Get the low and high index for the Cell Centered Variables
    IntVector idxLo = patch->getCellLowIndex();
    IntVector idxHi = patch->getCellHighIndex();

    // Allocate the interpolated velocities
    CCVariable<Vector> oldCCVel;
    CCVariable<Vector> newCCVel;
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    new_dw->allocate(oldCCVel, d_lab->d_oldCCVelocityLabel, matlIndex, patch);
    new_dw->allocate(newCCVel, d_lab->d_newCCVelocityLabel, matlIndex, patch);
    new_dw->allocate(newCCUVel, d_lab->d_newCCUVelocityLabel, matlIndex, patch);
    new_dw->allocate(newCCVVel, d_lab->d_newCCVVelocityLabel, matlIndex, patch);
    new_dw->allocate(newCCWVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch);

    // Interpolate the FC velocity to the CC
    for (int kk = idxLo.z(); kk < idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj < idxHi.y(); ++jj) {
	for (int ii = idxLo.x(); ii < idxHi.x(); ++ii) {
	  
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  // old U velocity (linear interpolation)
	  double old_u = 0.5*(oldUVel[idx] + 
			      oldUVel[idxU]);
	  // new U velocity (linear interpolation)
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  
	  // old V velocity (linear interpolation)
	  double old_v = 0.5*(oldVVel[idx] +
			      oldVVel[idxV]);
	  // new V velocity (linear interpolation)
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  
	  // old W velocity (linear interpolation)
	  double old_w = 0.5*(oldWVel[idx] +
			      oldWVel[idxW]);
	  // new W velocity (linear interpolation)
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  // Add the data to the CC Velocity Variables
	  oldCCVel[idx] = Vector(old_u,old_v,old_w);
	  newCCVel[idx] = Vector(new_u,new_v,new_w);
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }

  // Put the calculated stuff into the new_dw
    new_dw->put(oldCCVel, d_lab->d_oldCCVelocityLabel, matlIndex, patch);
    new_dw->put(newCCVel, d_lab->d_newCCVelocityLabel, matlIndex, patch);
    new_dw->put(newCCUVel, d_lab->d_newCCUVelocityLabel, matlIndex, patch);
    new_dw->put(newCCVVel, d_lab->d_newCCVVelocityLabel, matlIndex, patch);
    new_dw->put(newCCWVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch);
  }
}

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
    int nofGhostCells = 0;

  // Get the new velocity
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> pressure;
    constCCVariable<double> mixtureFraction;
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    new_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    new_dw->get(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    new_dw->get(mixtureFraction, d_lab->d_scalarSPLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    
    constCCVariable<double> mixFracVariance;
    if (d_props->getNumMixStatVars() > 0) {
      new_dw->get(mixFracVariance, d_lab->d_scalarVarSPLabel, matlIndex, patch, 
		  Ghost::None, nofGhostCells);
    }
    
    constCCVariable<double> gasfraction;
    if (d_MAlab)
      new_dw->get(gasfraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch, 
		  Ghost::None, nofGhostCells);

    constCCVariable<double> temperature;
    if (d_enthalpySolve) 
      new_dw->get(temperature, d_lab->d_tempINLabel, matlIndex, patch, 
		  Ghost::None, nofGhostCells);

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
	if (d_props->getNumMixStatVars() > 0) {
	  cerr << "MixFracVariance: " << mixFracVariance[*iter] << endl;
	}
	if (d_MAlab)
	  cerr << "gas vol fraction: " << gasfraction[*iter] << endl;

      }
    }
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


