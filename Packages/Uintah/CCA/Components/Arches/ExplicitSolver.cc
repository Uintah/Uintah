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
  d_boundaryCondition->sched_correctOutletBC(sched, patches, matls);
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

  int Runge_Kutta_current_step;
  bool Runge_Kutta_last_step;

  Runge_Kutta_current_step = Arches::FIRST;
  #ifdef correctorstep
    Runge_Kutta_last_step = false;
  #else
    Runge_Kutta_last_step = true;
  #endif

  // check if filter is defined...only required if using dynamic or scalesimilarity models
#ifdef PetscFilter
  if (d_turbModel->getFilter()) {
    // if the matrix is not initialized
    if (!d_turbModel->getFilter()->isInitialized()) 
      d_turbModel->sched_initFilterMatrix(level, sched, patches, matls);
  }
#endif
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
      d_turbModel->sched_computeScalarVariance(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
    }
    d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
  }

  if (d_enthalpySolve)
    d_enthalpySolver->solvePred(sched, patches, matls);

  #ifdef correctorstep
    d_props->sched_computePropsPred(sched, patches, matls);
    d_props->sched_computeDenRefArrayPred(sched, patches, matls);
  #else
    d_props->sched_reComputeProps(sched, patches, matls);
    d_props->sched_computeDenRefArray(sched, patches, matls);
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
  d_pressSolver->solvePred(level, sched,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
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
    d_boundaryCondition->sched_predcomputePressureBC(sched, patches, matls);
  // Schedule an interpolation of the face centered velocity data 
    sched_interpolateFromFCToCCPred(sched, patches, matls);
  #else
    d_boundaryCondition->sched_lastcomputePressureBC(sched, patches, matls);
  // Schedule an interpolation of the face centered velocity data 
    sched_interpolateFromFCToCC(sched, patches, matls);
  #endif
    d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);

    sched_printTotalKE(sched, patches, matls, Runge_Kutta_current_step,
		       Runge_Kutta_last_step);

  #ifdef Runge_Kutta_3d
    // intermediate step for 3d order Runge-Kutta method
    Runge_Kutta_current_step = Arches::SECOND;
    Runge_Kutta_last_step = false;
    for (int index = 0;index < nofScalars; index ++) {
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve multiple scalars
      d_scalarSolver->solveInterm(sched, patches, matls, index);
    }
    if (d_reactingScalarSolver) {
      int index = 0;
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve multiple scalars
      d_reactingScalarSolver->solveInterm(sched, patches, matls, index);
    }
    if (nofScalarVars > 0) {
      for (int index = 0;index < nofScalarVars; index ++) {
      // in this case we're only solving for one scalarVar...but
      // the same subroutine can be used to solve multiple scalarVars
        d_turbModel->sched_computeScalarVariance(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
      }
      d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
    }
    if (d_enthalpySolve)
      d_enthalpySolver->solveInterm(sched, patches, matls);
    // same as corrector
    // Underrelaxation for density is done with initial density, not with
    // density from the previous substep
    d_props->sched_computePropsInterm(sched, patches, matls);
    d_props->sched_computeDenRefArrayInterm(sched, patches, matls);
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
    d_pressSolver->solveInterm(level, sched,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
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
      d_momSolver->solveInterm(sched, patches, matls, index);
    }
  
    d_boundaryCondition->sched_intermcomputePressureBC(sched, patches, matls);
  // Schedule an interpolation of the face centered velocity data 
    sched_interpolateFromFCToCCInterm(sched, patches, matls);
    d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);

    sched_printTotalKE(sched, patches, matls, Runge_Kutta_current_step,
		       Runge_Kutta_last_step);
  #endif

  #ifdef correctorstep
    // corrected step
    #ifdef Runge_Kutta_3d
       Runge_Kutta_current_step = Arches::THIRD;
    #else
       Runge_Kutta_current_step = Arches::SECOND;
    #endif
    Runge_Kutta_last_step = true;
    for (int index = 0;index < nofScalars; index ++) {
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve multiple scalars
      d_scalarSolver->solveCorr(sched, patches, matls, index);
    }
    if (d_reactingScalarSolver) {
      int index = 0;
      // in this case we're only solving for one scalar...but
      // the same subroutine can be used to solve multiple scalars
      d_reactingScalarSolver->solveCorr(sched, patches, matls, index);
    }

    if (nofScalarVars > 0) {
      for (int index = 0;index < nofScalarVars; index ++) {
      // in this case we're only solving for one scalarVar...but
      // the same subroutine can be used to solve multiple scalarVars
        d_turbModel->sched_computeScalarVariance(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
      }
      d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
    }
    if (d_enthalpySolve)
      d_enthalpySolver->solveCorr(sched, patches, matls);
    // same as corrector
    // Underrelaxation for density is done with initial density, not with
    // density from the previous substep
    d_props->sched_reComputeProps(sched, patches, matls);
    d_props->sched_computeDenRefArray(sched, patches, matls);
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
    d_pressSolver->solveCorr(level, sched,
			Runge_Kutta_current_step, Runge_Kutta_last_step);
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
  
    d_boundaryCondition->sched_lastcomputePressureBC(sched, patches, matls);
  // Schedule an interpolation of the face centered velocity data 
    sched_interpolateFromFCToCC(sched, patches, matls);
    d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
			Runge_Kutta_current_step, Runge_Kutta_last_step);

    sched_printTotalKE(sched, patches, matls, Runge_Kutta_current_step,
		       Runge_Kutta_last_step);
  #endif

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

  int Runge_Kutta_current_step;
  bool Runge_Kutta_last_step;

  Runge_Kutta_current_step = Arches::FIRST;
  #ifdef correctorstep
    Runge_Kutta_last_step = false;
  #else
    Runge_Kutta_last_step = true;
  #endif

  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP, 
  // densityCP, viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN

  sched_setInitialGuess(sched, patches, matls);

  d_props->sched_computePropsFirst_mm(sched, patches, matls);

  sched_dummySolve(sched, patches, matls);

  d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
					   Runge_Kutta_current_step, 
					   Runge_Kutta_last_step);
  
  d_pressSolver->sched_addHydrostaticTermtoPressure(sched, patches, matls);
 
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
  if (d_MAlab) 
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  else
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_pressureSPBCLabel,
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

  int nofScalarVars = d_props->getNumMixStatVars();
  // warning **only works for one scalarVar
  if (nofScalarVars > 0) {
    for (int ii = 0; ii < nofScalarVars; ii++) {
      tsk->requires(Task::OldDW, d_lab->d_scalarVarSPLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    }
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
// Schedule data copy for first time step of Multimaterial algorithm
// ****************************************************************************
void
ExplicitSolver::sched_dummySolve(SchedulerP& sched,
			       const PatchSet* patches,
			       const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::dataCopy",
			   this, &ExplicitSolver::dummySolve);
  int numGhostCells = 0;

  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_pressureINLabel,
		Ghost::None, numGhostCells);

  int nofScalars = d_props->getNumMixVars();
  for (int ii = 0; ii < nofScalars; ii++) {
    tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		  Ghost::None, numGhostCells);
  }

  int nofScalarVars = d_props->getNumMixStatVars();
  for (int ii = 0; ii < nofScalarVars; ii++) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarINLabel,
		  Ghost::None, numGhostCells);
  }

  if (d_reactingScalarSolve)
    tsk->requires(Task::NewDW, d_lab->d_reactscalarINLabel,
		  Ghost::None, numGhostCells);

  if (d_enthalpySolve) 
    tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel,
		  Ghost::None, numGhostCells);

  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);
  tsk->computes(d_lab->d_uVelRhoHatLabel);
  tsk->computes(d_lab->d_vVelRhoHatLabel);
  tsk->computes(d_lab->d_wVelRhoHatLabel);
  tsk->computes(d_lab->d_pressureSPBCLabel);
  tsk->computes(d_lab->d_pressurePSLabel);
  tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

  // warning **only works for one scalar
  for (int ii = 0; ii < nofScalars; ii++)
    tsk->computes(d_lab->d_scalarSPLabel);

  // warning **only works for one scalar variance
  for (int ii = 0; ii < nofScalarVars; ii++)
    tsk->computes(d_lab->d_scalarVarSPLabel);

  if (d_reactingScalarSolve)
    tsk->computes(d_lab->d_reactscalarSPLabel);

  if (d_enthalpySolve) {
    tsk->computes(d_lab->d_enthalpySPLabel);
    tsk->computes(d_lab->d_enthalpySPBCLabel);
  }

  tsk->computes(d_lab->d_uvwoutLabel);
  tsk->computes(d_lab->d_totalflowINLabel);
  tsk->computes(d_lab->d_totalflowOUTLabel);
  tsk->computes(d_lab->d_totalflowOUToutbcLabel);
  tsk->computes(d_lab->d_denAccumLabel);

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

  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel, 
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->computes(d_lab->d_oldCCVelocityLabel);
  tsk->computes(d_lab->d_newCCVelocityLabel);
  tsk->computes(d_lab->d_newCCUVelocityLabel);
  tsk->computes(d_lab->d_newCCVVelocityLabel);
  tsk->computes(d_lab->d_newCCWVelocityLabel);
      
  tsk->computes(d_lab->d_uVelRhoHat_CCLabel);
  tsk->computes(d_lab->d_vVelRhoHat_CCLabel);
  tsk->computes(d_lab->d_wVelRhoHat_CCLabel);

  tsk->computes(d_lab->d_kineticEnergyLabel);
  tsk->computes(d_lab->d_totalKineticEnergyLabel);
      
  sched->addTask(tsk, patches, matls);  
}

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
  tsk->requires(Task::NewDW, d_lab->d_pressureSPBCLabel,
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

  int nofScalarVars = d_props->getNumMixStatVars();
  if (nofScalarVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, 
    		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_tempINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

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
    constCCVariable<double> pressure;
    old_dw->get(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
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

    int nofScalarVars = d_props->getNumMixStatVars();
    StaticArray< constCCVariable<double> > scalarVar (nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	old_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, matlIndex, patch, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      }
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
    CCVariable<double> pressure_new;
    new_dw->allocateAndPut(pressure_new, d_lab->d_pressureINLabel, matlIndex, patch);
    pressure_new.copyData(pressure); // copy old into new

    SFCXVariable<double> uVelocity_new;
    new_dw->allocateAndPut(uVelocity_new, d_lab->d_uVelocityINLabel, matlIndex, patch);
    uVelocity_new.copyData(uVelocity); // copy old into new
    SFCYVariable<double> vVelocity_new;
    new_dw->allocateAndPut(vVelocity_new, d_lab->d_vVelocityINLabel, matlIndex, patch);
    vVelocity_new.copyData(vVelocity); // copy old into new
    SFCZVariable<double> wVelocity_new;
    new_dw->allocateAndPut(wVelocity_new, d_lab->d_wVelocityINLabel, matlIndex, patch);
    wVelocity_new.copyData(wVelocity); // copy old into new

    StaticArray<CCVariable<double> > scalar_new(nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->allocateAndPut(scalar_new[ii], d_lab->d_scalarINLabel, matlIndex, patch);
      scalar_new[ii].copyData(scalar[ii]); // copy old into new
    }

    StaticArray<CCVariable<double> > scalarVar_new(nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->allocateAndPut(scalarVar_new[ii], d_lab->d_scalarVarINLabel, matlIndex, patch);
	scalarVar_new[ii].copyData(scalarVar[ii]); // copy old into new
      }
    }

    constCCVariable<double> reactscalar;
    CCVariable<double> new_reactscalar;
    if (d_reactingScalarSolve) {
      old_dw->get(reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(new_reactscalar, d_lab->d_reactscalarINLabel, matlIndex,
		       patch);
      new_reactscalar.copyData(reactscalar);
    }


    CCVariable<double> new_enthalpy;
    if (d_enthalpySolve) {
      new_dw->allocateAndPut(new_enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch);
      new_enthalpy.copyData(enthalpy);
    }
    CCVariable<double> density_new;
    new_dw->allocateAndPut(density_new, d_lab->d_densityINLabel, matlIndex, patch);
    density_new.copyData(density); // copy old into new

    CCVariable<double> viscosity_new;
    new_dw->allocateAndPut(viscosity_new, d_lab->d_viscosityINLabel, matlIndex, patch);
    viscosity_new.copyData(viscosity); // copy old into new

    // Copy the variables into the new datawarehouse
    // allocateAndPut instead:
    /* new_dw->put(cellType_new, d_lab->d_cellTypeLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressure_new, d_lab->d_pressureINLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(uVelocity_new, d_lab->d_uVelocityINLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(vVelocity_new, d_lab->d_vVelocityINLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(wVelocity_new, d_lab->d_wVelocityINLabel, matlIndex, patch); */;

    for (int ii = 0; ii < nofScalars; ii++) {
      // allocateAndPut instead:
      /* new_dw->put(scalar_new[ii], d_lab->d_scalarINLabel, matlIndex, patch); */;
    }

    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	// allocateAndPut instead:
	/* new_dw->put(scalarVar_new[ii], d_lab->d_scalarVarINLabel, matlIndex, patch); */;
      }
    }
    if (d_reactingScalarSolve)
      // allocateAndPut instead:
      /* new_dw->put(new_reactscalar, d_lab->d_reactscalarINLabel, matlIndex, patch); */;
    if (d_enthalpySolve)
      // allocateAndPut instead:
      /* new_dw->put(new_enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(density_new, d_lab->d_densityINLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(viscosity_new, d_lab->d_viscosityINLabel, matlIndex, patch); */;
    if (d_MAlab)
      // allocateAndPut instead:
      /* new_dw->put(denMicro_new, d_lab->d_densityMicroINLabel, matlIndex, patch); */;
  }
}


// ****************************************************************************
// Actual Data Copy for first time step of MPMArches
// ****************************************************************************

void 
ExplicitSolver::dummySolve(const ProcessorGroup* ,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse* ,
			   DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // gets for old dw variables

    constSFCXVariable<double> uVelocity;
    new_dw->get(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    constSFCYVariable<double> vVelocity;
    new_dw->get(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    constSFCZVariable<double> wVelocity;
    new_dw->get(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    constCCVariable<double> pressure;
    new_dw->get(pressure, d_lab->d_pressureINLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    int nofScalars = d_props->getNumMixVars();
    StaticArray< constCCVariable<double> > scalar (nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->get(scalar[ii], d_lab->d_scalarINLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    int nofScalarVars = d_props->getNumMixStatVars();
    StaticArray< constCCVariable<double> > scalarVar (nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->get(scalarVar[ii], d_lab->d_scalarVarINLabel, matlIndex, patch, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      }
    }

    constCCVariable<double> reactscalar;
    if (d_reactingScalarSolve)
      new_dw->get(reactscalar, d_lab->d_reactscalarINLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    constCCVariable<double> enthalpy;
    if (d_enthalpySolve) 
      new_dw->get(enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    // allocates and puts for new dw variables

    SFCXVariable<double> uVelocity_new;
    new_dw->allocateAndPut(uVelocity_new, d_lab->d_uVelocitySPBCLabel, 
			   matlIndex, patch);
    uVelocity_new.copyData(uVelocity);

    SFCYVariable<double> vVelocity_new;
    new_dw->allocateAndPut(vVelocity_new, d_lab->d_vVelocitySPBCLabel, 
			   matlIndex, patch);
    vVelocity_new.copyData(vVelocity);

    SFCZVariable<double> wVelocity_new;
    new_dw->allocateAndPut(wVelocity_new, d_lab->d_wVelocitySPBCLabel, 
			   matlIndex, patch);
    wVelocity_new.copyData(wVelocity);

    SFCXVariable<double> uVelocityHat;
    new_dw->allocateAndPut(uVelocityHat, d_lab->d_uVelRhoHatLabel, 
			   matlIndex, patch);
    uVelocityHat.copyData(uVelocity);

    SFCYVariable<double> vVelocityHat;
    new_dw->allocateAndPut(vVelocityHat, d_lab->d_vVelRhoHatLabel, 
			   matlIndex, patch);
    vVelocityHat.copyData(vVelocity);

    SFCZVariable<double> wVelocityHat;
    new_dw->allocateAndPut(wVelocityHat, d_lab->d_wVelRhoHatLabel, 
			   matlIndex, patch);
    wVelocityHat.copyData(wVelocity);

    CCVariable<double> pressure_new;
    new_dw->allocateAndPut(pressure_new, d_lab->d_pressureSPBCLabel, 
			   matlIndex, patch);
    pressure_new.copyData(pressure);

    CCVariable<double> pressurePS_new;
    new_dw->allocateAndPut(pressurePS_new, d_lab->d_pressurePSLabel, 
			   matlIndex, patch);
    pressurePS_new.copyData(pressure);

    CCVariable<double> pressureNLSource;
    new_dw->allocateAndPut(pressureNLSource, d_lab->d_presNonLinSrcPBLMLabel, 
			   matlIndex, patch);
    pressureNLSource.initialize(0.);

    StaticArray<CCVariable<double> > scalar_new(nofScalars);
    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->allocateAndPut(scalar_new[ii], d_lab->d_scalarSPLabel, 
			     matlIndex, patch);
      scalar_new[ii].copyData(scalar[ii]); 
    }

    StaticArray<CCVariable<double> > scalarVar_new(nofScalarVars);
    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->allocateAndPut(scalarVar_new[ii], d_lab->d_scalarVarSPLabel, 
			       matlIndex, patch);
	scalarVar_new[ii].copyData(scalarVar[ii]);
      }
    }

    CCVariable<double> new_reactscalar;
    if (d_reactingScalarSolve) {
      new_dw->allocateAndPut(new_reactscalar, d_lab->d_reactscalarSPLabel, 
			     matlIndex, patch);
      new_reactscalar.copyData(reactscalar);
    }

    CCVariable<double> enthalpy_new;
    CCVariable<double> enthalpy_sp;
    if (d_enthalpySolve) {
      new_dw->allocateAndPut(enthalpy_sp, d_lab->d_enthalpySPLabel, 
			     matlIndex, patch);
      enthalpy_sp.copyData(enthalpy);

      new_dw->allocateAndPut(enthalpy_new, d_lab->d_enthalpySPBCLabel, 
			     matlIndex, patch);
      enthalpy_new.copyData(enthalpy);
    }

    cout << "DOING DUMMY SOLVE " << endl;

    double uvwout = 0.0;
    double flowIN = 0.0;
    double flowOUT = 0.0;
    double flowOUToutbc = 0.0;
    double denAccum = 0.0;

    // Copy the variables into the new datawarehouse
    /* not needed with allocateAndPut

    new_dw->put(uVelocity_new, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->put(vVelocity_new, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->put(wVelocity_new, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->put(uVelocityHat, d_lab->d_uVelRhoHatLabel, matlIndex, patch);
    new_dw->put(vVelocityHat, d_lab->d_vVelRhoHatLabel, matlIndex, patch);
    new_dw->put(wVelocityHat, d_lab->d_wVelRhoHatLabel, matlIndex, patch);
    new_dw->put(pressure_new, d_lab->d_pressureSPBCLabel, matlIndex, patch);
    new_dw->put(pressurePS_new, d_lab->d_pressurePSLabel, matlIndex, patch);
    new_dw->put(pressureNLSource, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

    for (int ii = 0; ii < nofScalars; ii++) {
      new_dw->put(scalar_new[ii], d_lab->d_scalarSPLabel, matlIndex, patch);
    }

    if (nofScalarVars > 0) {
      for (int ii = 0; ii < nofScalarVars; ii++) {
	new_dw->put(scalarVar_new[ii], d_lab->d_scalarVarSPLabel, matlIndex, patch);
      }
    }

    if (d_reactingScalarSolve)
      new_dw->put(new_reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch);

    if (d_enthalpySolve) {
      new_dw->put(enthalpy_sp, d_lab->d_enthalpySPLabel, matlIndex, patch);
      new_dw->put(enthalpy_new, d_lab->d_enthalpySPBCLabel, matlIndex, patch);
    }
    
    end of unnecessary puts */

    new_dw->put(delt_vartype(uvwout), d_lab->d_uvwoutLabel);
    new_dw->put(delt_vartype(flowIN), d_lab->d_totalflowINLabel);
    new_dw->put(delt_vartype(flowOUT), d_lab->d_totalflowOUTLabel);
    new_dw->put(delt_vartype(flowOUToutbc), d_lab->d_totalflowOUToutbcLabel);
    new_dw->put(delt_vartype(denAccum), d_lab->d_denAccumLabel);

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

    // Get the old velocity
    constSFCXVariable<double> oldUVel;
    constSFCYVariable<double> oldVVel;
    constSFCZVariable<double> oldWVel;
    new_dw->get(oldUVel, d_lab->d_uVelocityINLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(oldVVel, d_lab->d_vVelocityINLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(oldWVel, d_lab->d_wVelocityINLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    // Get the new velocity
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    constSFCXVariable<double> uHatVel_FCX;
    constSFCYVariable<double> vHatVel_FCY;
    constSFCZVariable<double> wHatVel_FCZ;      

    new_dw->get(uHatVel_FCX, d_lab->d_uVelRhoHatLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vHatVel_FCY, d_lab->d_vVelRhoHatLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wHatVel_FCZ, d_lab->d_wVelRhoHatLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    // Get the low and high index for the Cell Centered Variables
    IntVector idxLo = patch->getCellLowIndex();
    IntVector idxHi = patch->getCellHighIndex();

    // Allocate the interpolated velocities
    CCVariable<Vector> oldCCVel;
    CCVariable<Vector> newCCVel;
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    CCVariable<double> kineticEnergy;
    new_dw->allocateAndPut(oldCCVel, d_lab->d_oldCCVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(newCCVel, d_lab->d_newCCVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(newCCUVel, d_lab->d_newCCUVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(newCCVVel, d_lab->d_newCCVVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(newCCWVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(kineticEnergy, d_lab->d_kineticEnergyLabel, matlIndex, patch);

    CCVariable<double> uHatVel_CC;
    CCVariable<double> vHatVel_CC;
    CCVariable<double> wHatVel_CC;

    new_dw->allocateAndPut(uHatVel_CC, d_lab->d_uVelRhoHat_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vHatVel_CC, d_lab->d_vVelRhoHat_CCLabel, matlIndex, patch);
    new_dw->allocateAndPut(wHatVel_CC, d_lab->d_wVelRhoHat_CCLabel, matlIndex, patch);

    double total_kin_energy = 0.0;
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
	  
	  // uhat 
	  double uhat = 0.5*(uHatVel_FCX[idx] +
			     uHatVel_FCX[idxU]);
	  
	  // old V velocity (linear interpolation)
	  double old_v = 0.5*(oldVVel[idx] +
			      oldVVel[idxV]);
	  // new V velocity (linear interpolation)
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  
	  //vhat
	  double vhat = 0.5*(vHatVel_FCY[idx] +
			     vHatVel_FCY[idxV]);
	  
	  // old W velocity (linear interpolation)
	  double old_w = 0.5*(oldWVel[idx] +
			      oldWVel[idxW]);
	  // new W velocity (linear interpolation)
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  //what
	  double what = 0.5*(wHatVel_FCZ[idx] +
			     wHatVel_FCZ[idxW]);
	  
	  // Add the data to the CC Velocity Variables
	  oldCCVel[idx] = Vector(old_u,old_v,old_w);
	  newCCVel[idx] = Vector(new_u,new_v,new_w);
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
          kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
	  total_kin_energy += kineticEnergy[idx];

	  uHatVel_CC[idx] = uhat;
	  vHatVel_CC[idx] = vhat;
	  wHatVel_CC[idx] = what;
	  
	}
      }
    }
    new_dw->put(sum_vartype(total_kin_energy), d_lab->d_totalKineticEnergyLabel); 

  // Put the calculated stuff into the new_dw
    // allocateAndPut instead:
    /* new_dw->put(oldCCVel, d_lab->d_oldCCVelocityLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(newCCVel, d_lab->d_newCCVelocityLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(newCCUVel, d_lab->d_newCCUVelocityLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(newCCVVel, d_lab->d_newCCVVelocityLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(newCCWVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch); */;
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
    new_dw->get(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(mixtureFraction, d_lab->d_scalarSPLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    
    constCCVariable<double> mixFracVariance;
    if (d_props->getNumMixStatVars() > 0) {
      new_dw->get(mixFracVariance, d_lab->d_scalarVarSPLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    
    constCCVariable<double> gasfraction;
    if (d_MAlab)
      new_dw->get(gasfraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

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

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void 
ExplicitSolver::sched_interpolateFromFCToCCPred(SchedulerP& sched, 
						   const PatchSet* patches,
						   const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::interpFCToCCPred",
			   this, &ExplicitSolver::interpolateFromFCToCCPred);

  tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->computes(d_lab->d_newCCUVelocityPredLabel);
  tsk->computes(d_lab->d_newCCVVelocityPredLabel);
  tsk->computes(d_lab->d_newCCWVelocityPredLabel);
      
  tsk->computes(d_lab->d_totalKineticEnergyPredLabel);

  sched->addTask(tsk, patches, matls);

  
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector 
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void 
ExplicitSolver::interpolateFromFCToCCPred(const ProcessorGroup* ,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse*,
					     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the new velocity
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    new_dw->get(newUVel, d_lab->d_uVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newVVel, d_lab->d_vVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newWVel, d_lab->d_wVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    // Get the low and high index for the Cell Centered Variables
    IntVector idxLo = patch->getCellLowIndex();
    IntVector idxHi = patch->getCellHighIndex();

    // Allocate the interpolated velocities
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    new_dw->allocateAndPut(newCCUVel, d_lab->d_newCCUVelocityPredLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCVVel, d_lab->d_newCCVVelocityPredLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCWVel, d_lab->d_newCCWVelocityPredLabel,
			   matlIndex, patch);

    double total_kin_energy = 0.0;
    // Interpolate the FC velocity to the CC
    for (int kk = idxLo.z(); kk < idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj < idxHi.y(); ++jj) {
	for (int ii = idxLo.x(); ii < idxHi.x(); ++ii) {
	  
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  // new U velocity (linear interpolation)
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  
	  // new V velocity (linear interpolation)
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  
	  // new W velocity (linear interpolation)
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  // Add the data to the CC Velocity Variables
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	  total_kin_energy += (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
	}
      }
    }
    new_dw->put(sum_vartype(total_kin_energy), d_lab->d_totalKineticEnergyPredLabel); 
  }
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void 
ExplicitSolver::sched_interpolateFromFCToCCInterm(SchedulerP& sched, 
						   const PatchSet* patches,
						   const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::interpFCToCCInterm",
			   this, &ExplicitSolver::interpolateFromFCToCCInterm);

  tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->computes(d_lab->d_newCCUVelocityIntermLabel);
  tsk->computes(d_lab->d_newCCVVelocityIntermLabel);
  tsk->computes(d_lab->d_newCCWVelocityIntermLabel);
      
  tsk->computes(d_lab->d_totalKineticEnergyIntermLabel);

  sched->addTask(tsk, patches, matls);

  
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector 
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void 
ExplicitSolver::interpolateFromFCToCCInterm(const ProcessorGroup* ,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse*,
					     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the new velocity
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    new_dw->get(newUVel, d_lab->d_uVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newVVel, d_lab->d_vVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newWVel, d_lab->d_wVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    // Get the low and high index for the Cell Centered Variables
    IntVector idxLo = patch->getCellLowIndex();
    IntVector idxHi = patch->getCellHighIndex();

    // Allocate the interpolated velocities
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    new_dw->allocateAndPut(newCCUVel, d_lab->d_newCCUVelocityIntermLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCVVel, d_lab->d_newCCVVelocityIntermLabel,
			   matlIndex, patch);
    new_dw->allocateAndPut(newCCWVel, d_lab->d_newCCWVelocityIntermLabel,
			   matlIndex, patch);

    double total_kin_energy = 0.0;
    // Interpolate the FC velocity to the CC
    for (int kk = idxLo.z(); kk < idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj < idxHi.y(); ++jj) {
	for (int ii = idxLo.x(); ii < idxHi.x(); ++ii) {
	  
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  // new U velocity (linear interpolation)
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  
	  // new V velocity (linear interpolation)
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  
	  // new W velocity (linear interpolation)
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  // Add the data to the CC Velocity Variables
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	  total_kin_energy += (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
	}
      }
    }
    new_dw->put(sum_vartype(total_kin_energy), d_lab->d_totalKineticEnergyIntermLabel); 
  }
}
void 
ExplicitSolver::sched_printTotalKE(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   const int Runge_Kutta_current_step,
				   const bool Runge_Kutta_last_step)
{
  Task* tsk = scinew Task( "ExplicitSolver::printTotalKE",
			  this, &ExplicitSolver::printTotalKE, Runge_Kutta_current_step, Runge_Kutta_last_step);
  
  if (Runge_Kutta_last_step)
  tsk->requires(Task::NewDW, d_lab->d_totalKineticEnergyLabel);
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
  tsk->requires(Task::NewDW, d_lab->d_totalKineticEnergyPredLabel);
	 break;

	 case Arches::SECOND:
  tsk->requires(Task::NewDW, d_lab->d_totalKineticEnergyIntermLabel);
	 break;

	 default:
  throw InvalidValue("Invalid Runge-Kutta step in printTKE");
	 }
  }

  sched->addTask(tsk, patches, matls);
  
}
void 
ExplicitSolver::printTotalKE(const ProcessorGroup* ,
			     const PatchSubset* ,
			     const MaterialSubset*,
			     DataWarehouse*,
			     DataWarehouse* new_dw,
			     const int Runge_Kutta_current_step,
			     const bool Runge_Kutta_last_step)
{

  sum_vartype tke;
  if (Runge_Kutta_last_step)
  new_dw->get(tke, d_lab->d_totalKineticEnergyLabel);
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
  new_dw->get(tke, d_lab->d_totalKineticEnergyPredLabel);
	 break;

	 case Arches::SECOND:
  new_dw->get(tke, d_lab->d_totalKineticEnergyIntermLabel);
	 break;

	 default:
  throw InvalidValue("Invalid Runge-Kutta step in printTKE");
	 }
  }
  double total_kin_energy = tke;
  int me = d_myworld->myrank();
  if (me == 0)
     cerr << "Total kinetic energy " <<  total_kin_energy << endl;

}
