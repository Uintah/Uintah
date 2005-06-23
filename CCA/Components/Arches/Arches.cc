//----- Arches.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ExplicitSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/PicardNonlinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
#include <Packages/Uintah/CCA/Components/Arches/IncDynamicProcedure.h>
#include <Packages/Uintah/CCA/Components/Arches/CompDynamicProcedure.h>
#include <Packages/Uintah/CCA/Components/Arches/CompLocalDynamicProcedure.h>
#include <Packages/Uintah/CCA/Components/Arches/OdtClosure.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <fstream>
using std::cerr;
using std::endl;

using std::string;
using namespace Uintah;
using namespace SCIRun;
#ifdef PetscFilter
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#endif

#include <Packages/Uintah/CCA/Components/Arches/fortran/initScal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/init_fort.h>

const int Arches::NDIM = 3;

// ****************************************************************************
// Actual constructor for Arches
// ****************************************************************************
Arches::Arches(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  d_lab = scinew ArchesLabel();
  d_MAlab = 0; // will be set by setMPMArchesLabel
  d_props = 0;
  d_turbModel = 0;
  d_initTurb = 0;
  d_scaleSimilarityModel = 0;
  d_boundaryCondition = 0;
  d_nlSolver = 0;
  d_physicalConsts = 0;
  d_calcReactingScalar = 0;
  d_calcThermalNOx = 0;
  d_calcEnthalpy =0;
#ifdef multimaterialform
  d_mmInterface = 0;
#endif
  nofTimeSteps = 0;
  init_timelabel_allocated = false;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Arches::~Arches()
{
  delete d_lab;
  delete d_props;
  delete d_turbModel;
  delete d_initTurb;
  delete d_scaleSimilarityModel;
  delete d_boundaryCondition;
  delete d_nlSolver;
  delete d_physicalConsts;
#ifdef multimaterialform
  delete d_mmInterface;
#endif
  if (init_timelabel_allocated)
    delete init_timelabel;
}

// ****************************************************************************
// problem set up
// ****************************************************************************
void 
Arches::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
  d_sharedState= sharedState;
  d_lab->setSharedState(sharedState);
  ArchesMaterial* mat= scinew ArchesMaterial();
  sharedState->registerArchesMaterial(mat);
  ProblemSpecP db = params->findBlock("CFD")->findBlock("ARCHES");
  // not sure, do we need to reduce and put in datawarehouse
  db->require("grow_dt", d_deltaT);
  db->require("variable_dt", d_variableTimeStep);
  db->require("transport_scalar", d_calcScalar);
  db->getWithDefault("set_initial_condition",d_set_initial_condition,false);
  if (d_set_initial_condition)
    db->require("init_cond_input_file", d_init_inputfile);
  if (d_calcScalar) {
    db->require("transport_reacting_scalar", d_calcReactingScalar);
    db->require("transport_enthalpy", d_calcEnthalpy);
    db->getWithDefault("solve_thermalnox", d_calcThermalNOx,false);
  }
  db->getWithDefault("turnonMixedModel",d_mixedModel,false);
  db->getWithDefault("recompileTaskgraph",d_recompile,false);

  // physical constant
  // physical constants
  d_physicalConsts = scinew PhysicalConstants();

  // ** BB 5/19/2000 ** For now read the Physical constants from the
  // ** BB 5/19/2000 ** For now read the Physical constants from the 
  // CFD-ARCHES block
  // for gravity, read it from shared state 
  //d_physicalConsts->problemSetup(params);
  d_physicalConsts->problemSetup(db);
  // read properties
  // d_MAlab = multimaterial arches common labels
  d_props = scinew Properties(d_lab, d_MAlab, d_physicalConsts, 
                              d_calcEnthalpy ,d_calcThermalNOx);
  d_props->problemSetup(db);
  d_nofScalars = d_props->getNumMixVars();
  d_nofScalarStats = d_props->getNumMixStatVars();

  // read turbulence mode
  // read turbulence model

  // read boundary
  d_boundaryCondition = scinew BoundaryCondition(d_lab, d_MAlab, d_physicalConsts,
						 d_props, d_calcReactingScalar,
						 d_calcEnthalpy);
  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->problemSetup(db);
  db->require("turbulence_model", turbModel);
  if (turbModel == "smagorinsky") 
    d_turbModel = scinew SmagorinskyModel(d_lab, d_MAlab, d_physicalConsts,
					  d_boundaryCondition);
  else  if (turbModel == "dynamicprocedure") 
    d_turbModel = scinew IncDynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
					  d_boundaryCondition);
  else if (turbModel == "compdynamicprocedure")
    d_turbModel = scinew CompDynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
					  d_boundaryCondition);
  else if (turbModel == "complocaldynamicprocedure") {
    d_initTurb = scinew CompLocalDynamicProcedure(d_lab, d_MAlab, d_physicalConsts, d_boundaryCondition); 
    d_turbModel = scinew CompLocalDynamicProcedure(d_lab, d_MAlab, d_physicalConsts, d_boundaryCondition);
  }
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
//  if (d_turbModel)
  d_turbModel->problemSetup(db);
  d_dynScalarModel = d_turbModel->getDynScalarModel();
  if (d_dynScalarModel)
    d_turbModel->setCombustionSpecifics(d_calcScalar, d_calcEnthalpy,
		                        d_calcReactingScalar);

#ifdef PetscFilter
    d_filter = scinew Filter(d_lab, d_boundaryCondition, d_myworld);
    d_filter->problemSetup(db);
    d_turbModel->setFilter(d_filter);
#endif

  d_turbModel->setMixedModel(d_mixedModel);
  if (d_mixedModel) {
    d_scaleSimilarityModel=scinew ScaleSimilarityModel(d_lab, d_MAlab, d_physicalConsts,
                                                       d_boundaryCondition);	
    d_scaleSimilarityModel->problemSetup(db);

    d_scaleSimilarityModel->setMixedModel(d_mixedModel);
#ifdef PetscFilter
    d_scaleSimilarityModel->setFilter(d_filter);
#endif
  }

  d_props->setBC(d_boundaryCondition);

  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard") {
    d_nlSolver = scinew PicardNonlinearSolver(d_lab, d_MAlab, d_props, 
					      d_boundaryCondition,
					      d_turbModel, d_physicalConsts,
					      d_calcReactingScalar,
					      d_calcEnthalpy,
					      d_myworld);
  }
  else if (nlSolver == "explicit") {
        d_nlSolver = scinew ExplicitSolver(d_lab, d_MAlab, d_props,
					   d_boundaryCondition,
					   d_turbModel, d_scaleSimilarityModel, 
					   d_physicalConsts,
					   d_calcReactingScalar,
					   d_calcEnthalpy,
				       	   d_calcThermalNOx,
					   d_myworld);
  }
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver);

  d_nlSolver->problemSetup(db);
  d_timeIntegratorType = d_nlSolver->getTimeIntegratorType();
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
Arches::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
  const PatchSet* patches= level->eachPatch();
  const MaterialSet* matls = d_sharedState->allArchesMaterials();

  // schedule the initialization of parameters
  // require : None
  // compute : [u,v,w]VelocityIN, pressureIN, scalarIN, densityIN,
  //           viscosityIN
  sched_paramInit(level, sched);
  if (d_set_initial_condition) {
    sched_readCCInitialCondition(level, sched);
    sched_interpInitialConditionToStaggeredGrid(level, sched);
  }
  // schedule init of cell type
  // require : NONE
  // compute : cellType
  d_boundaryCondition->sched_cellTypeInit(sched, patches, matls);

  // computing flow inlet areas
  if (d_boundaryCondition->getInletBC())
    d_boundaryCondition->sched_calculateArea(sched, patches, matls);

  // Set the profile (output Varlabel have SP appended to them)
  // require : densityIN,[u,v,w]VelocityIN
  // compute : densitySP, [u,v,w]VelocitySP, scalarSP
  d_boundaryCondition->sched_setProfile(sched, patches, matls);

  // if multimaterial, update celltype for mm intrusions for exact
  // initialization.
  // require: voidFrac_CC, cellType
  // compute: mmcellType, mmgasVolFrac

#ifdef ExactMPMArchesInitialize
  if (d_MAlab)
    d_boundaryCondition->sched_mmWallCellTypeInit_first(sched, patches, matls);
#endif

  IntVector periodic_vector = level->getPeriodicBoundaries();
  bool d_3d_periodic = (periodic_vector == IntVector(1,1,1));
  d_turbModel->set3dPeriodic(d_3d_periodic);
  d_props->set3dPeriodic(d_3d_periodic);
  // Compute props (output Varlabel have CP appended to them)
  // require : densitySP
  // require scalarSP
  // compute : densityCP
  init_timelabel = scinew TimeIntegratorLabel(d_lab,
		  			      TimeIntegratorStepType::FE);
  init_timelabel_allocated = true;
  d_props->sched_reComputeProps(sched, patches, matls,
				init_timelabel, true, true);

  d_boundaryCondition->sched_initInletBC(sched, patches, matls);

  sched_getCCVelocities(level, sched);
  // Compute Turb subscale model (output Varlabel have CTS appended to them)
  // require : densityCP, viscosityIN, [u,v,w]VelocitySP
  // compute : viscosityCTS

  if (!d_MAlab) {

    // check if filter is defined...
#ifdef PetscFilter
    if (d_turbModel->getFilter()) {
      // if the matrix is not initialized
      if (!d_turbModel->getFilter()->isInitialized()) 
	d_turbModel->sched_initFilterMatrix(level, sched, patches, matls);
    }
#endif

    if (d_mixedModel) {
      d_scaleSimilarityModel->sched_reComputeTurbSubmodel(sched, patches, matls,
                                                            init_timelabel);
    }

    if (turbModel == "complocaldynamicprocedure")
      d_initTurb->sched_initializeSmagCoeff(sched, patches, matls, init_timelabel);
    else
      d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls, init_timelabel);
  }

  // Computes velocities at apecified pressure b.c's
  // require : densityCP, pressureIN, [u,v,w]VelocitySP
  // compute : pressureSPBC, [u,v,w]VelocitySPBC
  if (d_boundaryCondition->getPressureBC()) 
    d_boundaryCondition->sched_computePressureBC(sched, patches, matls);
}

// ****************************************************************************
// schedule the initialization of parameters
// ****************************************************************************
void 
Arches::sched_paramInit(const LevelP& level,
			SchedulerP& sched)
{
    // primitive variable initialization
    Task* tsk = scinew Task( "Arches::paramInit",
			    this, &Arches::paramInit);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
    tsk->computes(d_lab->d_wVelocitySPBCLabel);
    tsk->computes(d_lab->d_uVelRhoHatLabel);
    tsk->computes(d_lab->d_vVelRhoHatLabel);
    tsk->computes(d_lab->d_wVelRhoHatLabel);
    tsk->computes(d_lab->d_newCCUVelocityLabel);
    tsk->computes(d_lab->d_newCCVVelocityLabel);
    tsk->computes(d_lab->d_newCCWVelocityLabel);
    tsk->computes(d_lab->d_pressurePSLabel);
    if (!((d_timeIntegratorType == "FE")||(d_timeIntegratorType == "BE")))
      tsk->computes(d_lab->d_pressurePredLabel);
    if (d_timeIntegratorType == "RK3SSP")
      tsk->computes(d_lab->d_pressureIntermLabel);

    for (int ii = 0; ii < d_nofScalars; ii++) 
      tsk->computes(d_lab->d_scalarSPLabel); // only work for 1 scalar
    if (d_nofScalarStats > 0) {
      for (int ii = 0; ii < d_nofScalarStats; ii++)
        tsk->computes(d_lab->d_scalarVarSPLabel); // only work for 1 scalarStat
      tsk->computes(d_lab->d_scalarDissSPLabel); // only work for 1 scalarStat
    }
    if (d_calcReactingScalar)
      tsk->computes(d_lab->d_reactscalarSPLabel);
    //Thermal NOx 
    if (d_calcThermalNOx)
      tsk->computes(d_lab->d_thermalnoxSPLabel);
    if (d_calcEnthalpy) {
      tsk->computes(d_lab->d_enthalpySPLabel); 
      tsk->computes(d_lab->d_radiationSRCINLabel);
      tsk->computes(d_lab->d_radiationFluxEINLabel); 
      tsk->computes(d_lab->d_radiationFluxWINLabel);
      tsk->computes(d_lab->d_radiationFluxNINLabel);
      tsk->computes(d_lab->d_radiationFluxSINLabel); 
      tsk->computes(d_lab->d_radiationFluxTINLabel);
      tsk->computes(d_lab->d_radiationFluxBINLabel); 
      tsk->computes(d_lab->d_abskgINLabel); 
    }

    tsk->computes(d_lab->d_densityCPLabel);
    tsk->computes(d_lab->d_viscosityCTSLabel);
    if (d_dynScalarModel) {
      if (d_calcScalar)
        tsk->computes(d_lab->d_scalarDiffusivityLabel);
      if (d_calcEnthalpy)
        tsk->computes(d_lab->d_enthalpyDiffusivityLabel);
      if (d_calcReactingScalar)
        tsk->computes(d_lab->d_reactScalarDiffusivityLabel);
    }
    tsk->computes(d_lab->d_oldDeltaTLabel);
    // for reacting flows save temperature and co2 
    if (d_MAlab) {
      tsk->computes(d_lab->d_pressPlusHydroLabel);
      tsk->computes(d_lab->d_mmgasVolFracLabel);
    }

    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}

// ****************************************************************************
// Actual initialization
// ****************************************************************************
void
Arches::paramInit(const ProcessorGroup* ,
		  const PatchSubset* patches,
		  const MaterialSubset*,
		  DataWarehouse* ,
		  DataWarehouse* new_dw)
{
    double old_delta_t = 0.0;
    new_dw->put(delt_vartype(old_delta_t), d_lab->d_oldDeltaTLabel);
  // ....but will only compute for computational domain
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;
    CCVariable<double> uVelocityCC;
    CCVariable<double> vVelocityCC;
    CCVariable<double> wVelocityCC;
    CCVariable<double> pressure;
    CCVariable<double> pressurePred;
    CCVariable<double> pressureInterm;
    StaticArray< CCVariable<double> > scalar(d_nofScalars);
    StaticArray< CCVariable<double> > scalarVar_new(d_nofScalarStats);
    CCVariable<double> scalarDiss_new;
    CCVariable<double> enthalpy;
    CCVariable<double> density;
    CCVariable<double> viscosity;
    CCVariable<double> scalarDiffusivity;
    CCVariable<double> enthalpyDiffusivity;
    CCVariable<double> reactScalarDiffusivity;
    CCVariable<double> pPlusHydro;
    CCVariable<double> mmgasVolFrac;
    std::cerr << "Material Index: " << matlIndex << endl;
    new_dw->allocateAndPut(uVelocityCC, d_lab->d_newCCUVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(vVelocityCC, d_lab->d_newCCVVelocityLabel, matlIndex, patch);
    new_dw->allocateAndPut(wVelocityCC, d_lab->d_newCCWVelocityLabel, matlIndex, patch);
    uVelocityCC.initialize(0.0);
    vVelocityCC.initialize(0.0);
    wVelocityCC.initialize(0.0);
    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->allocateAndPut(uVelRhoHat, d_lab->d_uVelRhoHatLabel, matlIndex, patch);
    new_dw->allocateAndPut(vVelRhoHat, d_lab->d_vVelRhoHatLabel, matlIndex, patch);
    new_dw->allocateAndPut(wVelRhoHat, d_lab->d_wVelRhoHatLabel, matlIndex, patch);
    uVelRhoHat.initialize(0.0);
    vVelRhoHat.initialize(0.0);
    wVelRhoHat.initialize(0.0);
    new_dw->allocateAndPut(pressure, d_lab->d_pressurePSLabel, matlIndex, patch);
    if (!((d_timeIntegratorType == "FE")||(d_timeIntegratorType == "BE"))) {
      new_dw->allocateAndPut(pressurePred, d_lab->d_pressurePredLabel,
			     matlIndex, patch);
      pressurePred.initialize(0.0);
    }
    if (d_timeIntegratorType == "RK3SSP") {
      new_dw->allocateAndPut(pressureInterm, d_lab->d_pressureIntermLabel,
			     matlIndex, patch);
      pressureInterm.initialize(0.0);
    }

    if (d_MAlab) {
      new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel, matlIndex, patch);
      pPlusHydro.initialize(0.0);
      new_dw->allocateAndPut(mmgasVolFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch);
      mmgasVolFrac.initialize(1.0);
    }
    // will only work for one scalar
    for (int ii = 0; ii < d_nofScalars; ii++) {
      new_dw->allocateAndPut(scalar[ii], d_lab->d_scalarSPLabel, matlIndex, patch);
    }
    if (d_nofScalarStats > 0) {
      for (int ii = 0; ii < d_nofScalarStats; ii++) {
        new_dw->allocateAndPut(scalarVar_new[ii], d_lab->d_scalarVarSPLabel, matlIndex, patch);
        scalarVar_new[ii].initialize(0.0);
      }
      new_dw->allocateAndPut(scalarDiss_new, d_lab->d_scalarDissSPLabel, matlIndex, patch);
      scalarDiss_new.initialize(0.0);  
    }
    CCVariable<double> reactscalar;
    if (d_calcReactingScalar) {
      new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarSPLabel,
		       matlIndex, patch);
      reactscalar.initialize(0.0);
    }
     // Thermal NOx 
    CCVariable<double> thermalnox;
    if (d_calcThermalNOx) {
      new_dw->allocateAndPut(thermalnox, d_lab->d_thermalnoxSPLabel,
                             matlIndex, patch);
      thermalnox.initialize(0.0);
    }

    if (d_calcEnthalpy) {
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);
      enthalpy.initialize(0.0);

      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
      CCVariable<double> abskg;
      CCVariable<double> radEnthalpySrc;;

      new_dw->allocateAndPut(radEnthalpySrc, d_lab->d_radiationSRCINLabel,
			     matlIndex, patch);
      radEnthalpySrc.initialize(0.0);

      new_dw->allocateAndPut(qfluxe, d_lab->d_radiationFluxEINLabel,
			     matlIndex, patch);
      qfluxe.initialize(0.0);
      new_dw->allocateAndPut(qfluxw, d_lab->d_radiationFluxWINLabel,
			     matlIndex, patch);
      qfluxw.initialize(0.0);
      new_dw->allocateAndPut(qfluxn, d_lab->d_radiationFluxNINLabel,
			     matlIndex, patch);
      qfluxn.initialize(0.0);
      new_dw->allocateAndPut(qfluxs, d_lab->d_radiationFluxSINLabel,
			     matlIndex, patch);
      qfluxs.initialize(0.0);
      new_dw->allocateAndPut(qfluxt, d_lab->d_radiationFluxTINLabel,
			     matlIndex, patch);
      qfluxt.initialize(0.0);
      new_dw->allocateAndPut(qfluxb, d_lab->d_radiationFluxBINLabel,
			     matlIndex, patch);
      qfluxb.initialize(0.0);
      new_dw->allocateAndPut(abskg, d_lab->d_abskgINLabel,
			     matlIndex, patch);
      abskg.initialize(0.0);

    }
    new_dw->allocateAndPut(density, d_lab->d_densityCPLabel, matlIndex, patch);
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch);
    if (d_dynScalarModel) {
      if (d_calcScalar)
        new_dw->allocateAndPut(scalarDiffusivity, d_lab->d_scalarDiffusivityLabel, matlIndex, patch);
      if (d_calcEnthalpy)
        new_dw->allocateAndPut(enthalpyDiffusivity, d_lab->d_enthalpyDiffusivityLabel, matlIndex, patch);
      if (d_calcReactingScalar)
        new_dw->allocateAndPut(reactScalarDiffusivity, d_lab->d_reactScalarDiffusivityLabel, matlIndex, patch);
    }  

  // ** WARNING **  this needs to be changed soon (6/9/2000)
    IntVector domLoU = uVelocity.getFortLowIndex();
    IntVector domHiU = uVelocity.getFortHighIndex();
    IntVector idxLoU = domLoU;
    IntVector idxHiU = domHiU;
    IntVector domLoV = vVelocity.getFortLowIndex();
    IntVector domHiV = vVelocity.getFortHighIndex();
    IntVector idxLoV = domLoV;
    IntVector idxHiV = domHiV;
    IntVector domLoW = wVelocity.getFortLowIndex();
    IntVector domHiW = wVelocity.getFortHighIndex();
    IntVector idxLoW = domLoW;
    IntVector idxHiW = domHiW;
    IntVector domLo = pressure.getFortLowIndex();
    IntVector domHi = pressure.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;

  //can read these values from input file 
    double uVal = 0.0, vVal = 0.0, wVal = 0.0;
    double pVal = 0.0, denVal = 0.0;
    double visVal = d_physicalConsts->getMolecularViscosity();
    fort_init(idxLoU, idxHiU, uVelocity, uVal, idxLoV, idxHiV,
	      vVelocity, vVal, idxLoW, idxHiW, wVelocity, wVal,
	      idxLo, idxHi, pressure, pVal, density, denVal,
	      viscosity, visVal);
    if (d_dynScalarModel) {
      if (d_calcScalar)
        scalarDiffusivity.initialize(visVal/0.4);
      if (d_calcEnthalpy)
        enthalpyDiffusivity.initialize(visVal/0.4);
      if (d_calcReactingScalar)
        reactScalarDiffusivity.initialize(visVal/0.4);
    }
    for (int ii = 0; ii < d_nofScalars; ii++) {
      double scalVal = 0.0;
      fort_initscal(idxLo, idxHi, scalar[ii], scalVal);
    }
  }
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
Arches::scheduleComputeStableTimestep(const LevelP& level,
				      SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "Arches::computeStableTimeStep",
			   this, &Arches::computeStableTimeStep);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }
  else {
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);


  tsk->computes(d_sharedState->get_delt_label());
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());


}

// ****************************************************************************
// actually compute stable time step
// ****************************************************************************
void 
Arches::computeStableTimeStep(const ProcessorGroup* ,
			      const PatchSubset* patches,
			      const MaterialSubset*,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> den;
    constCCVariable<double> visc;
    constCCVariable<int> cellType;

    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (d_MAlab) {
      new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->get(den, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }
    else {
      new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(den, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    new_dw->get(visc, d_lab->d_viscosityCTSLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);


    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int press_celltypeval = d_boundaryCondition->pressureCellType();
    int out_celltypeval = d_boundaryCondition->outletCellType();
    if ((xminus)&&((cellType[indexLow - IntVector(1,0,0)]==press_celltypeval)
		 ||(cellType[indexLow - IntVector(1,0,0)]==out_celltypeval)))
     indexLow = indexLow - IntVector(1,0,0);
    if ((yminus)&&((cellType[indexLow - IntVector(0,1,0)]==press_celltypeval)
		 ||(cellType[indexLow - IntVector(0,1,0)]==out_celltypeval)))
     indexLow = indexLow - IntVector(0,1,0);
    if ((zminus)&&((cellType[indexLow - IntVector(0,0,1)]==press_celltypeval)
		 ||(cellType[indexLow - IntVector(0,0,1)]==out_celltypeval)))
     indexLow = indexLow - IntVector(0,0,1);
    if (xplus)
     indexHigh = indexHigh + IntVector(1,0,0);
    if (yplus)
     indexHigh = indexHigh + IntVector(0,1,0);
    if (zplus)
     indexHigh = indexHigh + IntVector(0,0,1);

    double delta_t = d_deltaT; // max value allowed
    double small_num = 1e-30;
    double delta_t2 = delta_t;

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double tmp_time;

	  if (d_MAlab) {
	    int flag = 1;
	    int colXm = colX - 1;
	    int colXp = colX + 1;
	    int colYm = colY - 1;
	    int colYp = colY + 1;
	    int colZm = colZ - 1;
	    int colZp = colZ + 1;
	    if (colXm < indexLow.x()) colXm = indexLow.x();
	    if (colXp > indexHigh.x())colXp = indexHigh.x();
	    if (colYm < indexLow.y()) colYm = indexLow.y();
	    if (colYp > indexHigh.y())colYp = indexHigh.y();
	    if (colZm < indexLow.z()) colZm = indexLow.z();
	    if (colZp > indexHigh.z())colZp = indexHigh.z();
	    IntVector xMinusCell(colXm,colY,colZ);
	    IntVector xPlusCell(colXp,colY,colZ);
	    IntVector yMinusCell(colX,colYm,colZ);
	    IntVector yPlusCell(colX,colYp,colZ);
	    IntVector zMinusCell(colX,colY,colZm);
	    IntVector zPlusCell(colX,colY,colZp);
	    double uvel = uVelocity[currCell];
	    double vvel = vVelocity[currCell];
	    double wvel = wVelocity[currCell];

	    if (den[xMinusCell] < 1.0e-12) uvel=uVelocity[xPlusCell];
	    if (den[yMinusCell] < 1.0e-12) vvel=vVelocity[yPlusCell];
	    if (den[zMinusCell] < 1.0e-12) wvel=wVelocity[zPlusCell];
	    if (den[currCell] < 1.0e-12) flag = 0;
	    if ((den[xMinusCell] < 1.0e-12)&&(den[xPlusCell] < 1.0e-12)) flag = 0;
	    if ((den[yMinusCell] < 1.0e-12)&&(den[yPlusCell] < 1.0e-12)) flag = 0;
	    if ((den[zMinusCell] < 1.0e-12)&&(den[zPlusCell] < 1.0e-12)) flag = 0;

	    tmp_time=1.0;
	    if (flag != 0)
	      tmp_time=Abs(uvel)/(cellinfo->sew[colX])+
		Abs(vvel)/(cellinfo->sns[colY])+
		Abs(wvel)/(cellinfo->stb[colZ])+
		(visc[currCell]/den[currCell])* 
		(1.0/(cellinfo->sew[colX]*cellinfo->sew[colX]) +
		 1.0/(cellinfo->sns[colY]*cellinfo->sns[colY]) +
		 1.0/(cellinfo->stb[colZ]*cellinfo->stb[colZ])) +
		small_num;
	  }
	  else
	    tmp_time=Abs(uVelocity[currCell])/(cellinfo->sew[colX])+
	      Abs(vVelocity[currCell])/(cellinfo->sns[colY])+
	      Abs(wVelocity[currCell])/(cellinfo->stb[colZ])+
	      (visc[currCell]/den[currCell])* 
	      (1.0/(cellinfo->sew[colX]*cellinfo->sew[colX]) +
	       1.0/(cellinfo->sns[colY]*cellinfo->sns[colY]) +
	       1.0/(cellinfo->stb[colZ]*cellinfo->stb[colZ])) +
	      small_num;

	  delta_t2=Min(1.0/tmp_time, delta_t2);
	}
      }
    }


    if (d_variableTimeStep) {
      delta_t = delta_t2;
    }
    else {
      cout << " Courant condition for time step: " << delta_t2 << endl;
    }

    //    cout << "time step used: " << delta_t << endl;
    new_dw->put(delt_vartype(delta_t),  d_sharedState->get_delt_label()); 
  }
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
Arches::scheduleTimeAdvance( const LevelP& level, 
			     SchedulerP& sched,
			     int /*step*/, int /*nsteps*/ ) // AMR Parameters
{
  double time = d_lab->d_sharedState->getElapsedTime();
  nofTimeSteps++ ;
  if (d_MAlab) {
#ifndef ExactMPMArchesInitialize
    //    if (nofTimeSteps < 2) {
    if (time < 1.0E-10) {
      cout << "Calculating at time step = " << nofTimeSteps << endl;
      d_nlSolver->noSolve(level, sched);
    }
    else
      d_nlSolver->nonlinearSolve(level, sched);
#else
    d_nlSolver->nonlinearSolve(level, sched);
#endif
  }
  else {
    d_nlSolver->nonlinearSolve(level, sched);
  }
}

// ****************************************************************************
// Function to return boolean for recompiling taskgraph
// ****************************************************************************
bool Arches::needRecompile(double time, double dt, 
			    const GridP& grid) {
 return d_recompile;
}
// ****************************************************************************
// schedule reading of initial condition for velocity and pressure
// ****************************************************************************
void 
Arches::sched_readCCInitialCondition(const LevelP& level,
				     SchedulerP& sched)
{
    // primitive variable initialization
    Task* tsk = scinew Task( "Arches::readCCInitialCondition",
			    this, &Arches::readCCInitialCondition);
    tsk->modifies(d_lab->d_newCCUVelocityLabel);
    tsk->modifies(d_lab->d_newCCVVelocityLabel);
    tsk->modifies(d_lab->d_newCCWVelocityLabel);
    tsk->modifies(d_lab->d_pressurePSLabel);
    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}

// ****************************************************************************
// Actual read
// ****************************************************************************
void
Arches::readCCInitialCondition(const ProcessorGroup* ,
		  	       const PatchSubset* patches,
			       const MaterialSubset*,
	 		       DataWarehouse* ,
			       DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<double> uVelocityCC;
    CCVariable<double> vVelocityCC;
    CCVariable<double> wVelocityCC;
    CCVariable<double> pressure;
    new_dw->getModifiable(uVelocityCC, d_lab->d_newCCUVelocityLabel, matlIndex, patch);
    new_dw->getModifiable(vVelocityCC, d_lab->d_newCCVVelocityLabel, matlIndex, patch);
    new_dw->getModifiable(wVelocityCC, d_lab->d_newCCWVelocityLabel, matlIndex, patch);
    new_dw->getModifiable(pressure, d_lab->d_pressurePSLabel, matlIndex, patch);

    ifstream fd(d_init_inputfile.c_str());
    if(fd.fail()) {
      cout << " Unable to open the given input file " << d_init_inputfile << endl;
      exit(1);
    }
    int nx,ny,nz;
    fd >> nx >> ny >> nz;
    const Level* level = patch->getLevel();
    IntVector low, high;
    level->findCellIndexRange(low, high);
    IntVector range = high-low;//-IntVector(2,2,2);
    if (!(range == IntVector(nx,ny,nz))) {
      cout << "Wrong grid size in input file" << endl;
      exit(1);
    }
    double tmp;
    fd >> tmp >> tmp >> tmp;
    fd >> tmp >> tmp >> tmp;
    double uvel,vvel,wvel,pres;
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    for (int colZ = 1; colZ <= nz; colZ ++) {
      for (int colY = 1; colY <= ny; colY ++) {
	for (int colX = 1; colX <= nx; colX ++) {
	  IntVector currCell(colX-1, colY-1, colZ-1);

	  fd >> uvel >> vvel >> wvel >> pres >> tmp;
	  if ((currCell.x() <= idxHi.x() && currCell.y() <= idxHi.y() && currCell.z() <= idxHi.z()) &&
              (currCell.x() >= idxLo.x() && currCell.y() >= idxLo.y() && currCell.z() >= idxLo.z())) {
	    uVelocityCC[currCell] = 0.01*uvel;
	    vVelocityCC[currCell] = 0.01*vvel;
	    wVelocityCC[currCell] = 0.01*wvel;
	    pressure[currCell] = 0.1*pres;
          }
	}
      }
    }
    fd.close();  
  }
}


// ****************************************************************************
// schedule interpolation of initial condition for velocity to staggered
// ****************************************************************************
void 
Arches::sched_interpInitialConditionToStaggeredGrid(const LevelP& level,
						    SchedulerP& sched)
{
    // primitive variable initialization
    Task* tsk = scinew Task( "Arches::interpInitialConditionToStaggeredGrid",
			    this, &Arches::interpInitialConditionToStaggeredGrid);
    tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->modifies(d_lab->d_uVelocitySPBCLabel);
    tsk->modifies(d_lab->d_vVelocitySPBCLabel);
    tsk->modifies(d_lab->d_wVelocitySPBCLabel);
    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}

// ****************************************************************************
// Actual interpolation
// ****************************************************************************
void
Arches::interpInitialConditionToStaggeredGrid(const ProcessorGroup* ,
		  			      const PatchSubset* patches,
		  			      const MaterialSubset*,
		  			      DataWarehouse* ,
		 			      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<double> uVelocityCC;
    constCCVariable<double> vVelocityCC;
    constCCVariable<double> wVelocityCC;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    new_dw->get(uVelocityCC, d_lab->d_newCCUVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(vVelocityCC, d_lab->d_newCCVVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(wVelocityCC, d_lab->d_newCCWVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    
    IntVector idxLo, idxHi;

    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector xminusCell(colX-1, colY, colZ);

	  uVelocity[currCell] = 0.5*(uVelocityCC[currCell] +
			             uVelocityCC[xminusCell]);

	}
      }
    }
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector yminusCell(colX, colY-1, colZ);

	  vVelocity[currCell] = 0.5*(vVelocityCC[currCell] +
			             vVelocityCC[yminusCell]);

	}
      }
    }
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  IntVector zminusCell(colX, colY, colZ-1);

	  wVelocity[currCell] = 0.5*(wVelocityCC[currCell] +
			             wVelocityCC[zminusCell]);

	}
      }
    }
  }
}
// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC
// ****************************************************************************
void 
Arches::sched_getCCVelocities(const LevelP& level, SchedulerP& sched)
{
  string taskname =  "Arches::getCCVelocities";
  Task* tsk = scinew Task(taskname, this, &Arches::getCCVelocities);

  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
                Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->modifies(d_lab->d_newCCUVelocityLabel);
  tsk->modifies(d_lab->d_newCCVVelocityLabel);
  tsk->modifies(d_lab->d_newCCWVelocityLabel);
      
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable
// ****************************************************************************
void 
Arches::getCCVelocities(const ProcessorGroup* ,
		        const PatchSubset* patches,
		        const MaterialSubset*,
		        DataWarehouse* ,
		        DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 


    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    new_dw->getModifiable(newCCUVel, d_lab->d_newCCUVelocityLabel,
			   matlIndex, patch);
    new_dw->getModifiable(newCCVVel, d_lab->d_newCCVVelocityLabel,
			   matlIndex, patch);
    new_dw->getModifiable(newCCWVel, d_lab->d_newCCWVelocityLabel,
			   matlIndex, patch);
    newCCUVel.initialize(0.0);
    newCCVVel.initialize(0.0);
    newCCWVel.initialize(0.0);


    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
	for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {
	  
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
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
    // boundary conditions not to compute erroneous values in the case of ramping
    if (xminus) {
      int ii = idxLo.x()-1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
	for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idxU] +
			      newUVel[idxU]);
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
    if (xplus) {
      int ii =  idxHi.x()+1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
	for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idx]);
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
    if (yminus) {
      int jj = idxLo.y()-1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
	for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  double new_v = 0.5*(newVVel[idxV] +
			      newVVel[idxV]);
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
    if (yplus) {
      int jj =  idxHi.y()+1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
	for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idx]);
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idxW]);
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
    if (zminus) {
      int kk = idxLo.z()-1;
      for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
	for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  double new_w = 0.5*(newWVel[idxW] +
			      newWVel[idxW]);
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
    if (zplus) {
      int kk =  idxHi.z()+1;
      for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
	for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
	  IntVector idx(ii,jj,kk);
	  IntVector idxU(ii+1,jj,kk);
	  IntVector idxV(ii,jj+1,kk);
	  IntVector idxW(ii,jj,kk+1);
	  
	  double new_u = 0.5*(newUVel[idx] +
			      newUVel[idxU]);
	  double new_v = 0.5*(newVVel[idx] +
			      newVVel[idxV]);
	  double new_w = 0.5*(newWVel[idx] +
			      newWVel[idx]);
	  
	  newCCUVel[idx] = new_u;
	  newCCVVel[idx] = new_v;
	  newCCWVel[idx] = new_w;
	}
      }
    }
  }
}

double Arches::recomputeTimestep(double current_dt) {
  return d_nlSolver->recomputeTimestep(current_dt);
}
      
bool Arches::restartableTimesteps() {
  return d_nlSolver->restartableTimesteps();
}
