//----- Arches.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
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
#include <Packages/Uintah/CCA/Components/Arches/DynamicProcedure.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
using std::cerr;
using std::endl;

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
  d_boundaryCondition = 0;
  d_nlSolver = 0;
  d_physicalConsts = 0;
  d_calcReactingScalar = 0;
  d_calcEnthalpy =0;
#ifdef multimaterialform
  d_mmInterface = 0;
#endif
  nofTimeSteps = 0;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Arches::~Arches()
{
  delete d_lab;
  delete d_props;
  delete d_turbModel;
  delete d_boundaryCondition;
  delete d_nlSolver;
  delete d_physicalConsts;
#ifdef multimaterialform
  delete d_mmInterface;
#endif
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
  db->require("reacting_flow", d_reactingFlow);
  if (d_reactingFlow) {
    db->require("solve_reactingscalar", d_calcReactingScalar);
    db->require("solve_enthalpy", d_calcEnthalpy);
  }

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
  d_props = scinew Properties(d_lab, d_MAlab, d_reactingFlow, d_calcEnthalpy);
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
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "smagorinsky") 
    d_turbModel = scinew SmagorinskyModel(d_lab, d_MAlab, d_physicalConsts,
					  d_boundaryCondition);
  else if (turbModel == "dynamicprocedure") 
    d_turbModel = scinew DynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
					  d_boundaryCondition);
  else if (turbModel == "mixmodel") { 
    d_turbModel = scinew ScaleSimilarityModel(d_lab, d_MAlab, d_physicalConsts,
					      d_boundaryCondition);
  }
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
  if (d_turbModel)
    d_turbModel->problemSetup(db);

#ifdef PetscFilter
    d_filter = scinew Filter(d_lab, d_boundaryCondition, d_myworld);
    d_filter->problemSetup(db);
    d_turbModel->setFilter(d_filter);
#endif

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
					   d_turbModel, d_physicalConsts,
					   d_calcReactingScalar,
					   d_calcEnthalpy,
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
  // schedule init of cell type
  // require : NONE
  // compute : cellType
  d_boundaryCondition->sched_cellTypeInit(sched, patches, matls);

  // computing flow inlet areas
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

  // Compute props (output Varlabel have CP appended to them)
  // require : densitySP
  // require scalarSP
  // compute : densityCP
  d_props->sched_computeProps(sched, patches, matls);

  // Compute Turb subscale model (output Varlabel have CTS appended to them)
  // require : densityCP, viscosityIN, [u,v,w]VelocitySP
  // compute : viscosityCTS
  d_turbModel->sched_computeTurbSubmodel(level, sched, patches, matls);

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
    if (d_calcEnthalpy) {
      tsk->computes(d_lab->d_enthalpySPLabel); 
      tsk->computes(d_lab->d_radiationSRCINLabel);
      tsk->computes(d_lab->d_radiationFluxEINLabel); 
      tsk->computes(d_lab->d_radiationFluxWINLabel);
      tsk->computes(d_lab->d_radiationFluxNINLabel);
      tsk->computes(d_lab->d_radiationFluxSINLabel); 
      tsk->computes(d_lab->d_radiationFluxTINLabel);
      tsk->computes(d_lab->d_radiationFluxBINLabel); 
    }

    tsk->computes(d_lab->d_densityCPLabel);
    tsk->computes(d_lab->d_viscosityCTSLabel);
    tsk->computes(d_lab->d_oldDeltaTLabel);
    // for reacting flows save temperature and co2 
    if (d_MAlab)
      tsk->computes(d_lab->d_pressPlusHydroLabel);
    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

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
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_MAlab->KStabilityULabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->KStabilityVLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_MAlab->KStabilityWLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  tsk->computes(d_sharedState->get_delt_label());
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());


}

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

    constCCVariable<double> KStabilityU;
    constCCVariable<double> KStabilityV;
    constCCVariable<double> KStabilityW;
    
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(den, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(visc, d_lab->d_viscosityCTSLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    if (d_MAlab) {
      new_dw->get(KStabilityU, d_MAlab->KStabilityULabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(KStabilityV, d_MAlab->KStabilityULabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(KStabilityW, d_MAlab->KStabilityULabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex() + IntVector(1,1,1);
  // set density for the whole domain
    double delta_t = d_deltaT; // max value allowed
    double small_num = 1e-30;
    double delta_t2 = delta_t;
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double tmp_time;
// if statement to handle Kumar's wall with zero density
	  if (den[currCell] > 0.0) {
	    if (d_MAlab) {
	      tmp_time=Abs(uVelocity[currCell])/(cellinfo->sew[colX])+
		Abs(vVelocity[currCell])/(cellinfo->sns[colY])+
		Abs(wVelocity[currCell])/(cellinfo->stb[colZ])+
		(visc[currCell]/den[currCell])* 
		(1.0/(cellinfo->sew[colX]*cellinfo->sew[colX]) +
		 1.0/(cellinfo->sns[colY]*cellinfo->sns[colY]) +
		 1.0/(cellinfo->stb[colZ]*cellinfo->stb[colZ])) +
		//		2.0*KStabilityU[currCell] +
		//		2.0*KStabilityV[currCell] +
		//		2.0*KStabilityW[currCell] +
		small_num;
	    }
	    else {
	      tmp_time=Abs(uVelocity[currCell])/(cellinfo->sew[colX])+
		Abs(vVelocity[currCell])/(cellinfo->sns[colY])+
		Abs(wVelocity[currCell])/(cellinfo->stb[colZ])+
		(visc[currCell]/den[currCell])* 
		(1.0/(cellinfo->sew[colX]*cellinfo->sew[colX]) +
		 1.0/(cellinfo->sns[colY]*cellinfo->sns[colY]) +
		 1.0/(cellinfo->stb[colZ]*cellinfo->stb[colZ])) +
		small_num;
	    }

	  delta_t2=Min(1.0/tmp_time, delta_t2);
#if 0								  
	  delta_t2=Min(Abs(cellinfo->sew[colX]/
			  (uVelocity[currCell]+small_num)),delta_t2);
	  delta_t2=Min(Abs(cellinfo->sns[colY]/
			  (vVelocity[currCell]+small_num)), delta_t2);
	  delta_t2=Min(Abs(cellinfo->stb[colZ]/
			  (wVelocity[currCell]+small_num)), delta_t2);
#endif
	  }
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
    d_recompile = false;
  }
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
    CCVariable<double> pPlusHydro;
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

    if (d_calcEnthalpy) {
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);
      enthalpy.initialize(0.0);

      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
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

    }
    new_dw->allocateAndPut(density, d_lab->d_densityCPLabel, matlIndex, patch);
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch);

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
    for (int ii = 0; ii < d_nofScalars; ii++) {
      double scalVal = 0.0;
      fort_initscal(idxLo, idxHi, scalar[ii], scalVal);
    }
  }
}

// ****************************************************************************
// Function to return boolean for recompiling taskgraph
// ****************************************************************************
bool Arches::need_recompile(double time, double dt, 
			    const GridP& grid) {
 return d_recompile;
}
