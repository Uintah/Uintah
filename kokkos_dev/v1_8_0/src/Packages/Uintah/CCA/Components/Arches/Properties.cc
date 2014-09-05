//----- Properties.cc --------------------------------------------------
#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/FlameletMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MeanMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>

#include <iostream>
using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties(const ArchesLabel* label, const MPMArchesLabel* MAlb,
		       bool reactingFlow, bool enthalpySolver):
  d_lab(label), d_MAlab(MAlb), d_reactingFlow(reactingFlow),
  d_enthalpySolve(enthalpySolver)
{
  d_flamelet = false;
  d_bc = 0;
#ifdef PetscFilter
  d_filter = 0;
#endif
  d_mixingModel = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
Properties::~Properties()
{
  delete d_mixingModel;
}

//****************************************************************************
// Problem Setup for Properties
//****************************************************************************
void 
Properties::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Properties");
  db->require("denUnderrelax", d_denUnderrelax);
  db->require("ref_point", d_denRef);
  db->require("radiation",d_radiationCalc);
  if (d_radiationCalc) {
    db->getWithDefault("discrete_ordinates",d_DORadiationCalc,true);
  }
  // read type of mixing model
  string mixModel;
  db->require("mixing_model",mixModel);
  if (mixModel == "coldFlowMixingModel")
    d_mixingModel = scinew ColdflowMixingModel();
  else if (mixModel == "pdfMixingModel")
    d_mixingModel = scinew PDFMixingModel();
  else if (mixModel == "meanMixingModel")
    d_mixingModel = scinew MeanMixingModel();
  else if (mixModel == "flameletModel") {
    d_mixingModel = scinew FlameletMixingModel();
    d_flamelet = true;
  }
  else
    throw InvalidValue("Mixing Model not supported" + mixModel);
  d_mixingModel->problemSetup(db);
  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = d_mixingModel->getNumMixVars();
  d_numMixStatVars = d_mixingModel->getNumMixStatVars();
  if (d_flamelet) {
    d_reactingFlow = false;
    d_radiationCalc = false;
    d_DORadiationCalc = false;
  }

}

//****************************************************************************
// compute density for inlet streams: only for cold streams
//****************************************************************************

void
Properties::computeInletProperties(const InletStream& inStream, 
				   Stream& outStream)
{
  if (dynamic_cast<const ColdflowMixingModel*>(d_mixingModel))
    d_mixingModel->computeProps(inStream, outStream);
  else if (dynamic_cast<const FlameletMixingModel*>(d_mixingModel)) {
    d_mixingModel->computeProps(inStream, outStream);
  }
  else {
    vector<double> mixVars = inStream.d_mixVars;
    outStream = d_mixingModel->speciesStateSpace(mixVars);
  }
}
  
//****************************************************************************
// Schedule the computation of properties
//****************************************************************************
void 
Properties::sched_computeProps(SchedulerP& sched, const PatchSet* patches,
			       const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::ComputeProps",
			  this,
			  &Properties::computeProps);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // requires scalars
  tsk->modifies(d_lab->d_densityINLabel);
  // will only work for one mixing variables
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }
  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityCPLabel);

#ifdef ExactMPMArchesInitialize
  if (d_MAlab) 
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
  else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
#else
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
#endif

  if (d_enthalpySolve)
    tsk->computes(d_lab->d_enthalpySPLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    tsk->computes(d_lab->d_cpINLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINLabel);

  }
  if (d_flamelet) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_fvtfiveINLabel);
    tsk->computes(d_lab->d_tfourINLabel);
    tsk->computes(d_lab->d_tfiveINLabel);
    tsk->computes(d_lab->d_tnineINLabel);
    tsk->computes(d_lab->d_qrgINLabel);
    tsk->computes(d_lab->d_qrsINLabel);
  }
  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
  }
  if (d_MAlab) {
#ifdef ExactMPMArchesInitialize
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, Ghost::None,
    		  Arches::ZEROGHOSTCELLS);
#endif
    tsk->computes(d_lab->d_densityMicroLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Schedule the computation of properties for the first actual time step
// in an MPMArches run
//****************************************************************************
void 
Properties::sched_computePropsFirst_mm(SchedulerP& sched, const PatchSet* patches,
				       const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::mmComputePropsFirst",
			  this,
			  &Properties::computePropsFirst_mm);

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  
  // Require densityIN from old_dw for consistency with
  // gets/requires of nonlinearSolve (we don't do anything 
  // with this densityIN)
  tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_reactingFlow) {
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_cpINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_co2INLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_enthalpyRXNLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_mixingModel->getNumRxnVars())
      tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_radiationCalc) {
    tsk->requires(Task::OldDW, d_lab->d_absorpINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_sootFVINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_DORadiationCalc) {
    tsk->requires(Task::OldDW, d_lab->d_h2oINLabel,
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

  tsk->computes(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityMicroLabel);

  tsk->modifies(d_lab->d_densityINLabel);

  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_cpINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINLabel);
  }

  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
    if (d_DORadiationCalc) {
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_radiationSRCINLabel);
    tsk->computes(d_lab->d_radiationFluxEINLabel);
    tsk->computes(d_lab->d_radiationFluxWINLabel);
    tsk->computes(d_lab->d_radiationFluxNINLabel);
    tsk->computes(d_lab->d_radiationFluxSINLabel);
    tsk->computes(d_lab->d_radiationFluxTINLabel);
    tsk->computes(d_lab->d_radiationFluxBINLabel);
    }
  }
  tsk->computes(d_lab->d_oldDeltaTLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::ReComputeProps",
			  this,
			  &Properties::reComputeProps);

  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    //    tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }

  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_drhodfCPLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_cpINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINLabel);
  }
  if (d_flamelet) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_fvtfiveINLabel);
    tsk->computes(d_lab->d_tfourINLabel);
    tsk->computes(d_lab->d_tfiveINLabel);
    tsk->computes(d_lab->d_tnineINLabel);
    tsk->computes(d_lab->d_qrgINLabel);
    tsk->computes(d_lab->d_qrsINLabel);


  }

  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
  }
  if (d_MAlab) 
    tsk->computes(d_lab->d_densityMicroLabel);

#ifdef scalarSolve_debug
  if (d_MAlab) 
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  numGhostCells);
#endif

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Schedule the computation of density reference array here
//****************************************************************************
void 
Properties::sched_computeDenRefArray(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls)

{

  // primitive variable initialization

  Task* tsk = scinew Task("Properties::computeDenRefArray",
			  this, &Properties::computeDenRefArray);


  tsk->requires(Task::NewDW, d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_denRefArrayLabel);

  if (d_MAlab) {

    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  sched->addTask(tsk, patches, matls);

}
  
//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
Properties::computeProps(const ProcessorGroup* pc,
			 const PatchSubset* patches,
			 const MaterialSubset*,
			 DataWarehouse*,
			 DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the cellType and density from the old datawarehouse

    constCCVariable<int> cellType;
    constCCVariable<double> density_old;
    CCVariable<double> density;
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);
    CCVariable<double> enthalpy;
    new_dw->getModifiable(density, d_lab->d_densityINLabel, 
		matlIndex, patch);

    if (d_enthalpySolve) {
      //constCCVariable<double> enthalpy_old;
      //new_dw->get(enthalpy_old, d_lab->d_enthalpySPBCLabel, 
      //	  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);
      enthalpy.initialize(0.0);
      //enthalpy.copyData(enthalpy_old);
    }
    
#ifdef ExactMPMArchesInitialize
    if (d_MAlab) {
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch, 
    		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    else {
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }
#else
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

    CCVariable<double> temperature;
    CCVariable<double> cp;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpyRXN;
    CCVariable<double> reactscalarSRC;
    if (d_reactingFlow) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->allocateAndPut(cp, d_lab->d_cpINLabel, matlIndex, patch);
      new_dw->allocateAndPut(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->allocateAndPut(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
      new_dw->allocateAndPut(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch);
      if (d_mixingModel->getNumRxnVars()) {
	new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			 matlIndex, patch);
	reactscalarSRC.initialize(0.0);
      }
    }
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    CCVariable<double> fvtfive;
    CCVariable<double> tfour;
    CCVariable<double> tfive;
    CCVariable<double> tnine;
    CCVariable<double> qrg;
    CCVariable<double> qrs;
    if (d_flamelet) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      new_dw->allocateAndPut(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->allocateAndPut(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
      new_dw->allocateAndPut(fvtfive, d_lab->d_fvtfiveINLabel, matlIndex, patch);
      new_dw->allocateAndPut(tfour, d_lab->d_tfourINLabel, matlIndex, patch);
      new_dw->allocateAndPut(tfive, d_lab->d_tfiveINLabel, matlIndex, patch);
      new_dw->allocateAndPut(tnine, d_lab->d_tnineINLabel, matlIndex, patch);
      new_dw->allocateAndPut(qrg, d_lab->d_qrgINLabel, matlIndex, patch);
      new_dw->allocateAndPut(qrs, d_lab->d_qrsINLabel, matlIndex, patch);

    }
    if (d_radiationCalc) {
      new_dw->allocateAndPut(absorption, d_lab->d_absorpINLabel, matlIndex, patch);
      new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }
    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    StaticArray<constCCVariable<double> > scalarVar(d_numMixStatVars);

    if (d_numMixStatVars > 0) {
    for (int ii = 0; ii < d_numMixStatVars; ii++)
      new_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    StaticArray<constCCVariable<double> > reactScalar(d_mixingModel->getNumRxnVars());
    
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->get(reactScalar[ii], d_lab->d_reactscalarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    //CCVariable<double> new_density;
    //new_dw->allocate(new_density, d_densityCPLabel, matlIndex, patch);

    // get multimaterial vars

    CCVariable<double> denMicro;
    constCCVariable<double> voidFraction;

    if (d_MAlab){
#ifdef ExactMPMArchesInitialize
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
      new_dw->allocateAndPut(denMicro, d_lab->d_densityMicroLabel,
		       matlIndex, patch);
    }
    
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    // construct an InletStream for input to the computeProps of mixingModel
    InletStream inStream(d_numMixingVars, d_mixingModel->getNumMixStatVars(),
		         d_mixingModel->getNumRxnVars());
    Stream outStream;

    // set density for the whole domain

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // Store current cell
	  IntVector currCell(colX, colY, colZ);

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f

	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {

	    inStream.d_mixVars[ii] = (scalar[ii])[currCell];

	  }

	  if (d_numMixStatVars > 0) {

	    for (int ii = 0; ii < d_numMixStatVars; ii++ ) {

	      inStream.d_mixVarVariance[ii] = (scalarVar[ii])[currCell];

	    }

	  }

	  if (d_mixingModel->getNumRxnVars() > 0) {
	    for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++ ) 
	      inStream.d_rxnVars[ii] = (reactScalar[ii])[currCell];
	  }
	  if (d_flamelet) {
	    if (colX >= 0)
	      inStream.d_axialLoc = colX;
	    else
	      inStream.d_axialLoc = 0;
	  }

	  if (!d_mixingModel->isAdiabatic())
	    inStream.d_enthalpy = 0.0;
	  d_mixingModel->computeProps(inStream, outStream);
	  double local_den = outStream.getDensity();
	  if (d_enthalpySolve)
	    enthalpy[currCell] = outStream.getEnthalpy();
	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    cp[currCell] = outStream.getCP();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    enthalpyRXN[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  if (d_flamelet) {
	    temperature[currCell] = outStream.getTemperature();
	    sootFV[currCell] = outStream.getSootFV();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    fvtfive[currCell] = outStream.getfvtfive();
	    tfour[currCell] = outStream.gettfour();
	    tfive[currCell] = outStream.gettfive();
	    tnine[currCell] = outStream.gettnine();
	    qrg[currCell] = outStream.getqrg();
	    qrs[currCell] = outStream.getqrs();


	  }
	  if (d_bc == 0)
	    throw InvalidValue("BoundaryCondition pointer not assigned");

	  if (d_MAlab) {
	    denMicro[currCell] = local_den;
#ifdef ExactMPMArchesInitialize
	    local_den *= voidFraction[currCell];
#endif
	  }
	  
	  if (cellType[currCell] != d_bc->wallCellType()) 
	    //	    density[currCell] = d_denUnderrelax*local_den +
	    //  (1.0-d_denUnderrelax)*density[currCell];
	    density[currCell] = local_den;
	}
      }
    }
    if ((d_bc->getIntrusionBC())&&(d_reactingFlow||d_flamelet))
      d_bc->intrusionTemperatureBC(pc, patch, cellType, temperature);
    if (patch->containsCell(d_denRef)) {

      double den_ref = density[d_denRef];

#ifdef ARCHES_DEBUG
      cerr << "density_ref " << den_ref << endl;
#endif

      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
    }

    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);
    
    // Write the computed density to the new data warehouse
    CCVariable<double> density_cp;
    new_dw->allocateAndPut(density_cp, d_lab->d_densityCPLabel, matlIndex, patch);
    density_cp.copyData(density);

  }
}
  
//****************************************************************************
// Actually compute the properties here for the first actual time step for
// MPMArches
//****************************************************************************
void 
Properties::computePropsFirst_mm(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  new_dw->put(delT, d_lab->d_oldDeltaTLabel);

  for (int p = 0; p < patches->size(); p++) {
 
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> denMicro;
    new_dw->get(denMicro, d_lab->d_densityMicroINLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);

    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    constCCVariable<double> voidFraction;
    new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    CCVariable<double> density;
    new_dw->allocateAndPut(density, d_lab->d_densityCPLabel, 
			   matlIndex, patch);

    CCVariable<double> densityIN;
    new_dw->getModifiable(densityIN, d_lab->d_densityINLabel, matlIndex, patch);

    // Get densityIN from old_dw for consistency with
    // gets/requires of nonlinearSolve (we don't do 
    // anything with this densityIN)

    CCVariable<double> densityIN_old;
    old_dw->getModifiable(densityIN_old, d_lab->d_densityINLabel, matlIndex, patch);

    CCVariable<double> denMicro_new;
    new_dw->allocateAndPut(denMicro_new, d_lab->d_densityMicroLabel, 
			   matlIndex, patch);
    denMicro_new.copyData(denMicro);

    constCCVariable<double> tempIN;
    constCCVariable<double> cpIN;
    constCCVariable<double> co2IN;
    constCCVariable<double> enthalpyRXN; 
    constCCVariable<double> reactScalarSrc;
    CCVariable<double> tempIN_new;
    CCVariable<double> cpIN_new;
    CCVariable<double> co2IN_new;
    CCVariable<double> enthalpyRXN_new;
    CCVariable<double> reactScalarSrc_new;
    if (d_reactingFlow) {
      old_dw->get(tempIN, d_lab->d_tempINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(cpIN, d_lab->d_cpINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(co2IN, d_lab->d_co2INLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      if (d_mixingModel->getNumRxnVars()) {
	old_dw->get(reactScalarSrc, d_lab->d_reactscalarSRCINLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      }

      new_dw->allocateAndPut(tempIN_new, d_lab->d_tempINLabel, 
			     matlIndex, patch);
      tempIN_new.copyData(tempIN);

      new_dw->allocateAndPut(cpIN_new, d_lab->d_cpINLabel, 
			     matlIndex, patch);
      cpIN_new.copyData(cpIN);

      new_dw->allocateAndPut(co2IN_new, d_lab->d_co2INLabel, 
			     matlIndex, patch);
      co2IN_new.copyData(co2IN);

      new_dw->allocateAndPut(enthalpyRXN_new, d_lab->d_enthalpyRXNLabel, 
			     matlIndex, patch);
      enthalpyRXN_new.copyData(enthalpyRXN);

      if (d_mixingModel->getNumRxnVars()) {
	new_dw->allocateAndPut(reactScalarSrc_new, d_lab->d_reactscalarSRCINLabel,
			       matlIndex, patch);
	reactScalarSrc_new.copyData(reactScalarSrc);
      }
    }

    constCCVariable<double> absorpIN;
    constCCVariable<double> sootFVIN;
    constCCVariable<double> h2oIN;
    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> radiationFluxEIN;
    constCCVariable<double> radiationFluxWIN;
    constCCVariable<double> radiationFluxNIN;
    constCCVariable<double> radiationFluxSIN;
    constCCVariable<double> radiationFluxTIN;
    constCCVariable<double> radiationFluxBIN;
    CCVariable<double> absorpIN_new;
    CCVariable<double> sootFVIN_new;
    CCVariable<double> h2oIN_new;
    CCVariable<double> radiationSRCIN_new;
    CCVariable<double> radiationFluxEIN_new;
    CCVariable<double> radiationFluxWIN_new;
    CCVariable<double> radiationFluxNIN_new;
    CCVariable<double> radiationFluxSIN_new;
    CCVariable<double> radiationFluxTIN_new;
    CCVariable<double> radiationFluxBIN_new;
    if (d_radiationCalc) {

      old_dw->get(absorpIN, d_lab->d_absorpINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      old_dw->get(sootFVIN, d_lab->d_sootFVINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->allocateAndPut(absorpIN_new, d_lab->d_absorpINLabel, 
		       matlIndex, patch);
      absorpIN_new.copyData(absorpIN);

      new_dw->allocateAndPut(sootFVIN_new, d_lab->d_sootFVINLabel,
		       matlIndex, patch);
      sootFVIN_new.copyData(sootFVIN);

      if (d_DORadiationCalc) {
      old_dw->get(h2oIN, d_lab->d_h2oINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationSRCIN, d_lab->d_radiationSRCINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationFluxEIN, d_lab->d_radiationFluxEINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationFluxWIN, d_lab->d_radiationFluxWINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationFluxNIN, d_lab->d_radiationFluxNINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationFluxSIN, d_lab->d_radiationFluxSINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationFluxTIN, d_lab->d_radiationFluxTINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(radiationFluxBIN, d_lab->d_radiationFluxBINLabel,
		  matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(h2oIN_new, d_lab->d_h2oINLabel, 
		       matlIndex, patch);
      h2oIN_new.copyData(h2oIN);
      new_dw->allocateAndPut(radiationSRCIN_new, d_lab->d_radiationSRCINLabel, 
		       matlIndex, patch);
      radiationSRCIN_new.copyData(radiationSRCIN);
      new_dw->allocateAndPut(radiationFluxEIN_new,
			     d_lab->d_radiationFluxEINLabel, matlIndex, patch);
      radiationFluxEIN_new.copyData(radiationFluxEIN);
      new_dw->allocateAndPut(radiationFluxWIN_new,
			     d_lab->d_radiationFluxWINLabel, matlIndex, patch);
      radiationFluxWIN_new.copyData(radiationFluxWIN);
      new_dw->allocateAndPut(radiationFluxNIN_new,
			     d_lab->d_radiationFluxNINLabel, matlIndex, patch);
      radiationFluxNIN_new.copyData(radiationFluxNIN);
      new_dw->allocateAndPut(radiationFluxSIN_new,
			     d_lab->d_radiationFluxSINLabel, matlIndex, patch);
      radiationFluxSIN_new.copyData(radiationFluxSIN);
      new_dw->allocateAndPut(radiationFluxTIN_new,
			     d_lab->d_radiationFluxTINLabel, matlIndex, patch);
      radiationFluxTIN_new.copyData(radiationFluxTIN);
      new_dw->allocateAndPut(radiationFluxBIN_new,
			     d_lab->d_radiationFluxBINLabel, matlIndex, patch);
      radiationFluxBIN_new.copyData(radiationFluxBIN);
      }
    }

    // no need for if (d_MAlab),  since this routine is only 
    // called if d_MAlab

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    // modify density for the whole domain by multiplying with
    // void fraction

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  IntVector currCell(colX, colY, colZ);

	  double local_den = denMicro[currCell]*voidFraction[currCell];

	  if (cellType[currCell] != d_bc->getMMWallId()) {
	    density[currCell] = local_den;
	  }
	  else{
	    density[currCell] = 0.0;
	  }
	}
      }
    }

    if (patch->containsCell(d_denRef)) {

      double den_ref = density[d_denRef];
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);

    }
    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);

    densityIN.copyData(density);
    
  }
}
  
//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::reComputeProps(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse*,
			   DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
  TAU_PROFILE_TIMER(input, "Input", "[Properties::reCompute::input]" , TAU_USER);
  TAU_PROFILE_TIMER(compute, "Compute", "[Properties::reCompute::compute]" , TAU_USER);
  TAU_PROFILE_TIMER(mixing, "Mixing", "[Properties::reCompute::mixing]" , TAU_USER);

  TAU_PROFILE_START(input);
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the CCVariable (density) from the old datawarehouse
    // just write one function for computing properties
    double start_mixTime = Time::currentSeconds();
    constCCVariable<double> density;
    constCCVariable<double> voidFraction;
    CCVariable<double> temperature;
    CCVariable<double> cp;
    CCVariable<double> new_density;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpy;
    CCVariable<double> reactscalarSRC;
    CCVariable<double> drhodf;
    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		Ghost::None, 0);
    if (d_reactingFlow) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->allocateAndPut(cp, d_lab->d_cpINLabel, matlIndex, patch);
      new_dw->allocateAndPut(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->allocateAndPut(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpyRXNLabel, matlIndex, patch);
      if (d_mixingModel->getNumRxnVars()) {
	new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			 matlIndex, patch);
	reactscalarSRC.initialize(0.0);
      }
    }
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    CCVariable<double> fvtfive;
    CCVariable<double> tfour;
    CCVariable<double> tfive;
    CCVariable<double> tnine;
    CCVariable<double> qrg;
    CCVariable<double> qrs;
    if (d_flamelet) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      sootFV.initialize(0.0);
      new_dw->allocateAndPut(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->allocateAndPut(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
      new_dw->allocateAndPut(fvtfive, d_lab->d_fvtfiveINLabel, matlIndex, patch);
      new_dw->allocateAndPut(tfour, d_lab->d_tfourINLabel, matlIndex, patch);
      new_dw->allocateAndPut(tfive, d_lab->d_tfiveINLabel, matlIndex, patch);
      new_dw->allocateAndPut(tnine, d_lab->d_tnineINLabel, matlIndex, patch);
      new_dw->allocateAndPut(qrg, d_lab->d_qrgINLabel, matlIndex, patch);
      new_dw->allocateAndPut(qrs, d_lab->d_qrsINLabel, matlIndex, patch);



    }
    if (d_radiationCalc) {
      new_dw->allocateAndPut(absorption, d_lab->d_absorpINLabel, matlIndex, patch);
      new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);

    constCCVariable<double> enthalpy_comp;
    CCVariable<double> denMicro;
    //    CCVariable<double> denMicro_old;

    new_dw->allocateAndPut(drhodf, d_lab->d_drhodfCPLabel, matlIndex, patch);
    drhodf.initialize(0.0);

    new_dw->allocateAndPut(new_density, d_lab->d_densityCPLabel, matlIndex, patch);
    new_dw->get(density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_density.copyData(density);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      //new_dw->get(denMicro_old, d_lab->d_densityMicroINLabel, 
      //	  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch);
      //denMicro.copyData(denMicro_old);
    }

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    StaticArray<constCCVariable<double> > scalarVar(d_numMixStatVars);

    if (d_numMixStatVars > 0) {
      for (int ii = 0; ii < d_numMixStatVars; ii++)
	new_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    StaticArray<constCCVariable<double> > reactScalar(d_mixingModel->getNumRxnVars());
    
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->get(reactScalar[ii], d_lab->d_reactscalarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }


    if (!(d_mixingModel->isAdiabatic()))
      new_dw->get(enthalpy_comp, d_lab->d_enthalpySPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

#ifdef ARCHES_DEBUG
    IntVector test(6,9,9);
    cout << "printing test "<<test<<endl;
#endif
    TAU_PROFILE_STOP(input);
    TAU_PROFILE_START(compute);
    InletStream inStream(d_numMixingVars,
		         d_mixingModel->getNumMixStatVars(),
		         d_mixingModel->getNumRxnVars());
    Stream outStream;
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);
	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {

	    inStream.d_mixVars[ii] = (scalar[ii])[currCell];

#ifdef ARCHES_DEBUG
	    if ((colX==6) && (colY==9) && (colZ==9))
	      cerr << "Mixture Vars at test = " << (scalar[ii])[currCell];
#endif

	  }

	  if (d_numMixStatVars > 0) {

	    for (int ii = 0; ii < d_numMixStatVars; ii++ ) {

	      inStream.d_mixVarVariance[ii] = (scalarVar[ii])[currCell];

	    }

	  }

	  if (d_mixingModel->getNumRxnVars() > 0) {
	    for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++ ) 
	      inStream.d_rxnVars[ii] = (reactScalar[ii])[currCell];
	  }

	  // after computing variance get that too, for the time being setting the 
	  // value to zero
	  //	  inStream.d_mixVarVariance[0] = 0.0;
	  // currently not using any reaction progress variables

	  if ((!d_mixingModel->isAdiabatic()))
	      //	      &&(cellType[currCell] != d_bc->getIntrusionID()))
	    inStream.d_enthalpy = enthalpy_comp[currCell];
	  else
	    inStream.d_enthalpy = 0.0;
	  if (d_flamelet) {
	    if (colX >= 0)
	      inStream.d_axialLoc = colX;
	    else
	      inStream.d_axialLoc = 0;
	  }

  TAU_PROFILE_START(mixing);
	  d_mixingModel->computeProps(inStream, outStream);
  TAU_PROFILE_STOP(mixing);
	  double local_den = outStream.getDensity();
	  drhodf[currCell] = outStream.getdrhodf();
	  if (d_flamelet) {
	    temperature[currCell] = outStream.getTemperature();
	    sootFV[currCell] = outStream.getSootFV();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    fvtfive[currCell] = outStream.getfvtfive();
	    tfour[currCell] = outStream.gettfour();
	    tfive[currCell] = outStream.gettfive();
	    tnine[currCell] = outStream.gettnine();
	    qrg[currCell] = outStream.getqrg();
	    qrs[currCell] = outStream.getqrs();



	  }
	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    cp[currCell] = outStream.getCP();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    enthalpy[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  if (d_radiationCalc) {
	    // bc is the mass-atoms 0f carbon per mas of reactnat mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    // optical path length
	    double opl = 3.0;
	    if (d_mixingModel->getNumRxnVars()) 
	      sootFV[currCell] = outStream.getSootFV();
	    else {
	      if (temperature[currCell] > 1000) {
		double bc = inStream.d_mixVars[0]*(84.0/100.0)*local_den;
		double c3 = 0.1;
		double rhosoot = 1950.0;
		double cmw = 12.0;

		double factor = 0.01;
		if (inStream.d_mixVars[0] > 0.1)
		  sootFV[currCell] = c3*bc*cmw/rhosoot*factor;
		else
		  sootFV[currCell] = 0.0;
	      }
	      else 
		sootFV[currCell] = 0.0;
	    }
	    absorption[currCell] = 0.01+ Min(0.5,(4.0/opl)*log(1.0+350.0*
				   sootFV[currCell]*temperature[currCell]*opl));
	    }
	  if (d_MAlab) {
	    denMicro[IntVector(colX, colY, colZ)] = local_den;
	    local_den *= voidFraction[currCell];

	  }

	  // no under-relaxation for MPMArches
	  if (d_MAlab) {
	    new_density[IntVector(colX, colY, colZ)] = local_den;
	  }
	  else {
	    new_density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
	      (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
	  }
	}
      }
    }
    // Write the computed density to the new data warehouse
#ifdef ARCHES_PRES_DEBUG
    // Testing if correct values have been put
    cerr << " AFTER COMPUTE PROPERTIES " << endl;
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    density.print(cerr);
#endif

#ifdef scalarSolve_debug

    if (d_MAlab) {
      constCCVariable<int> cellType;
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
    		  Ghost::None, 0);

      cerr << " NEW DENSITY VALUES " << endl;
      for (int ii = 5; ii <= 9; ii++) {
	for (int jj = 7; jj <= 12; jj++) {
	  for (int kk = 7; kk <= 12; kk++) {
	    cerr.width(14);
	    cerr << " point coordinates "<< ii << " " << jj << " " << kk ;
	    cerr << " new density = " << new_density[IntVector(ii,jj,kk)] ; 
	    cerr << " cellType = " << cellType[IntVector(ii,jj,kk)] ; 
	    cerr << " void fraction = " << voidFraction[IntVector(ii,jj,kk)] << endl; 
	  }
	}
      }

    }
#endif

    if (patch->containsCell(d_denRef)) {
      double den_ref = new_density[d_denRef];
      cerr << "density_ref " << den_ref << endl;
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
    }
    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);
    if ((d_bc->getIntrusionBC())&&(d_reactingFlow||d_flamelet))
      d_bc->intrusionTemperatureBC(pc, patch, cellType, temperature);

    if (pc->myrank() == 0)
      cerr << "Time in the Mixing Model: " << Time::currentSeconds()-start_mixTime << " seconds\n";

  TAU_PROFILE_STOP(compute);
  }
}

//****************************************************************************
// Actually calculate the density reference array here
//****************************************************************************

void 
Properties::computeDenRefArray(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset*,
			       DataWarehouse*,
			       DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> denRefArray;
    constCCVariable<double> voidFraction;


    sum_vartype den_ref_var;
    new_dw->get(den_ref_var, d_lab->d_refDensity_label);

    double den_Ref = den_ref_var;

    //cerr << "getdensity_ref " << den_Ref << endl;

    if (d_MAlab) {

      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    new_dw->allocateAndPut(denRefArray, d_lab->d_denRefArrayLabel, 
		     matlIndex, patch);
		
    denRefArray.initialize(den_Ref);

    if (d_MAlab) {

      for (CellIterator iter = patch->getCellIterator();
	   !iter.done();iter++){

	denRefArray[*iter]  *= voidFraction[*iter];

      }
    }

    // allocateAndPut instead:
    /* new_dw->put(denRefArray, d_lab->d_denRefArrayLabel,
		matlIndex, patch); */;

  }
}


//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_computePropsPred(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::computePropsPred",
			  this,
			  &Properties::computePropsPred);

  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    //    tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }

  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarPredLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpyPredLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);


  tsk->computes(d_lab->d_refDensityPred_label);
  tsk->computes(d_lab->d_densityPredLabel);
  tsk->computes(d_lab->d_drhodfPredLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINPredLabel);
    tsk->computes(d_lab->d_co2INPredLabel);
    tsk->computes(d_lab->d_h2oINPredLabel);
    tsk->computes(d_lab->d_enthalpyRXNPredLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINPredLabel);
  }
  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINPredLabel);
    tsk->computes(d_lab->d_sootFVINPredLabel);
  }
  if (d_MAlab) 
    tsk->computes(d_lab->d_densityMicroLabel);
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::computePropsPred(const ProcessorGroup*,
			     const PatchSubset* patches,
			     const MaterialSubset*,
			     DataWarehouse*,
			     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the CCVariable (density) from the old datawarehouse
    // just write one function for computing properties

    constCCVariable<double> density;
    constCCVariable<double> voidFraction;
    CCVariable<double> temperature;
    CCVariable<double> new_density;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpy;
    CCVariable<double> reactscalarSRC;
    CCVariable<double> drhodf;

    if (d_reactingFlow) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINPredLabel, matlIndex, patch);
      new_dw->allocateAndPut(co2, d_lab->d_co2INPredLabel, matlIndex, patch);
      new_dw->allocateAndPut(h2o, d_lab->d_h2oINPredLabel, matlIndex, patch);
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpyRXNPredLabel, matlIndex, patch);
      if (d_mixingModel->getNumRxnVars()) {
	new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINPredLabel,
			 matlIndex, patch);
	reactscalarSRC.initialize(0.0);
      }
    }
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    if (d_radiationCalc) {
      new_dw->allocateAndPut(absorption, d_lab->d_absorpINPredLabel, matlIndex, patch);
      new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINPredLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }
 
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);
    //constCCVariable<double> denMicro;
    constCCVariable<double> enthalpy_comp;

    new_dw->allocateAndPut(drhodf, d_lab->d_drhodfPredLabel, matlIndex, patch);
    drhodf.initialize(0.0);

    new_dw->allocateAndPut(new_density, d_lab->d_densityPredLabel, matlIndex, patch);
    new_dw->get(density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_density.copyData(density);
    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      //new_dw->get(denMicro, d_lab->d_densityMicroINLabel, 
      //	  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarPredLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    StaticArray<constCCVariable<double> > scalarVar(d_numMixStatVars);

    if (d_numMixStatVars > 0) {
      for (int ii = 0; ii < d_numMixStatVars; ii++)
	new_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }


    StaticArray<constCCVariable<double> > reactScalar(d_mixingModel->getNumRxnVars());
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->get(reactScalar[ii], d_lab->d_reactscalarPredLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    if (!(d_mixingModel->isAdiabatic()))
      new_dw->get(enthalpy_comp, d_lab->d_enthalpyPredLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    //    voidFraction.print(cerr);
    // set density for the whole domain
    InletStream inStream(d_numMixingVars, 
		         d_mixingModel->getNumMixStatVars(),
		         d_mixingModel->getNumRxnVars());
    Stream outStream;

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);

	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {

	    inStream.d_mixVars[ii] = (scalar[ii])[currCell];

	  }

	  if (d_numMixStatVars > 0) {

	    for (int ii = 0; ii < d_numMixStatVars; ii++ ) {

	      inStream.d_mixVarVariance[ii] = (scalarVar[ii])[currCell];

	    }

	  }

	  if (d_mixingModel->getNumRxnVars() > 0) {
	    for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++ ) 
	      inStream.d_rxnVars[ii] = (reactScalar[ii])[currCell];
	  }

	  // after computing variance get that too, for the time being setting the 
	  // value to zero
	  //	  inStream.d_mixVarVariance[0] = 0.0;
	  // currently not using any reaction progress variables

	  if (!d_mixingModel->isAdiabatic())
	    inStream.d_enthalpy = enthalpy_comp[currCell];
	  d_mixingModel->computeProps(inStream, outStream);
	  double local_den = outStream.getDensity();
	  drhodf[currCell] = outStream.getdrhodf();

	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    enthalpy[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  if (d_radiationCalc) {
	    // bc is the mass-atoms 0f carbon per mas of reactnat mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    // optical path length
	    double opl = 3.0;
	    if (d_mixingModel->getNumRxnVars()) 
	      sootFV[currCell] = outStream.getSootFV();
	    else {
	      if (temperature[currCell] > 1000) {
	        double bc = inStream.d_mixVars[0]*(84.0/100.0)*local_den;
	        double c3 = 0.1;
	        double rhosoot = 1950.0;
	        double cmw = 12.0;
	      
	        double factor = 0.01;
	        if (inStream.d_mixVars[0] > 0.1)
		  sootFV[currCell] = c3*bc*cmw/rhosoot*factor;
	        else
		  sootFV[currCell] = 0.0;
	      }
	      else 
	        sootFV[currCell] = 0.0;
	    }
	    absorption[currCell] = 0.01+ Min(0.5,(4.0/opl)*log(1.0+350.0*
				   sootFV[currCell]*temperature[currCell]*opl));
	  }

	  if (d_MAlab) {
	    //denMicro[IntVector(colX, colY, colZ)] = local_den;
	    //	    if (voidFraction[currCell] > 0.01)
	    local_den *= voidFraction[currCell];

	  }
	  
	  new_density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
	    (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
	}
      }
    }
    // Write the computed density to the new data warehouse
#if 0
    //#ifdef ARCHES_PRES_DEBUG
    // Testing if correct values have been put
    cerr << " AFTER COMPUTE PROPERTIES " << endl;
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    density.print(cerr);
#endif
    if (patch->containsCell(d_denRef)) {
      double den_ref = new_density[d_denRef];
      cerr << "density_ref " << den_ref << endl;
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensityPred_label);
    }
    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensityPred_label);

    // allocateAndPut instead:
    /* new_dw->put(new_density, d_lab->d_densityPredLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(drhodf, d_lab->d_drhodfPredLabel, matlIndex, patch); */;
    //    if (d_MAlab)
    //      new_dw->put(denMicro, d_lab->densityMicroLabel, matlIndex, patch);
    if (d_reactingFlow) {
      // allocateAndPut instead:
      /* new_dw->put(temperature, d_lab->d_tempINPredLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(co2, d_lab->d_co2INPredLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpy, d_lab->d_enthalpyRXNPredLabel, matlIndex, patch); */;
      if (d_mixingModel->getNumRxnVars())
	// allocateAndPut instead:
	/* new_dw->put(reactscalarSRC, d_lab->d_reactscalarSRCINPredLabel,
		    matlIndex, patch); */;
    }
    if (d_radiationCalc) {
      // allocateAndPut instead:
      /* new_dw->put(absorption, d_lab->d_absorpINPredLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(sootFV, d_lab->d_sootFVINPredLabel, matlIndex, patch); */;
    }
  }
}

//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_computePropsInterm(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::computePropsInterm",
			  this,
			  &Properties::computePropsInterm);

  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  tsk->requires(Task::NewDW, d_lab->d_scalarIntermLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }

  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarIntermLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpyIntermLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);


  tsk->computes(d_lab->d_refDensityInterm_label);
  tsk->computes(d_lab->d_densityIntermLabel);
  tsk->computes(d_lab->d_drhodfIntermLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINIntermLabel);
    tsk->computes(d_lab->d_co2INIntermLabel);
    tsk->computes(d_lab->d_h2oINIntermLabel);
    tsk->computes(d_lab->d_enthalpyRXNIntermLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINIntermLabel);
  }
  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINIntermLabel);
    tsk->computes(d_lab->d_sootFVINIntermLabel);
  }
  if (d_MAlab) 
    tsk->computes(d_lab->d_densityMicroLabel);
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::computePropsInterm(const ProcessorGroup*,
			     const PatchSubset* patches,
			     const MaterialSubset*,
			     DataWarehouse*,
			     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the CCVariable (density) from the old datawarehouse
    // just write one function for computing properties

    constCCVariable<double> density;
    constCCVariable<double> voidFraction;
    CCVariable<double> temperature;
    CCVariable<double> new_density;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpy;
    CCVariable<double> reactscalarSRC;
    CCVariable<double> drhodf;

    if (d_reactingFlow) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINIntermLabel, matlIndex, patch);
      new_dw->allocateAndPut(co2, d_lab->d_co2INIntermLabel, matlIndex, patch);
      new_dw->allocateAndPut(h2o, d_lab->d_h2oINIntermLabel, matlIndex, patch);
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpyRXNIntermLabel, matlIndex, patch);
      if (d_mixingModel->getNumRxnVars()) {
	new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINIntermLabel,
			 matlIndex, patch);
	reactscalarSRC.initialize(0.0);
      }
    }
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    if (d_radiationCalc) {
      new_dw->allocateAndPut(absorption, d_lab->d_absorpINIntermLabel, matlIndex, patch);
      new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINIntermLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }
 
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);
    //constCCVariable<double> denMicro;
    constCCVariable<double> enthalpy_comp;

    new_dw->allocateAndPut(drhodf, d_lab->d_drhodfIntermLabel, matlIndex, patch);
    drhodf.initialize(0.0);

    new_dw->allocateAndPut(new_density, d_lab->d_densityIntermLabel, matlIndex, patch);
    new_dw->get(density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_density.copyData(density);
    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      //new_dw->get(denMicro, d_lab->d_densityMicroINLabel, 
      //	  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarIntermLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    StaticArray<constCCVariable<double> > scalarVar(d_numMixStatVars);

    if (d_numMixStatVars > 0) {
      for (int ii = 0; ii < d_numMixStatVars; ii++)
	new_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }


    StaticArray<constCCVariable<double> > reactScalar(d_mixingModel->getNumRxnVars());
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->get(reactScalar[ii], d_lab->d_reactscalarIntermLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    if (!(d_mixingModel->isAdiabatic()))
      new_dw->get(enthalpy_comp, d_lab->d_enthalpyIntermLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    //    voidFraction.print(cerr);
    // set density for the whole domain
    InletStream inStream(d_numMixingVars, 
		         d_mixingModel->getNumMixStatVars(),
		         d_mixingModel->getNumRxnVars());
    Stream outStream;

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);

	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {

	    inStream.d_mixVars[ii] = (scalar[ii])[currCell];

	  }

	  if (d_numMixStatVars > 0) {

	    for (int ii = 0; ii < d_numMixStatVars; ii++ ) {

	      inStream.d_mixVarVariance[ii] = (scalarVar[ii])[currCell];

	    }

	  }

	  if (d_mixingModel->getNumRxnVars() > 0) {
	    for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++ ) 
	      inStream.d_rxnVars[ii] = (reactScalar[ii])[currCell];
	  }

	  // after computing variance get that too, for the time being setting the 
	  // value to zero
	  //	  inStream.d_mixVarVariance[0] = 0.0;
	  // currently not using any reaction progress variables

	  if (!d_mixingModel->isAdiabatic())
	    inStream.d_enthalpy = enthalpy_comp[currCell];
	  d_mixingModel->computeProps(inStream, outStream);
	  double local_den = outStream.getDensity();
	  drhodf[currCell] = outStream.getdrhodf();

	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    enthalpy[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  if (d_radiationCalc) {
	    // bc is the mass-atoms 0f carbon per mas of reactnat mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    // optical path length
	    double opl = 3.0;
	    if (d_mixingModel->getNumRxnVars()) 
	      sootFV[currCell] = outStream.getSootFV();
	    else {
	      if (temperature[currCell] > 1000) {
	        double bc = inStream.d_mixVars[0]*(84.0/100.0)*local_den;
	        double c3 = 0.1;
	        double rhosoot = 1950.0;
	        double cmw = 12.0;
	      
	        double factor = 0.01;
	        if (inStream.d_mixVars[0] > 0.1)
		  sootFV[currCell] = c3*bc*cmw/rhosoot*factor;
	        else
		  sootFV[currCell] = 0.0;
	      }
	      else 
	        sootFV[currCell] = 0.0;
	    }
	    absorption[currCell] = 0.01+ Min(0.5,(4.0/opl)*log(1.0+350.0*
				   sootFV[currCell]*temperature[currCell]*opl));
	  }

	  if (d_MAlab) {
	    //denMicro[IntVector(colX, colY, colZ)] = local_den;
	    if (voidFraction[currCell] > 0.01)
	      local_den *= voidFraction[currCell];

	  }
	  
	  new_density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
	    (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
	}
      }
    }
    // Write the computed density to the new data warehouse
#if 0
    //#ifdef ARCHES_PRES_DEBUG
    // Testing if correct values have been put
    cerr << " AFTER COMPUTE PROPERTIES " << endl;
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    density.print(cerr);
#endif
    if (patch->containsCell(d_denRef)) {
      double den_ref = new_density[d_denRef];
      cerr << "density_ref " << den_ref << endl;
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensityInterm_label);
    }
    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensityInterm_label);
    
    // allocateAndPut instead:
    /* new_dw->put(new_density, d_lab->d_densityIntermLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(drhodf, d_lab->d_drhodfIntermLabel, matlIndex, patch); */;
    if (d_reactingFlow) {
      // allocateAndPut instead:
      /* new_dw->put(temperature, d_lab->d_tempINIntermLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(co2, d_lab->d_co2INIntermLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpy, d_lab->d_enthalpyRXNIntermLabel, matlIndex, patch); */;
      if (d_mixingModel->getNumRxnVars())
	// allocateAndPut instead:
	/* new_dw->put(reactscalarSRC, d_lab->d_reactscalarSRCINIntermLabel,
		    matlIndex, patch); */;
    }
    if (d_radiationCalc) {
      // allocateAndPut instead:
      /* new_dw->put(absorption, d_lab->d_absorpINIntermLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(sootFV, d_lab->d_sootFVINIntermLabel, matlIndex, patch); */;
    }
  }
}

//****************************************************************************
// Schedule the computation of density reference array here
//****************************************************************************
void 
Properties::sched_computeDenRefArrayPred(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls)

{

  // primitive variable initialization

  Task* tsk = scinew Task("Properties::computeDenRefArrayPred",
			  this, &Properties::computeDenRefArrayPred);


  tsk->requires(Task::NewDW, d_lab->d_refDensityPred_label);
  tsk->computes(d_lab->d_denRefArrayPredLabel);

  if (d_MAlab) {

    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  sched->addTask(tsk, patches, matls);

}
  
//****************************************************************************
// Actually calculate the density reference array here
//****************************************************************************

void 
Properties::computeDenRefArrayPred(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset*,
			       DataWarehouse*,
			       DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> denRefArray;
    constCCVariable<double> voidFraction;


    sum_vartype den_ref_var;
    new_dw->get(den_ref_var, d_lab->d_refDensityPred_label);

    double den_Ref = den_ref_var;

    //cerr << "getdensity_ref " << den_Ref << endl;

    if (d_MAlab) {

      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    new_dw->allocateAndPut(denRefArray, d_lab->d_denRefArrayPredLabel, 
		     matlIndex, patch);
		
    denRefArray.initialize(den_Ref);

    if (d_MAlab) {

      for (CellIterator iter = patch->getCellIterator();
	   !iter.done();iter++){

	denRefArray[*iter]  *= voidFraction[*iter];

      }
    }

    // allocateAndPut instead:
    /* new_dw->put(denRefArray, d_lab->d_denRefArrayPredLabel,
		matlIndex, patch); */;

  }
}

//****************************************************************************
// Schedule the computation of density reference array here
//****************************************************************************
void 
Properties::sched_computeDenRefArrayInterm(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls)

{

  // primitive variable initialization

  Task* tsk = scinew Task("Properties::computeDenRefArrayInterm",
			  this, &Properties::computeDenRefArrayInterm);


  tsk->requires(Task::NewDW, d_lab->d_refDensityInterm_label);
  tsk->computes(d_lab->d_denRefArrayIntermLabel);

  if (d_MAlab) {

    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
  sched->addTask(tsk, patches, matls);

}
  
//****************************************************************************
// Actually calculate the density reference array here
//****************************************************************************

void 
Properties::computeDenRefArrayInterm(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset*,
			       DataWarehouse*,
			       DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> denRefArray;
    constCCVariable<double> voidFraction;


    sum_vartype den_ref_var;
    new_dw->get(den_ref_var, d_lab->d_refDensityInterm_label);

    double den_Ref = den_ref_var;

    //cerr << "getdensity_ref " << den_Ref << endl;

    if (d_MAlab) {

      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    new_dw->allocateAndPut(denRefArray, d_lab->d_denRefArrayIntermLabel, 
		     matlIndex, patch);
		
    denRefArray.initialize(den_Ref);

    if (d_MAlab) {

      for (CellIterator iter = patch->getCellIterator();
	   !iter.done();iter++){

	denRefArray[*iter]  *= voidFraction[*iter];

      }
    }

    // allocateAndPut instead:
    /* new_dw->put(denRefArray, d_lab->d_denRefArrayIntermLabel,
		matlIndex, patch); */;

  }
}

//****************************************************************************
// Schedule the averaging of properties for Runge-Kutta step
//****************************************************************************
void 
Properties::sched_averageRKProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls,
				 const int Runge_Kutta_current_step,
				 const bool Runge_Kutta_last_step)
{
  Task* tsk = scinew Task("Properties::averageRKProps",
			  this,
			  &Properties::averageRKProps,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalarOUTBCLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpyOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  switch (Runge_Kutta_current_step) {
  case Arches::SECOND:
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  break;

  case Arches::THIRD:
    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  break;

  default:
    throw InvalidValue("Invalid Runge-Kutta step in averageRKProps");
  }

  if (Runge_Kutta_last_step) {
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->modifies(d_lab->d_scalarSPLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->modifies(d_lab->d_reactscalarSPLabel);
    if (!(d_mixingModel->isAdiabatic()))
      tsk->modifies(d_lab->d_enthalpySPLabel);
  }
  else {
    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->modifies(d_lab->d_scalarIntermLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->modifies(d_lab->d_reactscalarIntermLabel);
    if (!(d_mixingModel->isAdiabatic()))
      tsk->modifies(d_lab->d_enthalpyIntermLabel);
  }

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually average the Runge-Kutta properties here
//****************************************************************************
void 
Properties::averageRKProps(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse*,
			   DataWarehouse* new_dw,
			   const int Runge_Kutta_current_step,
			   const bool Runge_Kutta_last_step)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> old_density;
    constCCVariable<double> rho1_density;
    constCCVariable<double> new_density;
    StaticArray<constCCVariable<double> > old_scalar(d_numMixingVars);
    StaticArray<CCVariable<double> > new_scalar(d_numMixingVars);
    StaticArray<constCCVariable<double> > old_reactScalar(d_mixingModel->getNumRxnVars());
    StaticArray<CCVariable<double> > new_reactScalar(d_mixingModel->getNumRxnVars());
    constCCVariable<double> old_enthalpy;
    CCVariable<double> new_enthalpy;

    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    for (int ii = 0; ii < d_numMixingVars; ii++) {
      new_dw->get(old_scalar[ii], d_lab->d_scalarOUTBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++) {
	new_dw->get(old_reactScalar[ii], d_lab->d_reactscalarOUTBCLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      }
    }
    if (!(d_mixingModel->isAdiabatic())) {
      new_dw->get(old_enthalpy, d_lab->d_enthalpyOUTBCLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    switch (Runge_Kutta_current_step) {
    case Arches::SECOND:
      new_dw->get(rho1_density, d_lab->d_densityPredLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    break;

    case Arches::THIRD:
      new_dw->get(rho1_density, d_lab->d_densityIntermLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    break;

    default:
      throw InvalidValue("Invalid Runge-Kutta step in averageRKProps");
    }

    if (Runge_Kutta_last_step) {
      new_dw->get(new_density, d_lab->d_densityCPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      for (int ii = 0; ii < d_numMixingVars; ii++) {
        new_dw->getModifiable(new_scalar[ii], d_lab->d_scalarSPLabel, 
		              matlIndex, patch);
      }
      if (d_mixingModel->getNumRxnVars() > 0) {
        for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++) {
	  new_dw->getModifiable(new_reactScalar[ii],d_lab->d_reactscalarSPLabel,
		    		matlIndex, patch);
        }
      }
      if (!(d_mixingModel->isAdiabatic())) {
        new_dw->getModifiable(new_enthalpy, d_lab->d_enthalpySPLabel, 
			      matlIndex, patch);
      }
    }
    else {
      new_dw->get(new_density, d_lab->d_densityIntermLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      for (int ii = 0; ii < d_numMixingVars; ii++) {
        new_dw->getModifiable(new_scalar[ii], d_lab->d_scalarIntermLabel, 
		              matlIndex, patch);
      }
      if (d_mixingModel->getNumRxnVars() > 0) {
        for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++) {
	  new_dw->getModifiable(new_reactScalar[ii],
				d_lab->d_reactscalarIntermLabel,
		    		matlIndex, patch);
        }
      }
      if (!(d_mixingModel->isAdiabatic())) {
        new_dw->getModifiable(new_enthalpy, d_lab->d_enthalpyIntermLabel, 
			      matlIndex, patch);
      }
    }

    double factor_old, factor_new, factor_divide;
    switch (Runge_Kutta_current_step) {
    case Arches::SECOND:
      if (Runge_Kutta_last_step) {
	factor_old = 1.0;
	factor_new = 1.0;
	factor_divide = 2.0;
      }
      else {
	factor_old = 3.0;
	factor_new = 1.0;
	factor_divide = 4.0;
      }
    break;

    case Arches::THIRD:
	factor_old = 1.0;
	factor_new = 2.0;
	factor_divide = 3.0;
    break;

    default:
      throw InvalidValue("Invalid Runge-Kutta step in averageRKProps");
    }

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
          
          double predicted_density;
//          predicted_density = (old_density[currCell] +
//			       new_density[currCell])/2.0;
//          predicted_density = rho1_density[currCell];
	  if (old_density[currCell] > 0.0)
            predicted_density = 1.0/((factor_old/old_density[currCell] +
			       factor_new/new_density[currCell])/factor_divide);
	  else
	    predicted_density = new_density[currCell];

	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {
	    (new_scalar[ii])[currCell] = (factor_old*old_density[currCell]*
		(old_scalar[ii])[currCell] + factor_new*rho1_density[currCell]*
		(new_scalar[ii])[currCell])/(factor_divide*predicted_density);
            if ((new_scalar[ii])[currCell] > 1.0) 
		(new_scalar[ii])[currCell] = 1.0;
            else if ((new_scalar[ii])[currCell] < 0.0)
            	(new_scalar[ii])[currCell] = 0.0;
          }

	  if (d_mixingModel->getNumRxnVars() > 0) {
	    for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++ ) {
	      (new_reactScalar[ii])[currCell] = (factor_old *
		old_density[currCell]*(old_reactScalar[ii])[currCell] +
		factor_new*rho1_density[currCell]*
		(new_reactScalar[ii])[currCell])/
		(factor_divide*predicted_density);
            if ((new_reactScalar[ii])[currCell] > 1.0) 
		(new_reactScalar[ii])[currCell] = 1.0;
            else if ((new_reactScalar[ii])[currCell] < 0.0)
            	(new_reactScalar[ii])[currCell] = 0.0;
            }
	  }

	  if (!d_mixingModel->isAdiabatic())
	    new_enthalpy[currCell] = (factor_old*old_density[currCell]*
		old_enthalpy[currCell] + factor_new*rho1_density[currCell]*
		new_enthalpy[currCell])/(factor_divide*predicted_density);

	}
      }
    }
  }
}

//****************************************************************************
// Schedule the recomputation of properties for Runge-Kutta
//****************************************************************************
void 
Properties::sched_reComputeRKProps(SchedulerP& sched, const PatchSet* patches,
				   const MaterialSet* matls,
				   const int Runge_Kutta_current_step,
				   const bool Runge_Kutta_last_step)
{
  Task* tsk = scinew Task("Properties::ReComputeRKProps",
			  this,
			  &Properties::reComputeRKProps,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    //    tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }
// use densityPredLabel to store rho_2 for now
  tsk->modifies(d_lab->d_densityPredLabel);
  if (Runge_Kutta_last_step) {
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }

  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

//  tsk->modifies(d_lab->d_refDensity_label);
  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->modifies(d_lab->d_drhodfCPLabel);
  if (d_reactingFlow) {
    tsk->modifies(d_lab->d_tempINLabel);
    tsk->modifies(d_lab->d_co2INLabel);
    tsk->modifies(d_lab->d_h2oINLabel);
    tsk->modifies(d_lab->d_enthalpyRXNLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->modifies(d_lab->d_reactscalarSRCINLabel);
  }
  if (d_radiationCalc) {
    tsk->modifies(d_lab->d_absorpINLabel);
    tsk->modifies(d_lab->d_sootFVINLabel);
  }
  }
  else {
  tsk->requires(Task::NewDW, d_lab->d_scalarIntermLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }

  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarIntermLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpyIntermLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

//  tsk->modifies(d_lab->d_refDensity_label);
  tsk->modifies(d_lab->d_densityIntermLabel);
  tsk->modifies(d_lab->d_drhodfIntermLabel);
  if (d_reactingFlow) {
    tsk->modifies(d_lab->d_tempINIntermLabel);
    tsk->modifies(d_lab->d_co2INIntermLabel);
    tsk->modifies(d_lab->d_h2oINIntermLabel);
    tsk->modifies(d_lab->d_enthalpyRXNIntermLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->modifies(d_lab->d_reactscalarSRCINIntermLabel);
  }
  if (d_radiationCalc) {
    tsk->modifies(d_lab->d_absorpINIntermLabel);
    tsk->modifies(d_lab->d_sootFVINIntermLabel);
  }
  }

  if (d_MAlab) 
    tsk->modifies(d_lab->d_densityMicroLabel);

#ifdef scalarSolve_debug
  if (d_MAlab) 
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  numGhostCells);
#endif

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually recompute the Runge-Kutta properties here
//****************************************************************************
void 
Properties::reComputeRKProps(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse*,
			   DataWarehouse* new_dw,
			   const int Runge_Kutta_current_step,
			   const bool Runge_Kutta_last_step)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the CCVariable (density) from the old datawarehouse
    // just write one function for computing properties

    constCCVariable<double> density;
    constCCVariable<double> voidFraction;
    CCVariable<double> temperature;
    CCVariable<double> new_density;
    CCVariable<double> temp_density;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpy;
    CCVariable<double> reactscalarSRC;
    CCVariable<double> drhodf;
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);
    constCCVariable<double> enthalpy_comp;
    CCVariable<double> denMicro;
    //    CCVariable<double> denMicro_old;
    StaticArray<constCCVariable<double> > scalarVar(d_numMixStatVars);
    StaticArray<constCCVariable<double> > reactScalar(d_mixingModel->getNumRxnVars());
    if (Runge_Kutta_last_step) {
    if (d_reactingFlow) {
      new_dw->getModifiable(temperature, d_lab->d_tempINLabel,
			    matlIndex, patch);
      new_dw->getModifiable(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->getModifiable(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpyRXNLabel,
			    matlIndex, patch);
      temperature.initialize(0.0);
      co2.initialize(0.0);
      h2o.initialize(0.0);
      enthalpy.initialize(0.0);
      if (d_mixingModel->getNumRxnVars()) {
	new_dw->getModifiable(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			      matlIndex, patch);
	reactscalarSRC.initialize(0.0);
      }
    }
    if (d_radiationCalc) {
      new_dw->getModifiable(absorption, d_lab->d_absorpINLabel,
			    matlIndex, patch);
      new_dw->getModifiable(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }

    new_dw->getModifiable(drhodf, d_lab->d_drhodfCPLabel, matlIndex, patch);
    drhodf.initialize(0.0);

    new_dw->getModifiable(new_density, d_lab->d_densityCPLabel,
			  matlIndex, patch);

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_numMixStatVars > 0) {
      for (int ii = 0; ii < d_numMixStatVars; ii++)
	new_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->get(reactScalar[ii], d_lab->d_reactscalarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (!(d_mixingModel->isAdiabatic()))
      new_dw->get(enthalpy_comp, d_lab->d_enthalpySPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    else {
    if (d_reactingFlow) {
      new_dw->getModifiable(temperature, d_lab->d_tempINIntermLabel,
			    matlIndex, patch);
      new_dw->getModifiable(co2, d_lab->d_co2INIntermLabel, matlIndex, patch);
      new_dw->getModifiable(h2o, d_lab->d_h2oINIntermLabel, matlIndex, patch);
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpyRXNIntermLabel,
			    matlIndex, patch);
      temperature.initialize(0.0);
      co2.initialize(0.0);
      h2o.initialize(0.0);
      enthalpy.initialize(0.0);
      if (d_mixingModel->getNumRxnVars()) {
	new_dw->getModifiable(reactscalarSRC,
			      d_lab->d_reactscalarSRCINIntermLabel,
			      matlIndex, patch);
	reactscalarSRC.initialize(0.0);
      }
    }
    if (d_radiationCalc) {
      new_dw->getModifiable(absorption, d_lab->d_absorpINIntermLabel,
			    matlIndex, patch);
      new_dw->getModifiable(sootFV, d_lab->d_sootFVINIntermLabel,
			    matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }

    new_dw->getModifiable(drhodf, d_lab->d_drhodfIntermLabel, matlIndex, patch);
    drhodf.initialize(0.0);

    new_dw->getModifiable(new_density, d_lab->d_densityIntermLabel,
			  matlIndex, patch);


    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarIntermLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_numMixStatVars > 0) {
      for (int ii = 0; ii < d_numMixStatVars; ii++)
	new_dw->get(scalarVar[ii], d_lab->d_scalarVarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->get(reactScalar[ii], d_lab->d_reactscalarIntermLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (!(d_mixingModel->isAdiabatic()))
      new_dw->get(enthalpy_comp, d_lab->d_enthalpyIntermLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    new_dw->getModifiable(temp_density, d_lab->d_densityPredLabel,
			  matlIndex, patch);
    temp_density.copyData(new_density);
    new_dw->get(density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_density.copyData(density);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      //new_dw->get(denMicro_old, d_lab->d_densityMicroINLabel, 
      //	  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->getModifiable(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch);
      denMicro.initialize(0.0);
      //denMicro.copyData(denMicro_old);
    }

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

#ifdef ARCHES_DEBUG
    IntVector test(6,9,9);
    cout << "printing test "<<test<<endl;
#endif
    InletStream inStream(d_numMixingVars,
		         d_mixingModel->getNumMixStatVars(),
		         d_mixingModel->getNumRxnVars());
    Stream outStream;

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);

	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {

	    inStream.d_mixVars[ii] = (scalar[ii])[currCell];

#ifdef ARCHES_DEBUG
	    if ((colX==6) && (colY==9) && (colZ==9))
	      cerr << "Mixture Vars at test = " << (scalar[ii])[currCell];
#endif

	  }

	  if (d_numMixStatVars > 0) {

	    for (int ii = 0; ii < d_numMixStatVars; ii++ ) {

	      inStream.d_mixVarVariance[ii] = (scalarVar[ii])[currCell];

	    }

	  }

	  if (d_mixingModel->getNumRxnVars() > 0) {
	    for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++ ) 
	      inStream.d_rxnVars[ii] = (reactScalar[ii])[currCell];
	  }

	  // after computing variance get that too, for the time being setting the 
	  // value to zero
	  //	  inStream.d_mixVarVariance[0] = 0.0;
	  // currently not using any reaction progress variables

	  if (!d_mixingModel->isAdiabatic())
	    inStream.d_enthalpy = enthalpy_comp[currCell];
	  d_mixingModel->computeProps(inStream, outStream);
	  double local_den = outStream.getDensity();
	  drhodf[currCell] = outStream.getdrhodf();
	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    enthalpy[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  if (d_radiationCalc) {
	    // bc is the mass-atoms 0f carbon per mas of reactnat mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    // optical path length
	    double opl = 3.0;
	    if (d_mixingModel->getNumRxnVars()) 
	      sootFV[currCell] = outStream.getSootFV();
	    else {
	      if (temperature[currCell] > 1000) {
		double bc = inStream.d_mixVars[0]*(84.0/100.0)*local_den;
		double c3 = 0.1;
		double rhosoot = 1950.0;
		double cmw = 12.0;

		double factor = 0.01;
		if (inStream.d_mixVars[0] > 0.1)
		  sootFV[currCell] = c3*bc*cmw/rhosoot*factor;
		else
		  sootFV[currCell] = 0.0;
	      }
	      else 
		sootFV[currCell] = 0.0;
	    }
	    absorption[currCell] = 0.01+ Min(0.5,(4.0/opl)*log(1.0+350.0*
				   sootFV[currCell]*temperature[currCell]*opl));
	    }
	  if (d_MAlab) {
	    denMicro[IntVector(colX, colY, colZ)] = local_den;
	    local_den *= voidFraction[currCell];

	  }

	  // no under-relaxation for MPMArches
	  if (d_MAlab) {
	    new_density[IntVector(colX, colY, colZ)] = local_den;
	  }
	  else {
	    new_density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
	      (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
	  }
	}
      }
    }
    // Write the computed density to the new data warehouse
#ifdef ARCHES_PRES_DEBUG
    // Testing if correct values have been put
    cerr << " AFTER COMPUTE PROPERTIES " << endl;
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    density.print(cerr);
#endif

#ifdef scalarSolve_debug

    if (d_MAlab) {
      constCCVariable<int> cellType;
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
    		  Ghost::None, 0);

      cerr << " NEW DENSITY VALUES " << endl;
      for (int ii = 5; ii <= 9; ii++) {
	for (int jj = 7; jj <= 12; jj++) {
	  for (int kk = 7; kk <= 12; kk++) {
	    cerr.width(14);
	    cerr << " point coordinates "<< ii << " " << jj << " " << kk ;
	    cerr << " new density = " << new_density[IntVector(ii,jj,kk)] ; 
	    cerr << " cellType = " << cellType[IntVector(ii,jj,kk)] ; 
	    cerr << " void fraction = " << voidFraction[IntVector(ii,jj,kk)] << endl; 
	  }
	}
      }

    }
#endif

  /*  if (patch->containsCell(d_denRef)) {
      double den_ref = new_density[d_denRef];
      cerr << "density_ref " << den_ref << endl;
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
    }
    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);
    */
  }
}
//****************************************************************************
// Schedule the computation of drhodt
//****************************************************************************
void 
Properties::sched_computeDrhodt(SchedulerP& sched, const PatchSet* patches,
				const MaterialSet* matls, 
				const int Runge_Kutta_current_step,
				const bool Runge_Kutta_last_step)
{
  Task* tsk = scinew Task("Properties::computeDrhodt",
			  this,
			  &Properties::computeDrhodt,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::OldDW, d_lab->d_oldDeltaTLabel);

  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_densityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (Runge_Kutta_last_step) {
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
  }
  else { 
 	switch (Runge_Kutta_current_step) {
 	case Arches::FIRST:
    		tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
			      Ghost::None, Arches::ZEROGHOSTCELLS);
	 break;

	 case Arches::SECOND:
    		tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
			      Ghost::None, Arches::ZEROGHOSTCELLS);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in computeDrhodt");
	 }
  }
  if (Runge_Kutta_current_step == Arches::FIRST) {
     tsk->computes(d_lab->d_filterdrhodtLabel);
     tsk->computes(d_lab->d_oldDeltaTLabel);
  }
  else
     tsk->modifies(d_lab->d_filterdrhodtLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Compute drhodt
//****************************************************************************
void 
Properties::computeDrhodt(const ProcessorGroup* pc,
			  const PatchSubset* patches,
			  const MaterialSubset*,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const int Runge_Kutta_current_step,
			  const bool Runge_Kutta_last_step)
{
  int drhodt_1st_order = 1;
  int current_step = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  if (d_MAlab) drhodt_1st_order = 2;
  delt_vartype delT, old_delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  if (Runge_Kutta_current_step == Arches::FIRST)
    new_dw->put(delT, d_lab->d_oldDeltaTLabel);
  double delta_t = delT;
  old_dw->get(old_delT, d_lab->d_oldDeltaTLabel);
  double  old_delta_t = old_delT;


  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> new_density;
    constCCVariable<double> old_density;
    constCCVariable<double> old_old_density;
    CCVariable<double> drhodt;
    CCVariable<double> filterdrhodt;

    new_dw->get(old_density, d_lab->d_densityINLabel, matlIndex, patch,
	        Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->get(old_old_density, d_lab->d_densityINLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (Runge_Kutta_last_step) {
      new_dw->get(new_density, d_lab->d_densityCPLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
		new_dw->get(new_density, d_lab->d_densityPredLabel, matlIndex,
			patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	 break;

	 case Arches::SECOND:
		new_dw->get(new_density, d_lab->d_densityIntermLabel, matlIndex,
			patch, Ghost::None, Arches::ZEROGHOSTCELLS);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in computeDrhodt");
	 }
    }
    if (Runge_Kutta_current_step == Arches::FIRST)
      new_dw->allocateAndPut(filterdrhodt, d_lab->d_filterdrhodtLabel,
		             matlIndex, patch);
    else
      new_dw->getModifiable(filterdrhodt, d_lab->d_filterdrhodtLabel,
		            matlIndex, patch);
      filterdrhodt.initialize(0.0);

    // Get the patch and variable indices
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    // compute drhodt and add its filtered value
    drhodt.resize(patch->getLowIndex(), patch->getHighIndex());

    if (current_step <= drhodt_1st_order) {
// 1st order drhodt
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	    IntVector currcell(ii,jj,kk);

	    double vol = cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
	    drhodt[currcell] = (new_density[currcell] -
				old_density[currcell])*vol/delta_t;
          }
        }
      }
    }
    else {
// 2nd order drhodt, assuming constant volume
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	    IntVector currcell(ii,jj,kk);

	    double vol = cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
	    double factor = 1.0 + old_delta_t/delta_t;
            double new_factor = factor * factor - 1.0;
	    double old_factor = factor * factor;
	    drhodt[currcell] = (new_factor*new_density[currcell] -
				old_factor*old_density[currcell] +
				old_old_density[currcell])*vol /
			       (old_delta_t*factor);
          }
        }
      }
    }

#ifdef FILTER_DRHODT
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch, drhodt, filterdrhodt);
#else
    filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
		      drhodt.getHighIndex());
#endif
#else
    filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
		      drhodt.getHighIndex());
#endif
  }
}
