//----- Properties.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MeanMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
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
  d_bc = 0;
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
  // read type of mixing model
  string mixModel;
  db->require("mixing_model",mixModel);
  if (mixModel == "coldFlowMixingModel")
    d_mixingModel = scinew ColdflowMixingModel();
  else if (mixModel == "pdfMixingModel")
    d_mixingModel = scinew PDFMixingModel();
  else if (mixModel == "meanMixingModel")
    d_mixingModel = scinew MeanMixingModel();
  else
    throw InvalidValue("Mixing Model not supported" + mixModel);
  d_mixingModel->problemSetup(db);
  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = d_mixingModel->getNumMixVars();
  d_numMixStatVars = d_mixingModel->getNumMixStatVars();
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

  // commenting stuff below for debug, sk, 01/22/02
  //  if (d_MAlab) 
  //    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, Ghost::None,
  //		  Arches::ZEROGHOSTCELLS);
  //  else
  //    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
  //		  Arches::ZEROGHOSTCELLS);

  if (d_enthalpySolve)
    tsk->computes(d_lab->d_enthalpySPLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINLabel);

  }
  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
  }
  if (d_MAlab) {
    //    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, Ghost::None,
    //		  Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_densityMicroLabel);
  }

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
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_h2oINLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINLabel);
  }
  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINLabel);
    tsk->computes(d_lab->d_sootFVINLabel);
  }
  if (d_MAlab) 
    tsk->computes(d_lab->d_densityMicroLabel);
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
Properties::computeProps(const ProcessorGroup*,
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
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
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
      //enthalpy.copyData(enthalpy_old);
    }
    
    // temporary comment, for debug, sk, 01/22/02
    //    if (d_MAlab) {
    //      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);
    //    }
    //    else {
    //      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
    //		Ghost::None, Arches::ZEROGHOSTCELLS);
    //    }

    CCVariable<double> temperature;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpyRXN;
    CCVariable<double> reactscalarSRC;
    if (d_reactingFlow) {
      new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, matlIndex, patch);
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
      // temporary comment for debug, sk, 01/22/02
      //      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
      //      		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(denMicro, d_lab->d_densityMicroLabel,
		       matlIndex, patch);
    }
    
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    // set density for the whole domain

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // Store current cell
	  IntVector currCell(colX, colY, colZ);

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  InletStream inStream(d_numMixingVars, d_mixingModel->getNumMixStatVars(),
			       d_mixingModel->getNumRxnVars());

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

	  if (!d_mixingModel->isAdiabatic())
	    inStream.d_enthalpy = 0.0;
	  Stream outStream;
	  d_mixingModel->computeProps(inStream, outStream);
	  double local_den = outStream.getDensity();
	  if (d_enthalpySolve)
	    enthalpy[currCell] = outStream.getEnthalpy();
	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    enthalpyRXN[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  if (d_bc == 0)
	    throw InvalidValue("BoundaryCondition pointer not assigned");

	  if (d_MAlab) {
	    denMicro[currCell] = local_den;
	    //	    if (voidFraction[currCell] > 0.01)
	    // temporary comment for debug, sk, 01/22/02
	    //	    local_den *= voidFraction[currCell];

	  }
	  
	  if (cellType[currCell] != d_bc->wallCellType()) 
	    //	    density[currCell] = d_denUnderrelax*local_den +
	    //  (1.0-d_denUnderrelax)*density[currCell];
	    density[currCell] = local_den;
	}
      }
    }
#ifdef ARCHES_DEBUG
    // Testing if correct values have been put
    cerr << " AFTER COMPUTE PROPERTIES " << endl;
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Density for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << density[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

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
    // allocateAndPut instead:
    /* new_dw->put(density_cp,d_lab->d_densityCPLabel, matlIndex, patch); */;
    if (d_enthalpySolve)
      // allocateAndPut instead:
      /* new_dw->put(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch); */;
    if (d_reactingFlow) {
      // allocateAndPut instead:
      /* new_dw->put(temperature, d_lab->d_tempINLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(co2, d_lab->d_co2INLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch); */;
      if (d_mixingModel->getNumRxnVars())
	// allocateAndPut instead:
	/* new_dw->put(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
		    matlIndex, patch); */;
    }
    if (d_radiationCalc){
      // allocateAndPut instead:
      /* new_dw->put(absorption, d_lab->d_absorpINLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch); */;
    }
    if (d_MAlab)
      // allocateAndPut instead:
      /* new_dw->put(denMicro,d_lab->d_densityMicroLabel, matlIndex, patch); */;

  }
}
  
//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::reComputeProps(const ProcessorGroup*,
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
      new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, matlIndex, patch);
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


    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);
	  InletStream inStream(d_numMixingVars,
			       d_mixingModel->getNumMixStatVars(),
			       d_mixingModel->getNumRxnVars());

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
	  Stream outStream;
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
	    //	    if (voidFraction[currCell] > 0.01)
	    local_den *= voidFraction[currCell];

	  }
	  
	  new_density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
	    (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
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
    if (patch->containsCell(d_denRef)) {
      double den_ref = new_density[d_denRef];
      cerr << "density_ref " << den_ref << endl;
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
    }
    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);
    
    // allocateAndPut instead:
    /* new_dw->put(new_density, d_lab->d_densityCPLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(drhodf, d_lab->d_drhodfCPLabel, matlIndex, patch); */;
    if (d_reactingFlow) {
      // allocateAndPut instead:
      /* new_dw->put(temperature, d_lab->d_tempINLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(co2, d_lab->d_co2INLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(enthalpy, d_lab->d_enthalpyRXNLabel, matlIndex, patch); */;
      if (d_mixingModel->getNumRxnVars())
	// allocateAndPut instead:
	/* new_dw->put(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
		    matlIndex, patch); */;
    }
    if (d_radiationCalc) {
      // allocateAndPut instead:
      /* new_dw->put(absorption, d_lab->d_absorpINLabel, matlIndex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch); */;
    }
    if (d_MAlab)
      // allocateAndPut instead:
      /* new_dw->put(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch); */;
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

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);
	  InletStream inStream(d_numMixingVars, 
			       d_mixingModel->getNumMixStatVars(),
			       d_mixingModel->getNumRxnVars());

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
	  Stream outStream;
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

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);
	  InletStream inStream(d_numMixingVars, 
			       d_mixingModel->getNumMixStatVars(),
			       d_mixingModel->getNumRxnVars());

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
	  Stream outStream;
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

