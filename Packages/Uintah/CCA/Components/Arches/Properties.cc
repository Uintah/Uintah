//----- Properties.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
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
}

//****************************************************************************
// Destructor
//****************************************************************************
Properties::~Properties()
{
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
  else
    throw InvalidValue("Mixing Model not supported" + mixModel);
  d_mixingModel->problemSetup(db);
  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = d_mixingModel->getNumMixVars();
}

//****************************************************************************
// compute density for inlet streams: only for cold streams
//****************************************************************************

void
Properties::computeInletProperties(const InletStream& inStream, 
				   Stream& outStream)
{
  d_mixingModel->computeProps(inStream, outStream);
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

  int numGhostCells = 0;
  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densitySPLabel, Ghost::None,
		numGhostCells);
  // will only work for one mixing variables
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		numGhostCells);
  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityCPLabel);
  if (d_enthalpySolve)
    tsk->computes(d_lab->d_enthalpySPLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
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
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::ReComputeProps",
			  this,
			  &Properties::reComputeProps);

  int numGhostCells = 0;
  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells,
		numGhostCells+2);
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, 
		  Ghost::None, numGhostCells);
  }
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		numGhostCells);
  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, Ghost::None,
		  numGhostCells);

  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityCPLabel);
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
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

  int zeroGhostCells = 0;

  tsk->requires(Task::OldDW, d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_denRefArrayLabel);

  if (d_MAlab) {

    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, zeroGhostCells);
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
    int nofGhostCells = 0;
    CCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
    CCVariable<double> density;
    StaticArray<CCVariable<double> > scalar(d_numMixingVars);
    CCVariable<double> enthalpy;

    new_dw->get(density, d_lab->d_densitySPLabel, 
		matlIndex, patch, Ghost::None, nofGhostCells);
    if (d_enthalpySolve)
      new_dw->get(enthalpy, d_lab->d_enthalpySPBCLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);

    CCVariable<double> temperature;
    CCVariable<double> co2;
    CCVariable<double> enthalpyRXN;
    if (d_reactingFlow) {
      new_dw->allocate(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->allocate(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->allocate(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch);
    }
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    if (d_radiationCalc) {
      new_dw->allocate(absorption, d_lab->d_absorpINLabel, matlIndex, patch);
      new_dw->allocate(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }
    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);

    // get multimaterial vars
    CCVariable<double> denMicro;
    //CCVariable<double> new_density;
    //new_dw->allocate(new_density, d_densityCPLabel, matlIndex, patch);
    if (d_MAlab) 
    new_dw->allocate(denMicro, d_lab->d_densityMicroLabel,
		     matlIndex, patch);
    
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

	  // after computing variance get that too, for the time being setting the 
	  // value to zero
	  //	  inStream.d_mixVarVariance[0] = 0.0;
	  // currently not using any reaction progress variables

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
	    enthalpyRXN[currCell] = outStream.getEnthalpy();
	  }
	  if (d_bc == 0)
	    throw InvalidValue("BoundaryCondition pointer not assigned");
	  if (d_MAlab)
	    denMicro[currCell] = local_den;

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

    new_dw->put(density,d_lab->d_densityCPLabel, matlIndex, patch);
    if (d_enthalpySolve)
      new_dw->put(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);
    if (d_reactingFlow) {
      new_dw->put(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->put(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->put(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch);
    }
    if (d_radiationCalc){
      new_dw->put(absorption, d_lab->d_absorpINLabel, matlIndex, patch);
      new_dw->put(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
    }
    if (d_MAlab)
      new_dw->put(denMicro,d_lab->d_densityMicroLabel, matlIndex, patch);

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
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // Get the CCVariable (density) from the old datawarehouse
    // just write one function for computing properties

    CCVariable<double> density;
    CCVariable<double> voidFraction;
    CCVariable<double> temperature;
    CCVariable<double> new_density;
    CCVariable<double> co2;
    CCVariable<double> enthalpy;
    if (d_reactingFlow) {
      new_dw->allocate(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->allocate(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->allocate(enthalpy, d_lab->d_enthalpyRXNLabel, matlIndex, patch);
    }
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    if (d_radiationCalc) {
      new_dw->allocate(absorption, d_lab->d_absorpINLabel, matlIndex, patch);
      new_dw->allocate(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }
    new_dw->allocate(new_density, d_lab->d_densityCPLabel, matlIndex, patch);
 
    StaticArray<CCVariable<double> > scalar(d_numMixingVars);
    CCVariable<double> enthalpy_comp;
    CCVariable<double> denMicro;

    int nofGhostCells = 0;

    new_dw->get(density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, nofGhostCells);
    new_density.copy(density);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);
      new_dw->get(denMicro, d_lab->d_densityMicroINLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);
    }

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);
    if (!(d_mixingModel->isAdiabatic()))
      new_dw->get(enthalpy_comp, d_lab->d_enthalpySPLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();


    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	  // for combustion calculations mixingmodel will be called
	  // this is similar to prcf.f
	  // construct an InletStream for input to the computeProps of mixingModel

	  IntVector currCell(colX, colY, colZ);
	  InletStream inStream(d_numMixingVars, d_mixingModel->getNumMixStatVars(),
			       d_mixingModel->getNumRxnVars());

	  for (int ii = 0; ii < d_numMixingVars; ii++ ) {

	    inStream.d_mixVars[ii] = (scalar[ii])[currCell];

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
	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    co2[currCell] = outStream.getCO2();
	    enthalpy[currCell] = outStream.getEnthalpy();
	  }
	  if (d_radiationCalc) {
	    // bc is the mass-atoms 0f carbon per mas of reactnat mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    if (temperature[currCell] > 1000) {
	      double bc = inStream.d_mixVars[0]*(84.0/100.0)*local_den;
	      double c3 = 0.1;
	      double rhosoot = 1950.0;
	      double cmw = 12.0;
	      double opl = 3.0;
	      double factor = 0.01;
	      if (inStream.d_mixVars[0] > 0.06)
		sootFV[currCell] = c3*bc*cmw/rhosoot*factor;
	      else
		sootFV[currCell] = 0.0;
	      absorption[currCell] = Min(0.5,(4.0/opl)*log(1.0+350.0*
		     sootFV[currCell]*temperature[currCell]*opl));
	      //	      absorption[currCell] = 0.01;
	    }
	    else {
	      absorption[currCell] = 0.0;
	    }
	  }
	  if (d_MAlab) {
	    denMicro[IntVector(colX, colY, colZ)] = local_den;
	    if (voidFraction[currCell] > 0.01)
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
    
    new_dw->put(new_density, d_lab->d_densityCPLabel, matlIndex, patch);
    if (d_reactingFlow) {
      new_dw->put(temperature, d_lab->d_tempINLabel, matlIndex, patch);
      new_dw->put(co2, d_lab->d_co2INLabel, matlIndex, patch);
      new_dw->put(enthalpy, d_lab->d_enthalpyRXNLabel, matlIndex, patch);
    }
    if (d_radiationCalc) {
      new_dw->put(absorption, d_lab->d_absorpINLabel, matlIndex, patch);
      new_dw->put(sootFV, d_lab->d_sootFVINLabel, matlIndex, patch);
    }
    if (d_MAlab)
      new_dw->put(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch);
  }

  
}

//****************************************************************************
// Actually calculate the density reference array here
//****************************************************************************

void 
Properties::computeDenRefArray(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset*,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> denRefArray;
    CCVariable<double> voidFraction;

    int zeroGhostCells = 0;

    sum_vartype den_ref_var;
    old_dw->get(den_ref_var, d_lab->d_refDensity_label);

    double den_Ref = den_ref_var;

    //cerr << "getdensity_ref " << den_Ref << endl;

    if (d_MAlab) {

      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, zeroGhostCells);

    }

    new_dw->allocate(denRefArray, d_lab->d_denRefArrayLabel, 
		     matlIndex, patch);
		
    denRefArray.initialize(den_Ref);

    if (d_MAlab) {

      for (CellIterator iter = patch->getCellIterator();
	   !iter.done();iter++){

	denRefArray[*iter]  *= voidFraction[*iter];

      }
    }

    new_dw->put(denRefArray, d_lab->d_denRefArrayLabel,
		matlIndex, patch);

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

  int numGhostCells = 0;
  // requires scalars
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells,
		numGhostCells+2);
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, 
		  Ghost::None, numGhostCells);
  }
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, Ghost::None,
		numGhostCells);

  tsk->computes(d_lab->d_densityPredLabel);
  //  tsk->computes(d_lab->d_tempINLabel);
  //  tsk->computes(d_lab->d_co2INLabel);
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

    CCVariable<double> density;
    CCVariable<double> new_density;
    CCVariable<double> voidFraction;
 
    StaticArray<CCVariable<double> > scalar(d_numMixingVars);
    CCVariable<double> denMicro;

    int nofGhostCells = 0;
    new_dw->allocate(new_density, d_lab->d_densityPredLabel, 
		     matlIndex, patch);
    new_dw->get(density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, nofGhostCells);
    new_density.copyPatch(density);
    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);
      new_dw->get(denMicro, d_lab->d_densityMicroINLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);
    }

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarPredLabel, 
		  matlIndex, patch, Ghost::None, nofGhostCells);


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

	  // after computing variance get that too, for the time being setting the 
	  // value to zero
	  //	  inStream.d_mixVarVariance[0] = 0.0;
	  // currently not using any reaction progress variables

	  if (!d_mixingModel->isAdiabatic())
	    // get absolute enthalpy from enthalpy eqn
	    cerr << "No eqn for enthalpy yet" << '\n';
	  Stream outStream;
	  d_mixingModel->computeProps(inStream, outStream);
	  double local_den = outStream.getDensity();
	  //	  temperature[currCell] = outStream.getTemperature();
	  //	  co2[currCell] = outStream.getCO2();


	  if (d_MAlab) {
	    denMicro[IntVector(colX, colY, colZ)] = local_den;
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
    new_dw->put(new_density, d_lab->d_densityPredLabel, matlIndex, patch);

  
  }
}
