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
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StaticMixingTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MeanMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
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
  d_DORadiationCalc = false;
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
  db->getWithDefault("filter_drhodt",d_filter_drhodt,false);
  db->require("ref_point", d_denRef);
  db->require("radiation",d_radiationCalc);
  if (d_radiationCalc) {
    db->getWithDefault("discrete_ordinates",d_DORadiationCalc,true);
//    db->getWithDefault("opl",d_opl,3.0); too sensitive to have default
    db->require("opl",d_opl);
    db->getWithDefault("empirical_soot",d_empirical_soot,true);
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
  else if (mixModel == "StaticMixingTable")
    d_mixingModel = scinew StaticMixingTable();
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
  else if (dynamic_cast<const StaticMixingTable*>(d_mixingModel)) {
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

  // requires scalars
  // will only work for one mixing variable
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  }
  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (d_MAlab) {
#ifdef ExactMPMArchesInitialize
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, Ghost::None,
    		  Arches::ZEROGHOSTCELLS);
#else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
#endif
    tsk->computes(d_lab->d_densityMicroLabel);
  }
  else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_refDensity_label);

  if (d_enthalpySolve)
    tsk->modifies(d_lab->d_enthalpySPLabel);

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
    CCVariable<double> density;
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);
    CCVariable<double> enthalpy;

    new_dw->getModifiable(density, d_lab->d_densityCPLabel, 
		matlIndex, patch);

    if (d_enthalpySolve) {
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel,
			     matlIndex, patch);
      enthalpy.initialize(0.0);
    }
    
    if (d_MAlab) {
#ifdef ExactMPMArchesInitialize
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch, 
    		  Ghost::None, Arches::ZEROGHOSTCELLS);
#else
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    }
    else {
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    }

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
          // This flag ensures properties for heatloss=0.0 during the initialization
          inStream.d_initEnthalpy = true;
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
	  
//	  if (cellType[currCell] != d_bc->wallCellType()) 
	    density[currCell] = local_den;
// d_denRef is usually [-1,-1,-1] so if not do the following, den_ref
// will be messed up since it will be set to 0 because of the condition above
// This error doesn't matter to much unless density underrelaxation is used
//	  if ((patch->containsCell(d_denRef))&&(currCell==d_denRef))
//	    density[currCell] = local_den;
	}
      }
    }
    if ((d_bc->getIntrusionBC())&&(d_reactingFlow||d_flamelet))
      d_bc->intrusionTemperatureBC(pc, patch, cellType, temperature);
    if (patch->containsCell(d_denRef)) {

      double den_ref = density[d_denRef];

      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
    }

    else
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);
    
  }
cout<<"Computation of properties for the first time is completed:"<<endl;
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
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_DORadiationCalc)
    tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);    

  if (d_reactingFlow) {
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_cpINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_co2INLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    /*
    tsk->requires(Task::OldDW, d_lab->d_enthalpyRXNLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    */
    if (d_mixingModel->getNumRxnVars())
      tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_radiationCalc) {
    tsk->requires(Task::OldDW, d_lab->d_absorpINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    //    tsk->requires(Task::OldDW, d_lab->d_abskgINLabel, 
    //		Ghost::None, Arches::ZEROGHOSTCELLS);
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

  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityMicroLabel);

  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_cpINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    /*
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    */
    if (d_mixingModel->getNumRxnVars())
      tsk->computes(d_lab->d_reactscalarSRCINLabel);
  }

  if (d_radiationCalc) {
    tsk->computes(d_lab->d_absorpINLabel);
    //    tsk->computes(d_lab->d_abskgINLabel);
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
    new_dw->getModifiable(density, d_lab->d_densityCPLabel, 
			   matlIndex, patch);

    // Get densityIN from old_dw for consistency with
    // gets/requires of nonlinearSolve (we don't do 
    // anything with this densityIN)

    CCVariable<double> densityCP_old;
    old_dw->getModifiable(densityCP_old, d_lab->d_densityCPLabel, matlIndex, patch);

    CCVariable<double> denMicro_new;
    new_dw->allocateAndPut(denMicro_new, d_lab->d_densityMicroLabel, 
			   matlIndex, patch);
    denMicro_new.copyData(denMicro);

    constCCVariable<double> tempIN;
    constCCVariable<double> cpIN;
    constCCVariable<double> co2IN;
    /*
    constCCVariable<double> enthalpyRXN; 
    */
    constCCVariable<double> reactScalarSrc;
    CCVariable<double> tempIN_new;
    CCVariable<double> cpIN_new;
    CCVariable<double> co2IN_new;
    /*
    CCVariable<double> enthalpyRXN_new;
    */
    CCVariable<double> reactScalarSrc_new;
    constCCVariable<double> solidTemp;

    if (d_DORadiationCalc)
      new_dw->get(solidTemp, d_MAlab->integTemp_CCLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    if (d_reactingFlow) {
      old_dw->get(tempIN, d_lab->d_tempINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(cpIN, d_lab->d_cpINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(co2IN, d_lab->d_co2INLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      /*
      old_dw->get(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      */
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

      /*
      new_dw->allocateAndPut(enthalpyRXN_new, d_lab->d_enthalpyRXNLabel, 
			     matlIndex, patch);
      enthalpyRXN_new.copyData(enthalpyRXN);
      */

      if (d_mixingModel->getNumRxnVars()) {
	new_dw->allocateAndPut(reactScalarSrc_new, d_lab->d_reactscalarSRCINLabel,
			       matlIndex, patch);
	reactScalarSrc_new.copyData(reactScalarSrc);
      }
    }

    constCCVariable<double> absorpIN;
    //    constCCVariable<double> abskgIN;
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
    //    CCVariable<double> abskgIN_new;
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

      //      old_dw->get(abskgIN, d_lab->d_abskgINLabel, matlIndex, patch,
      //		  Ghost::None, Arches::ZEROGHOSTCELLS);

      old_dw->get(sootFVIN, d_lab->d_sootFVINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      new_dw->allocateAndPut(absorpIN_new, d_lab->d_absorpINLabel, 
		       matlIndex, patch);
      absorpIN_new.copyData(absorpIN);

      //      new_dw->allocateAndPut(abskgIN_new, d_lab->d_abskgINLabel, 
      //		       matlIndex, patch);
      //      abskgIN_new.copyData(abskgIN);

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
	    if (d_DORadiationCalc)
	      tempIN_new[currCell] = solidTemp[currCell];
	    //	      tempIN_new[currCell] = 298.0;
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

  }
}
//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls,
				 const TimeIntegratorLabel* timelabels,
			         bool modify_ref_density)
{
  string md = "";
  if (!(modify_ref_density)) md += "RKSSP";
  string taskname =  "Properties::ReComputeProps" +
		     timelabels->integrator_step_name + md;
  Task* tsk = scinew Task(taskname, this,
			  &Properties::reComputeProps,
			  timelabels, modify_ref_density);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_DORadiationCalc)
      tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel, Ghost::None,
		    Arches::ZEROGHOSTCELLS);
  }

  if (d_numMixStatVars > 0) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->modifies(d_lab->d_densityCPLabel);

// assuming ref_density is not changed by RK averaging
  if (modify_ref_density)
    tsk->computes(timelabels->ref_density);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
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
  }
  else {
    tsk->modifies(d_lab->d_drhodfCPLabel);
    if (d_reactingFlow) {
      tsk->modifies(d_lab->d_tempINLabel);
      tsk->modifies(d_lab->d_cpINLabel);
      tsk->modifies(d_lab->d_co2INLabel);
      tsk->modifies(d_lab->d_h2oINLabel);
      tsk->modifies(d_lab->d_enthalpyRXNLabel);
      if (d_mixingModel->getNumRxnVars())
        tsk->modifies(d_lab->d_reactscalarSRCINLabel);
    }
    if (d_flamelet) {
      tsk->modifies(d_lab->d_tempINLabel);
      tsk->modifies(d_lab->d_sootFVINLabel);
      tsk->modifies(d_lab->d_co2INLabel);
      tsk->modifies(d_lab->d_h2oINLabel);
      tsk->modifies(d_lab->d_fvtfiveINLabel);
      tsk->modifies(d_lab->d_tfourINLabel);
      tsk->modifies(d_lab->d_tfiveINLabel);
      tsk->modifies(d_lab->d_tnineINLabel);
      tsk->modifies(d_lab->d_qrgINLabel);
      tsk->modifies(d_lab->d_qrsINLabel);
    }
    if (d_radiationCalc) {
      tsk->modifies(d_lab->d_absorpINLabel);
      tsk->modifies(d_lab->d_sootFVINLabel);
    }
  }

  if (d_MAlab) 
    tsk->computes(d_lab->d_densityMicroLabel);

  sched->addTask(tsk, patches, matls);
}

  
//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::reComputeProps(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw,
			   const TimeIntegratorLabel* timelabels,
			   bool modify_ref_density)
{
  for (int p = 0; p < patches->size(); p++) {

    TAU_PROFILE_TIMER(input, "Input", "[Properties::reCompute::input]" , TAU_USER);
    TAU_PROFILE_TIMER(compute, "Compute", "[Properties::reCompute::compute]" , TAU_USER);
    TAU_PROFILE_TIMER(mixing, "Mixing", "[Properties::reCompute::mixing]" , TAU_USER);

    TAU_PROFILE_START(input);

    double start_mixTime = Time::currentSeconds();

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
		     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<int> cellType;
    constCCVariable<double> density;
    StaticArray<constCCVariable<double> > scalar(d_numMixingVars);
    StaticArray<constCCVariable<double> > scalarVar(d_numMixStatVars);
    StaticArray<constCCVariable<double> > reactScalar(d_mixingModel->getNumRxnVars());
    constCCVariable<double> enthalpy;
    constCCVariable<double> voidFraction;
    CCVariable<double> new_density;
    CCVariable<double> temperature;
    CCVariable<double> cp;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> enthalpyRXN;
    CCVariable<double> reactscalarSRC;
    CCVariable<double> drhodf;
    CCVariable<double> absorption;
    CCVariable<double> sootFV;
    CCVariable<double> fvtfive;
    CCVariable<double> tfour;
    CCVariable<double> tfive;
    CCVariable<double> tnine;
    CCVariable<double> qrg;
    CCVariable<double> qrs;
    CCVariable<double> denMicro;
    constCCVariable<double> solidTemp;

    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);

    old_dw->get(density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
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
      new_dw->get(enthalpy, d_lab->d_enthalpySPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->getModifiable(new_density, d_lab->d_densityCPLabel,
			  matlIndex, patch);
    new_density.initialize(0.0);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(drhodf, d_lab->d_drhodfCPLabel, matlIndex, patch);

      if (d_reactingFlow) {
        new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel, 
			       matlIndex, patch);
        new_dw->allocateAndPut(cp, d_lab->d_cpINLabel, matlIndex, patch);
        new_dw->allocateAndPut(co2, d_lab->d_co2INLabel, matlIndex, patch);
        new_dw->allocateAndPut(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
        new_dw->allocateAndPut(enthalpyRXN, d_lab->d_enthalpyRXNLabel,
			       matlIndex, patch);
        if (d_mixingModel->getNumRxnVars())
	  new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			         matlIndex, patch);
      }

      if (d_flamelet) {
        new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel,
			       matlIndex, patch);
        new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, matlIndex,patch);
        new_dw->allocateAndPut(co2, d_lab->d_co2INLabel, matlIndex, patch);
        new_dw->allocateAndPut(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
        new_dw->allocateAndPut(fvtfive, d_lab->d_fvtfiveINLabel,
			       matlIndex, patch);
        new_dw->allocateAndPut(tfour, d_lab->d_tfourINLabel, matlIndex, patch);
        new_dw->allocateAndPut(tfive, d_lab->d_tfiveINLabel, matlIndex, patch);
        new_dw->allocateAndPut(tnine, d_lab->d_tnineINLabel, matlIndex, patch);
        new_dw->allocateAndPut(qrg, d_lab->d_qrgINLabel, matlIndex, patch);
        new_dw->allocateAndPut(qrs, d_lab->d_qrsINLabel, matlIndex, patch);
      }

      if (d_radiationCalc) {
        new_dw->allocateAndPut(absorption, d_lab->d_absorpINLabel,
			       matlIndex, patch);
        new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, matlIndex,patch);
      }
    }
    else {
      new_dw->getModifiable(drhodf, d_lab->d_drhodfCPLabel, matlIndex, patch);

      if (d_reactingFlow) {
        new_dw->getModifiable(temperature, d_lab->d_tempINLabel, 
			       matlIndex, patch);
        new_dw->getModifiable(cp, d_lab->d_cpINLabel, matlIndex, patch);
        new_dw->getModifiable(co2, d_lab->d_co2INLabel, matlIndex, patch);
        new_dw->getModifiable(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
        new_dw->getModifiable(enthalpyRXN, d_lab->d_enthalpyRXNLabel,
			       matlIndex, patch);
        if (d_mixingModel->getNumRxnVars())
	  new_dw->getModifiable(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			         matlIndex, patch);
      }

      if (d_flamelet) {
        new_dw->getModifiable(temperature, d_lab->d_tempINLabel,
			       matlIndex, patch);
        new_dw->getModifiable(sootFV, d_lab->d_sootFVINLabel, matlIndex,patch);
        new_dw->getModifiable(co2, d_lab->d_co2INLabel, matlIndex, patch);
        new_dw->getModifiable(h2o, d_lab->d_h2oINLabel, matlIndex, patch);
        new_dw->getModifiable(fvtfive, d_lab->d_fvtfiveINLabel,
			       matlIndex, patch);
        new_dw->getModifiable(tfour, d_lab->d_tfourINLabel, matlIndex, patch);
        new_dw->getModifiable(tfive, d_lab->d_tfiveINLabel, matlIndex, patch);
        new_dw->getModifiable(tnine, d_lab->d_tnineINLabel, matlIndex, patch);
        new_dw->getModifiable(qrg, d_lab->d_qrgINLabel, matlIndex, patch);
        new_dw->getModifiable(qrs, d_lab->d_qrsINLabel, matlIndex, patch);
      }

      if (d_radiationCalc) {
        new_dw->getModifiable(absorption, d_lab->d_absorpINLabel,
			       matlIndex, patch);
        new_dw->getModifiable(sootFV, d_lab->d_sootFVINLabel, matlIndex,patch);
      }
    }

    drhodf.initialize(0.0);
    if (d_reactingFlow) {
      temperature.initialize(0.0); 
      cp.initialize(0.0);
      co2.initialize(0.0);
      h2o.initialize(0.0);
      enthalpyRXN.initialize(0.0);
        if (d_mixingModel->getNumRxnVars())
	  reactscalarSRC.initialize(0.0);
    }    
    if (d_flamelet) {
      temperature.initialize(0.0);
      sootFV.initialize(0.0);
      co2.initialize(0.0);
      h2o.initialize(0.0);
      fvtfive.initialize(0.0);
      tfour.initialize(0.0);
      tfive.initialize(0.0);
      tnine.initialize(0.0);
      qrg.initialize(0.0);
      qrs.initialize(0.0);
    }
    if (d_radiationCalc) {
      absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch);
      if (d_DORadiationCalc)
	new_dw->get(solidTemp, d_MAlab->integTemp_CCLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    TAU_PROFILE_STOP(input);
    TAU_PROFILE_START(compute);
    InletStream inStream(d_numMixingVars,
		         d_mixingModel->getNumMixStatVars(),
		         d_mixingModel->getNumRxnVars());
    Stream outStream;
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);

	  // construct an InletStream for input to the computeProps
	  // of mixingModel
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

	  // currently not using any reaction progress variables

	  if ((!d_mixingModel->isAdiabatic()))
	      //	      &&(cellType[currCell] != d_bc->getIntrusionID()))
	    inStream.d_enthalpy = enthalpy[currCell];
	  else
	    inStream.d_enthalpy = 0.0;
          inStream.d_initEnthalpy = false;

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
	    enthalpyRXN[currCell] = outStream.getEnthalpy();
	    if (d_mixingModel->getNumRxnVars())
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  

	  if (d_radiationCalc) {
	    // bc is the mass-atoms 0f carbon per mas of reactnat mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    if (d_mixingModel->getNumRxnVars()) 
	      sootFV[currCell] = outStream.getSootFV();
	    else {
	      if (d_empirical_soot) {
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
	      else sootFV[currCell] = 0.0;
	    }  
	    absorption[currCell] = 0.01+ Min(0.5,(4.0/d_opl)*log(1.0+350.0*
				   sootFV[currCell]*temperature[currCell]*d_opl));
	  }
	  // check if the density is greater than air...implement a better way
          /*if (d_DORadiationCalc) {
	    double cutoff_air_density = 1.1845;
	    double cutoff_temperature = 298.0;
	    if ((scalar[0])[currCell] < 0.4) {
	      if (local_den > cutoff_air_density) {
	        local_den = cutoff_air_density;
	        temperature[currCell] = cutoff_temperature;
	      }
	    }
	  }*/

	  if (d_MAlab) {
	    denMicro[currCell] = local_den;
	    local_den *= voidFraction[currCell];
	    // no under-relaxation for MPMArches
	    new_density[currCell] = local_den;
	  }
	  else {
	    new_density[currCell] = d_denUnderrelax * local_den +
	      (1.0-d_denUnderrelax) * density[currCell];
	  }
	}
      }
    }
    // Write the computed density to the new data warehouse
    if (modify_ref_density) {
      double den_ref = 0.0;
      if (patch->containsCell(d_denRef)) {
        den_ref = new_density[d_denRef];
        cerr << "density_ref " << den_ref << endl;
      }
      new_dw->put(sum_vartype(den_ref),timelabels->ref_density);
    }

    if ((d_bc->getIntrusionBC())&&(d_reactingFlow||d_flamelet))
      d_bc->intrusionTemperatureBC(pc, patch, cellType, temperature);

    if (d_MAlab && d_DORadiationCalc)
      d_bc->mmWallTemperatureBC(pc, patch, cellType, solidTemp, temperature);

    if (pc->myrank() == 0)
      cerr << "Time in the Mixing Model: " << 
	   Time::currentSeconds()-start_mixTime << " seconds\n";

  TAU_PROFILE_STOP(compute);
  }
}
//****************************************************************************
// Schedule the computation of density reference array here
//****************************************************************************
void 
Properties::sched_computeDenRefArray(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls,
			             const TimeIntegratorLabel* timelabels)

{

  // primitive variable initialization

  string taskname =  "Properties::computeDenRefArray" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
			  this, &Properties::computeDenRefArray,
			  timelabels);


  tsk->requires(Task::NewDW, timelabels->ref_density);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->computes(d_lab->d_denRefArrayLabel);
  else
    tsk->modifies(d_lab->d_denRefArrayLabel);

  sched->addTask(tsk, patches, matls);

}
//****************************************************************************
// Actually calculate the density reference array here
//****************************************************************************

void 
Properties::computeDenRefArray(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset*,
			       DataWarehouse*,
			       DataWarehouse* new_dw,
			       const TimeIntegratorLabel* timelabels)

{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> denRefArray;
    constCCVariable<double> voidFraction;


    sum_vartype den_ref_var;
    new_dw->get(den_ref_var, timelabels->ref_density);

    double den_Ref = den_ref_var;

    //cerr << "getdensity_ref " << den_Ref << endl;

    if (d_MAlab) {

      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    }

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      new_dw->allocateAndPut(denRefArray, d_lab->d_denRefArrayLabel, 
		             matlIndex, patch);
    else
      new_dw->getModifiable(denRefArray, d_lab->d_denRefArrayLabel, 
		            matlIndex, patch);
		
    denRefArray.initialize(den_Ref);

    if (d_MAlab) {

      for (CellIterator iter = patch->getCellIterator();
	   !iter.done();iter++){

	denRefArray[*iter]  *= voidFraction[*iter];

      }
    }
  }
}

//****************************************************************************
// Schedule the averaging of properties for Runge-Kutta step
//****************************************************************************
void 
Properties::sched_averageRKProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls,
				 const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::averageRKProps" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &Properties::averageRKProps,
			  timelabels);

  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_mixingModel->getNumRxnVars())
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (!(d_mixingModel->isAdiabatic()))
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->modifies(d_lab->d_scalarSPLabel);
  if (d_mixingModel->getNumRxnVars())
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  if (!(d_mixingModel->isAdiabatic()))
    tsk->modifies(d_lab->d_enthalpySPLabel);
  tsk->modifies(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually average the Runge-Kutta properties here
//****************************************************************************
void 
Properties::averageRKProps(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw,
			   const TimeIntegratorLabel* timelabels)
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
    CCVariable<double> density_guess;

    old_dw->get(old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    for (int ii = 0; ii < d_numMixingVars; ii++) {
      old_dw->get(old_scalar[ii], d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (d_mixingModel->getNumRxnVars() > 0) {
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++) {
	old_dw->get(old_reactScalar[ii], d_lab->d_reactscalarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      }
    }
    if (!(d_mixingModel->isAdiabatic())) {
      old_dw->get(old_enthalpy, d_lab->d_enthalpySPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    new_dw->get(rho1_density, d_lab->d_densityTempLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    for (int ii = 0; ii < d_numMixingVars; ii++)
      new_dw->getModifiable(new_scalar[ii], d_lab->d_scalarSPLabel, 
		            matlIndex, patch);
    if (d_mixingModel->getNumRxnVars() > 0)
      for (int ii = 0; ii < d_mixingModel->getNumRxnVars(); ii++)
	new_dw->getModifiable(new_reactScalar[ii], d_lab->d_reactscalarSPLabel,
		    	      matlIndex, patch);
    if (!(d_mixingModel->isAdiabatic()))
      new_dw->getModifiable(new_enthalpy, d_lab->d_enthalpySPLabel, 
			    matlIndex, patch);

    new_dw->getModifiable(density_guess, d_lab->d_densityGuessLabel, 
			  matlIndex, patch);

    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
          
          double predicted_density;
	  if (old_density[currCell] > 0.0)
//            predicted_density = rho1_density[currCell];
//            predicted_density = 1.0/((factor_old/old_density[currCell] +
//			       factor_new/new_density[currCell])/factor_divide);
            predicted_density = (factor_old*old_density[currCell] +
			       factor_new*new_density[currCell])/factor_divide;
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

	  density_guess[currCell] = predicted_density;

	}
      }
    }
  }
}

//****************************************************************************
// Schedule saving of temp density
//****************************************************************************
void 
Properties::sched_saveTempDensity(SchedulerP& sched, const PatchSet* patches,
				  const MaterialSet* matls,
			   	  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::saveTempDensity" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &Properties::saveTempDensity,
			  timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
 
  tsk->modifies(d_lab->d_densityTempLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp density here
//****************************************************************************
void 
Properties::saveTempDensity(const ProcessorGroup*,
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

    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,
			  matlIndex, patch);
     new_dw->copyOut(temp_density, d_lab->d_densityCPLabel,
		     matlIndex, patch);

  }
}
//****************************************************************************
// Schedule the computation of drhodt
//****************************************************************************
void 
Properties::sched_computeDrhodt(SchedulerP& sched, const PatchSet* patches,
				const MaterialSet* matls, 
			        const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::computeDrhodt" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &Properties::computeDrhodt,
			  timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(parent_old_dw, d_lab->d_oldDeltaTLabel);

  tsk->requires(parent_old_dw, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(parent_old_dw, d_lab->d_densityOldOldLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
     tsk->computes(d_lab->d_filterdrhodtLabel);
     tsk->computes(d_lab->d_oldDeltaTLabel);
     tsk->computes(d_lab->d_densityOldOldLabel);
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
			  const TimeIntegratorLabel* timelabels)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  int drhodt_1st_order = 1;
  int current_step = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  if (d_MAlab) drhodt_1st_order = 2;
  delt_vartype delT, old_delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    new_dw->put(delT, d_lab->d_oldDeltaTLabel);
  double delta_t = delT;
  parent_old_dw->get(old_delT, d_lab->d_oldDeltaTLabel);
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
    CCVariable<double> density_oldold;

    parent_old_dw->get(old_density, d_lab->d_densityCPLabel, matlIndex, patch,
	        Ghost::None, Arches::ZEROGHOSTCELLS);
    parent_old_dw->get(old_old_density, d_lab->d_densityOldOldLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    new_dw->allocateAndPut(density_oldold, d_lab->d_densityOldOldLabel, matlIndex, patch);
    density_oldold.copyData(old_density);
    }

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(new_density, d_lab->d_densityCPLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
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
    drhodt.initialize(0.0);

    if (current_step <= drhodt_1st_order) {
// 1st order drhodt
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	    IntVector currcell(ii,jj,kk);

	    double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
	    drhodt[currcell] = (new_density[currcell] -
				old_density[currcell])*vol/delta_t;
          }
        }
      }
    }
    else {
// 2nd order drhodt, assuming constant volume
      double factor = 1.0 + old_delta_t/delta_t;
      double new_factor = factor * factor - 1.0;
      double old_factor = factor * factor;
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	    IntVector currcell(ii,jj,kk);
	    IntVector xminus(ii-1,jj,kk);

	    double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
	    drhodt[currcell] = (new_factor*new_density[currcell] -
				old_factor*old_density[currcell] +
				old_old_density[currcell])*vol /
			       (old_delta_t*factor);
          }
        }
      }
    }

    if (d_filter_drhodt) {
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch, drhodt, filterdrhodt);
#else
    // filtering without petsc is not implemented
    // if it needs to be then drhodt will have to be computed with ghostcells
    filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
		      drhodt.getHighIndex());
#endif
    }
    else
    filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
		      drhodt.getHighIndex());
  }
}
