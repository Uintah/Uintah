//----- Properties.cc --------------------------------------------------
#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/NewStaticMixingTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StandardTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties(const ArchesLabel* label, const MPMArchesLabel* MAlb,
                       PhysicalConstants* phys_const, bool calcReactingScalar,
		       bool calcEnthalpy, bool calcVariance):
                       d_lab(label), d_MAlab(MAlb), 
                       d_physicalConsts(phys_const), 
                       d_calcReactingScalar(calcReactingScalar),
                       d_calcEnthalpy(calcEnthalpy),
                       d_calcVariance(calcVariance)
{
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
  db->getWithDefault("filter_drhodt",d_filter_drhodt,false);
  db->getWithDefault("first_order_drhodt",d_first_order_drhodt,true);
  db->getWithDefault("inverse_density_average",d_inverse_density_average,false);
  d_denRef = d_physicalConsts->getRefPoint();
  d_reactingFlow = true;
  // read type of mixing model
  string mixModel;
  db->require("mixing_model",mixModel);
  if (mixModel == "coldFlowMixingModel") {
    d_mixingModel = scinew ColdflowMixingModel(d_calcReactingScalar,
                                               d_calcEnthalpy,
                                               d_calcVariance);
    d_reactingFlow = false;
  }
  else if (mixModel == "NewStaticMixingTable")
    d_mixingModel = scinew NewStaticMixingTable(d_calcReactingScalar,
                                                d_calcEnthalpy,
                                                d_calcVariance);
  else if (mixModel == "StandardTable")
    d_mixingModel = scinew StandardTable(d_calcReactingScalar,
                                         d_calcEnthalpy,
                                         d_calcVariance);
  
  else if (mixModel == "pdfMixingModel" || mixModel == "SteadyFlameletsTable"
	|| mixModel == "flameletModel"  || mixModel == "StaticMixingTable"
	|| mixModel == "meanMixingModel" )
    throw InvalidValue("DEPRECATED: Mixing Model no longer supported: " + mixModel, __FILE__, __LINE__);
  else
    throw InvalidValue("Mixing Model not supported" + mixModel, __FILE__, __LINE__);
  d_mixingModel->problemSetup(db);
  if (d_calcEnthalpy)
    d_H_air = d_mixingModel->getAdiabaticAirEnthalpy();


  d_co_output = d_mixingModel->getCOOutput();
  d_sulfur_chem = d_mixingModel->getSulfurChem();
  d_soot_precursors = d_mixingModel->getSootPrecursors();

  cout << "d_co_output "<< d_co_output << endl;

  if (d_calcEnthalpy) {
    ProblemSpecP params_non_constant = params;
    const ProblemSpecP params_root = params_non_constant->getRootNode();
    ProblemSpecP db_enthalpy_solver=params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver");
  
    if (!db_enthalpy_solver)
      db_enthalpy_solver=params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("PicardSolver");
    if (!db_enthalpy_solver) {
      ostringstream exception;
      exception << "Radiation information needed by Properties " <<
                   "is not found in EnthalpySolver block " <<
                   "for PicardSolver or ExplicitSolver" << endl;
      throw ProblemSetupException(exception.str(), __FILE__, __LINE__);
    }
    db_enthalpy_solver = db_enthalpy_solver->findBlock("EnthalpySolver");

    db_enthalpy_solver->require("radiation",d_radiationCalc);
    if (d_radiationCalc) {
      db_enthalpy_solver->getWithDefault("discrete_ordinates",d_DORadiationCalc,true);
      d_opl = 0.0;
      if (!d_DORadiationCalc)
        db->require("optically_thin_model_opl",d_opl);
      db->getWithDefault("empirical_soot",d_empirical_soot,true);
      if (d_empirical_soot)
        db->getWithDefault("SootFactor",d_sootFactor,1.0);
    }
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
  else if (dynamic_cast<const NewStaticMixingTable*>(d_mixingModel)) {
    d_mixingModel->computeProps(inStream, outStream);
  }
  else if (dynamic_cast<const StandardTable*>(d_mixingModel)) {
    d_mixingModel->computeProps(inStream, outStream);
  }
  else {
    throw InvalidValue("Mixing Model not supported", __FILE__, __LINE__);
  }
}
  

//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls,
				 const TimeIntegratorLabel* timelabels,
			         bool modify_ref_density, bool initialize)
{
  string md = "";
  if (!(modify_ref_density)) md += "RKSSP";
  string taskname =  "Properties::ReComputeProps" +
		     timelabels->integrator_step_name + md;
  Task* tsk = scinew Task(taskname, this,
			  &Properties::reComputeProps,
			  timelabels, modify_ref_density, initialize);

  tsk->modifies(d_lab->d_scalarSPLabel);


  if (d_calcVariance)
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_calcReactingScalar)
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_calcEnthalpy)
    tsk->modifies(d_lab->d_enthalpySPLabel);

  if (d_MAlab && initialize) {
#ifdef ExactMPMArchesInitialize
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, Ghost::None,
    		  Arches::ZEROGHOSTCELLS);
#else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
#endif
  }
  else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

  if (d_MAlab && !initialize) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, Ghost::None,
    		  Arches::ZEROGHOSTCELLS);
  if (d_DORadiationCalc && d_bc->getIfCalcEnergyExchange())
    tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
  }

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
      tsk->computes(d_lab->d_normalizedScalarVarLabel);
      tsk->computes(d_lab->d_heatLossLabel);
      tsk->computes(d_lab->d_enthalpyRXNLabel);
      if (d_calcReactingScalar)
        tsk->computes(d_lab->d_reactscalarSRCINLabel);
    }

    if (d_co_output) 
      tsk->computes(d_lab->d_coINLabel);
    if (d_sulfur_chem) {
      tsk->computes(d_lab->d_h2sINLabel);
      tsk->computes(d_lab->d_so2INLabel);
      tsk->computes(d_lab->d_so3INLabel);
    }
    if (d_soot_precursors) {
      tsk->computes(d_lab->d_c2h2INLabel);
      tsk->computes(d_lab->d_ch4INLabel);
    }

    if (d_radiationCalc) {
      if (!d_DORadiationCalc)
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
      tsk->modifies(d_lab->d_normalizedScalarVarLabel);
      tsk->modifies(d_lab->d_heatLossLabel);
      tsk->modifies(d_lab->d_enthalpyRXNLabel);
      if (d_calcReactingScalar)
        tsk->modifies(d_lab->d_reactscalarSRCINLabel);
    }

    if (d_co_output) 
      tsk->modifies(d_lab->d_coINLabel);
    if (d_sulfur_chem) {
      tsk->modifies(d_lab->d_h2sINLabel);
      tsk->modifies(d_lab->d_so2INLabel);
      tsk->modifies(d_lab->d_so3INLabel);
    }
    if (d_soot_precursors) {
      tsk->modifies(d_lab->d_c2h2INLabel);
      tsk->modifies(d_lab->d_ch4INLabel);
    }

    if (d_radiationCalc) {
      if (!d_DORadiationCalc)
        tsk->modifies(d_lab->d_absorpINLabel);
      tsk->modifies(d_lab->d_sootFVINLabel);
    }
  }

  if (d_MAlab) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      tsk->computes(d_lab->d_densityMicroLabel);
    else
      tsk->modifies(d_lab->d_densityMicroLabel);
  }
  sched->addTask(tsk, patches, matls);
}

  
//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::reComputeProps(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse* ,
			   DataWarehouse* new_dw,
			   const TimeIntegratorLabel* timelabels,
			   bool modify_ref_density,
			   bool initialize)
{
  if (d_bc == 0)
    throw InvalidValue("BoundaryCondition pointer not assigned", __FILE__, __LINE__);

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
    CCVariable<double> scalar;
    constCCVariable<double> scalarVar;
    constCCVariable<double> reactScalar;
    constCCVariable<double> scalarDisp;
    constCCVariable<double> voidFraction;
    CCVariable<double> new_density;
    CCVariable<double> temperature;
    CCVariable<double> cp;
    CCVariable<double> co2;
    CCVariable<double> h2o;
    CCVariable<double> normalizedScalarVar;
    CCVariable<double> heatLoss;
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
    CCVariable<double> enthalpy;

    CCVariable<double> co;
    CCVariable<double> h2s;
    CCVariable<double> so2;
    CCVariable<double> so3;
    CCVariable<double> c2h2;
    CCVariable<double> ch4;


    if (d_MAlab && initialize) {
#ifdef ExactMPMArchesInitialize
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch, 
    		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#else
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    }
    else
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, 
		  matlIndex, patch);

    if (d_calcVariance) {
      new_dw->get(scalarVar, d_lab->d_scalarVarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    
    if (d_calcReactingScalar) {
	new_dw->get(reactScalar, d_lab->d_reactscalarSPLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    if (d_calcEnthalpy)
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel, 
		            matlIndex, patch);

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
        new_dw->allocateAndPut(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, matlIndex, patch);
        new_dw->allocateAndPut(heatLoss, d_lab->d_heatLossLabel, matlIndex, patch);
        new_dw->allocateAndPut(enthalpyRXN, d_lab->d_enthalpyRXNLabel,
			       matlIndex, patch);
        if (d_calcReactingScalar)
	  new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			         matlIndex, patch);
      }

      if (d_co_output)
	new_dw->allocateAndPut(co, d_lab->d_coINLabel,
			       matlIndex, patch);
      if (d_sulfur_chem) {
	new_dw->allocateAndPut(h2s, d_lab->d_h2sINLabel,
			       matlIndex, patch);
	new_dw->allocateAndPut(so2, d_lab->d_so2INLabel,
			       matlIndex, patch);
	new_dw->allocateAndPut(so3, d_lab->d_so3INLabel,
			       matlIndex, patch);
      }
      if (d_soot_precursors) {
        new_dw->allocateAndPut(c2h2, d_lab->d_c2h2INLabel,
                               matlIndex, patch);
        new_dw->allocateAndPut(ch4, d_lab->d_ch4INLabel,
                               matlIndex, patch);
      }

      if (d_radiationCalc) {
        if (!d_DORadiationCalc)
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
        new_dw->getModifiable(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, matlIndex, patch);
        new_dw->getModifiable(heatLoss, d_lab->d_heatLossLabel, matlIndex, patch);
        new_dw->getModifiable(enthalpyRXN, d_lab->d_enthalpyRXNLabel,
			       matlIndex, patch);
        if (d_calcReactingScalar)
	  new_dw->getModifiable(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
			         matlIndex, patch);
      }

      if (d_co_output)
	new_dw->getModifiable(co, d_lab->d_coINLabel,
			      matlIndex, patch);
      if (d_sulfur_chem) {
	new_dw->getModifiable(h2s, d_lab->d_h2sINLabel,
			      matlIndex, patch);
	new_dw->getModifiable(so2, d_lab->d_so2INLabel,
			      matlIndex, patch);
	new_dw->getModifiable(so3, d_lab->d_so3INLabel,
			      matlIndex, patch);
      }
      if (d_soot_precursors) {
        new_dw->getModifiable(c2h2, d_lab->d_c2h2INLabel,
                              matlIndex, patch);
        new_dw->getModifiable(ch4, d_lab->d_ch4INLabel,
                              matlIndex, patch);
      }

      if (d_radiationCalc) {
        if (!d_DORadiationCalc)
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
      normalizedScalarVar.initialize(0.0);
      heatLoss.initialize(0.0);
      enthalpyRXN.initialize(0.0);
      if (d_calcReactingScalar)
	  reactscalarSRC.initialize(0.0);
    }    

    if (d_co_output)
      co.initialize(0.0);
    if (d_sulfur_chem) {
      h2s.initialize(0.0);
      so2.initialize(0.0);
      so3.initialize(0.0);
    }
    if (d_soot_precursors) {
      c2h2.initialize(0.0);
      ch4.initialize(0.0);
    }

    if (d_radiationCalc) {
      if (!d_DORadiationCalc)
        absorption.initialize(0.0);
      sootFV.initialize(0.0);
    }

    if (d_MAlab && !initialize) {
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      if (d_DORadiationCalc && d_bc->getIfCalcEnergyExchange())
	new_dw->get(solidTemp, d_MAlab->integTemp_CCLabel, 
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    if (d_MAlab) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
	new_dw->allocateAndPut(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch);
      else
	new_dw->getModifiable(denMicro, d_lab->d_densityMicroLabel, matlIndex, patch);
    }

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    TAU_PROFILE_STOP(input);
    TAU_PROFILE_START(compute);
    int variance_count = d_calcVariance;
    int reacting_scalar_count = d_calcReactingScalar;
    InletStream inStream(1,
		         variance_count,
		         reacting_scalar_count);
    Stream outStream;

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);

	  // construct an InletStream for input to the computeProps
	  // of mixingModel
	  bool local_enthalpy_init;
	  if (d_calcEnthalpy && (scalar[currCell] == -1.0)) {
	    scalar[currCell] = 0.0;
	    local_enthalpy_init = true;
          }
	  else
	    local_enthalpy_init = false;
	  
	  inStream.d_currentCell = currCell;
	  
	  inStream.d_mixVars[0] = scalar[currCell];

	  if (d_calcVariance) {
	    inStream.d_mixVarVariance[0] = scalarVar[currCell];
	  }

	  // currently not using any reaction progress variables
	  if (d_calcReactingScalar) {
	      inStream.d_rxnVars[0] = reactScalar[currCell];
	  }


          if (d_calcEnthalpy)
	      //	      &&(cellType[currCell] != d_bc->getIntrusionID()))
            if (initialize || local_enthalpy_init)
	      inStream.d_enthalpy = 0.0;
	    else
	      inStream.d_enthalpy = enthalpy[currCell];
	  else
	    inStream.d_enthalpy = 0.0;
           // This flag ensures properties for heatloss=0.0
	   // during the initialization
           inStream.d_initEnthalpy = (initialize || local_enthalpy_init);

  TAU_PROFILE_START(mixing);
	  d_mixingModel->computeProps(inStream, outStream);
  TAU_PROFILE_STOP(mixing);

	  double local_den = outStream.getDensity();
	  drhodf[currCell] = outStream.getdrhodf();

	  if (d_co_output) {
	    co[currCell] = outStream.getCO();
	  }
	  if (d_sulfur_chem) {
	    h2s[currCell] = outStream.getH2S();
	    so2[currCell] = outStream.getSO2();
	    so3[currCell] = outStream.getSO3();
	  }
          if (d_soot_precursors) {
            c2h2[currCell] = outStream.getC2H2();
            ch4[currCell] = outStream.getCH4();
          }

	  if (d_calcEnthalpy && (initialize || local_enthalpy_init))
	    enthalpy[currCell] = outStream.getEnthalpy();
	  if (d_reactingFlow) {
	    temperature[currCell] = outStream.getTemperature();
	    cp[currCell] = outStream.getCP();
	    co2[currCell] = outStream.getCO2();
	    h2o[currCell] = outStream.getH2O();
	    normalizedScalarVar[currCell] = outStream.getnormalizedScalarVar();
	    heatLoss[currCell] = outStream.getheatLoss();
	    enthalpyRXN[currCell] = outStream.getEnthalpy();
// Uncomment the next line to check enthalpy transport in adiabatic case
	    enthalpyRXN[currCell] -= enthalpy[currCell];
	    if (d_calcReactingScalar)
	      reactscalarSRC[currCell] = outStream.getRxnSource();
	  }
	  

	  if (d_radiationCalc) {
	    // bc is the mass-atoms of carbon per mass of reactant mixture
	    // taken from radcoef.f
	    //	double bc = d_mixingModel->getCarbonAtomNumber(inStream)*local_den;
	    if (d_calcReactingScalar) 
	      sootFV[currCell] = outStream.getSootFV();
	    else {
	      if (d_empirical_soot) {
	        if (temperature[currCell] > 1000) {
		  double bc = inStream.d_mixVars[0]*(84.0/100.0)*local_den;
		  double c3 = 0.1;
		  double rhosoot = 1950.0;
		  double cmw = 12.0;

		  if (inStream.d_mixVars[0] > 0.1)
		    sootFV[currCell] = c3*bc*cmw/rhosoot*d_sootFactor;
		  else
		    sootFV[currCell] = 0.0;
	        }
	        else 
		  sootFV[currCell] = 0.0;
	      }
	      else sootFV[currCell] = 0.0;
	    }  
            if (!d_DORadiationCalc)
	      absorption[currCell] = 0.01+ Min(0.5,(4.0/d_opl)*log(1.0+350.0*
				     sootFV[currCell]*temperature[currCell]*d_opl));
	  }

	  if (d_MAlab) {
	    denMicro[currCell] = local_den;
	    if (initialize) {
#ifdef ExactMPMArchesInitialize
	      local_den *= voidFraction[currCell];
#endif
	    }
	    else
	      local_den *= voidFraction[currCell];
	  }
	  // density underrelaxation is bogus here and has been removed
	  new_density[currCell] = local_den;

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

    if ((d_bc->getIntrusionBC())&& d_reactingFlow )
      d_bc->intrusionTemperatureBC(pc, patch, cellType, temperature);

    if (d_MAlab && d_DORadiationCalc && !initialize) {
      bool d_energyEx = d_bc->getIfCalcEnergyExchange();
      d_bc->mmWallTemperatureBC(pc, patch, cellType, solidTemp, temperature,
				d_energyEx);
    }

    if (pc->myrank() == 0)
      cerr << "Time in the Mixing Model: " << 
	   Time::currentSeconds()-start_mixTime << " seconds\n";

  TAU_PROFILE_STOP(compute);
  }
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

  if (d_bc->getIfCalcEnergyExchange()) 
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
    tsk->requires(Task::OldDW, d_lab->d_normalizedScalarVarLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_heatLossLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    /*
    tsk->requires(Task::OldDW, d_lab->d_enthalpyRXNLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    */
    if (d_calcReactingScalar)
      tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  }

  if (d_radiationCalc) {
    if (!d_DORadiationCalc)
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
    tsk->requires(Task::OldDW, d_lab->d_abskgINLabel,
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
    tsk->computes(d_lab->d_normalizedScalarVarLabel);
    tsk->computes(d_lab->d_heatLossLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    if (d_calcReactingScalar)
      tsk->computes(d_lab->d_reactscalarSRCINLabel);
  }

  if (d_radiationCalc) {
    if (!d_DORadiationCalc)
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
    tsk->computes(d_lab->d_abskgINLabel);
    }
  }

  if (d_co_output) {
    tsk->requires(Task::OldDW, d_lab->d_coINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_coINLabel);
  }

  if (d_sulfur_chem) {
    tsk->requires(Task::OldDW, d_lab->d_h2sINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_so2INLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_so3INLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->computes(d_lab->d_h2sINLabel);
    tsk->computes(d_lab->d_so2INLabel);
    tsk->computes(d_lab->d_so3INLabel);
  }

  if (d_soot_precursors) {
    tsk->requires(Task::OldDW, d_lab->d_c2h2INLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_ch4INLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->computes(d_lab->d_c2h2INLabel);
    tsk->computes(d_lab->d_ch4INLabel);
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
    constCCVariable<double> normalizedScalarVar;
    constCCVariable<double> heatLoss;

    constCCVariable<double> coIN;
    constCCVariable<double> h2sIN;
    constCCVariable<double> so2IN;
    constCCVariable<double> so3IN;
    constCCVariable<double> c2h2IN;
    constCCVariable<double> ch4IN;

    CCVariable<double> coIN_new;
    CCVariable<double> h2sIN_new;
    CCVariable<double> so2IN_new;
    CCVariable<double> so3IN_new;
    CCVariable<double> c2h2IN_new;
    CCVariable<double> ch4IN_new;

    /*
    constCCVariable<double> enthalpyRXN; 
    */
    constCCVariable<double> reactScalarSrc;
    CCVariable<double> tempIN_new;
    CCVariable<double> cpIN_new;
    CCVariable<double> co2IN_new;
    CCVariable<double> normalizedScalarVar_new;
    CCVariable<double> heatLoss_new;
    CCVariable<double> enthalpyRXN_new;
    CCVariable<double> reactScalarSrc_new;
    constCCVariable<double> solidTemp;

    if (d_bc->getIfCalcEnergyExchange())
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
      old_dw->get(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      old_dw->get(heatLoss, d_lab->d_heatLossLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      /*
      old_dw->get(enthalpyRXN, d_lab->d_enthalpyRXNLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      */
      if (d_calcReactingScalar) {
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
      new_dw->allocateAndPut(normalizedScalarVar_new, d_lab->d_normalizedScalarVarLabel, 
			     matlIndex, patch);
      normalizedScalarVar_new.copyData(normalizedScalarVar);
      new_dw->allocateAndPut(heatLoss_new, d_lab->d_heatLossLabel, 
			     matlIndex, patch);
      heatLoss_new.copyData(heatLoss);

      new_dw->allocateAndPut(enthalpyRXN_new, d_lab->d_enthalpyRXNLabel, 
			     matlIndex, patch);
      enthalpyRXN_new.initialize(0.0);

      if (d_calcReactingScalar) {
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
    constCCVariable<double> abskg;
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
    CCVariable<double> abskg_new;
    if (d_radiationCalc) {

      if (!d_DORadiationCalc) {
        old_dw->get(absorpIN, d_lab->d_absorpINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
        new_dw->allocateAndPut(absorpIN_new, d_lab->d_absorpINLabel, 
		       matlIndex, patch);
        absorpIN_new.copyData(absorpIN);
      }


      old_dw->get(sootFVIN, d_lab->d_sootFVINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

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
      old_dw->get(abskg, d_lab->d_abskgINLabel,
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
      new_dw->allocateAndPut(abskg_new,
			     d_lab->d_abskgINLabel, matlIndex, patch);
      abskg_new.copyData(abskg);
      }
    }

    if (d_co_output) {
      old_dw->get(coIN, d_lab->d_coINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(coIN_new, d_lab->d_coINLabel,
			     matlIndex, patch);
      coIN_new.copyData(coIN);
    }
    
    if (d_sulfur_chem) {
      old_dw->get(h2sIN, d_lab->d_h2sINLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(h2sIN_new, d_lab->d_h2sINLabel,
			     matlIndex, patch);
      h2sIN_new.copyData(h2sIN);      

      old_dw->get(so2IN, d_lab->d_so2INLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(so2IN_new, d_lab->d_so2INLabel,
			     matlIndex, patch);
      so2IN_new.copyData(so2IN);

      old_dw->get(so3IN, d_lab->d_so3INLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(so3IN_new, d_lab->d_so3INLabel,
			     matlIndex, patch);
      so3IN_new.copyData(so3IN);
    }
    
    if (d_soot_precursors) {
      old_dw->get(c2h2IN, d_lab->d_c2h2INLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(c2h2IN_new, d_lab->d_c2h2INLabel,
			     matlIndex, patch);
      c2h2IN_new.copyData(c2h2IN);      

      old_dw->get(ch4IN, d_lab->d_ch4INLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(ch4IN_new, d_lab->d_ch4INLabel,
			     matlIndex, patch);
      ch4IN_new.copyData(ch4IN);
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
	  bool fixTemp = d_bc->getIfFixTemp();

	  double local_den = denMicro[currCell]*voidFraction[currCell];

	  if (cellType[currCell] != d_bc->getMMWallId()) {
	    density[currCell] = local_den;
	  }
	  else{
	    density[currCell] = 0.0;
	    if (d_bc->getIfCalcEnergyExchange())
	      if (d_DORadiationCalc)

		if (fixTemp) 
		  tempIN_new[currCell] = 298.0;
		else
		  tempIN_new[currCell] = solidTemp[currCell];
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
  if (d_calcReactingScalar)
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_calcEnthalpy)
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_scalarFELabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_calcReactingScalar)
    tsk->requires(Task::NewDW, d_lab->d_reactscalarFELabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_calcEnthalpy)
    tsk->requires(Task::NewDW, d_lab->d_enthalpyFELabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->modifies(d_lab->d_scalarSPLabel);
  if (d_calcReactingScalar)
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  if (d_calcEnthalpy)
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
    constCCVariable<double> old_scalar;
    constCCVariable<double> fe_scalar;
    CCVariable<double> new_scalar;
    constCCVariable<double> old_reactScalar;
    constCCVariable<double> fe_reactScalar;
    CCVariable<double> new_reactScalar;
    constCCVariable<double> old_enthalpy;
    constCCVariable<double> fe_enthalpy;
    CCVariable<double> new_enthalpy;
    CCVariable<double> density_guess;

    old_dw->get(old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->get(old_scalar, d_lab->d_scalarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_calcReactingScalar) {
      old_dw->get(old_reactScalar, d_lab->d_reactscalarSPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (d_calcEnthalpy)
      old_dw->get(old_enthalpy, d_lab->d_enthalpySPLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(rho1_density, d_lab->d_densityTempLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->getModifiable(new_scalar, d_lab->d_scalarSPLabel, 
		            matlIndex, patch);
    new_dw->get(fe_scalar, d_lab->d_scalarFELabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_calcReactingScalar)
      new_dw->getModifiable(new_reactScalar, d_lab->d_reactscalarSPLabel,
		    	    matlIndex, patch);
    if (d_calcReactingScalar)
      new_dw->get(fe_reactScalar, d_lab->d_reactscalarFELabel,
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    if (d_calcEnthalpy) {
      new_dw->getModifiable(new_enthalpy, d_lab->d_enthalpySPLabel, 
			    matlIndex, patch);
      new_dw->get(fe_enthalpy, d_lab->d_enthalpyFELabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    new_dw->getModifiable(density_guess, d_lab->d_densityGuessLabel, 
			  matlIndex, patch);

    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;
    double epsilon = 1.0e-15;

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
          
//The following if statement is to eliminate Kumar's zero block density problem
	  if (new_density[currCell] > 0.0) {
          double predicted_density;
	  if (old_density[currCell] > 0.0)
//            predicted_density = rho1_density[currCell];
	    if (d_inverse_density_average)
              predicted_density = 1.0/((factor_old/old_density[currCell] +
			         factor_new/new_density[currCell])/factor_divide);
	    else
              predicted_density = (factor_old*old_density[currCell] +
			         factor_new*new_density[currCell])/factor_divide;
	  else
	    predicted_density = new_density[currCell];

	  bool average_failed = false;
	    if (d_inverse_density_average)
	      (new_scalar)[currCell] = (factor_old*(old_scalar)[currCell] +
		  factor_new*(new_scalar)[currCell])/factor_divide;
	    else
	      (new_scalar)[currCell] = (factor_old*old_density[currCell]*
		  (old_scalar)[currCell] + factor_new*new_density[currCell]*
		  (new_scalar)[currCell])/(factor_divide*predicted_density);
// Following lines to fix density delay problem for helium.
// One would also need to edit fortran/explicit.F to use it.
//            (new_scalar)[currCell] = (new_scalar)[currCell]*predicted_density;
//            (new_scalar)[currCell] = (new_scalar)[currCell]*0.133/(
//              0.133*1.184344+(new_scalar)[currCell]*(0.133-1.184344));
            if ((new_scalar)[currCell] > 1.0) {
              if ((new_scalar)[currCell] < 1.0 + epsilon)
		(new_scalar)[currCell] = 1.0;
              else {
	        cout << "average failed with scalar > 1 at " << currCell << " , average value was " << (new_scalar)[currCell] << endl;
	        (new_scalar)[currCell] = (fe_scalar)[currCell];
	        average_failed = true;
              }
	    }
            else if ((new_scalar)[currCell] < 0.0) {
              if ((new_scalar)[currCell] > - epsilon)
            	(new_scalar)[currCell] = 0.0;
              else {
	        cout << "average failed with scalar < 0 at " << currCell << " , average value was " << (new_scalar)[currCell] << endl;
	        (new_scalar)[currCell] = (fe_scalar)[currCell];
	        average_failed = true;
              }
            }

	  if (d_calcReactingScalar) {
	      if (!average_failed) {
	      (new_reactScalar)[currCell] = (factor_old *
		old_density[currCell]*(old_reactScalar)[currCell] +
		factor_new*new_density[currCell]*
		(new_reactScalar)[currCell])/
		(factor_divide*predicted_density);
            if ((new_reactScalar)[currCell] > 1.0)
		(new_reactScalar)[currCell] = 1.0;
            else if ((new_reactScalar)[currCell] < 0.0)
            	(new_reactScalar)[currCell] = 0.0;
	      }
	      else
            	(new_reactScalar)[currCell] = (fe_reactScalar)[currCell];
	  }

          if (d_calcEnthalpy)
	    if (!average_failed)
	    new_enthalpy[currCell] = (factor_old*old_density[currCell]*
		old_enthalpy[currCell] + factor_new*new_density[currCell]*
		new_enthalpy[currCell])/(factor_divide*predicted_density);
	    else
	    new_enthalpy[currCell] = fe_enthalpy[currCell];

	  //density_guess[currCell] = predicted_density;

	}
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
			   const TimeIntegratorLabel* )
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
  delta_t *= timelabels->time_multiplier *
	     timelabels->time_position_multiplier_after_average;
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
    // compute drhodt and its filtered value
    drhodt.allocate(patch->getLowIndex(), patch->getHighIndex());
    drhodt.initialize(0.0);

    if ((d_first_order_drhodt)||(current_step <= drhodt_1st_order)) {
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

	    double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
	    drhodt[currcell] = (new_factor*new_density[currcell] -
				old_factor*old_density[currcell] +
				old_old_density[currcell])*vol /
			       (old_delta_t*factor);
          }
        }
      }
    }

    if ((d_filter_drhodt)&&(!(d_3d_periodic))) {
    // filtering for periodic case is not implemented 
    // if it needs to be then drhodt will require 1 layer of boundary cells to be computed
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

    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
          IntVector currcell(ii,jj,kk);

	  double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
	  if (Abs(filterdrhodt[currcell]/vol) < 1.0e-9)
	      filterdrhodt[currcell] = 0.0;
        }
      }
    }
  }
}
