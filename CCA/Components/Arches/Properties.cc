/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- Properties.cc --------------------------------------------------
#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#if HAVE_TABPROPS
# include <CCA/Components/Arches/ChemMix/TabPropsInterface.h>
#endif 
# include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
# include <CCA/Components/Arches/ChemMix/ColdFlow.h>
#include <CCA/Components/Arches/Mixing/MixingModel.h>
#include <CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <CCA/Components/Arches/Mixing/NewStaticMixingTable.h>
#include <CCA/Components/Arches/Mixing/StandardTable.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Thread/Time.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties(const ArchesLabel* label, 
                       const MPMArchesLabel* MAlb,
                       PhysicalConstants* phys_const, 
                       bool calcReactingScalar,
                       bool calcEnthalpy, 
                       bool calcVariance, 
                       const ProcessorGroup* myworld):
                       d_lab(label), d_MAlab(MAlb), 
                       d_physicalConsts(phys_const), 
                       d_calcReactingScalar(calcReactingScalar),
                       d_calcEnthalpy(calcEnthalpy),
                       d_calcVariance(calcVariance),
                       d_myworld(myworld)
{
  d_DORadiationCalc = false;
  d_radiationCalc = false;
  d_co_output       = false;
  d_sulfur_chem     = false;
  d_soot_precursors = false;
  d_tabulated_soot  = false;
  d_newEnthalpySolver = false; 
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

  if ( mixModel == "TabProps" || mixModel == "ClassicTable" ){ 
    delete d_mixingRxnTable; 
  }
}

//****************************************************************************
// Problem Setup for Properties
//****************************************************************************
void 
Properties::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Properties");
  db->getWithDefault("filter_drhodt",          d_filter_drhodt,          false);
  db->getWithDefault("first_order_drhodt",     d_first_order_drhodt,     true);
  db->getWithDefault("inverse_density_average",d_inverse_density_average,false);
  d_denRef = d_physicalConsts->getRefPoint();
  d_reactingFlow = true;
  // check to see if gas is adiabatic and (if DQMOM) particles are not:
  d_adiabGas_nonadiabPart = false; 
  if (params->findBlock("DQMOM")) {
    ProblemSpecP db_dqmom = params->findBlock("DQMOM");
    db_dqmom->getWithDefault("adiabGas_nonadiabPart", d_adiabGas_nonadiabPart, false); 
  }

  // read type of mixing model
  if (db->findBlock("NewStaticMixingTable"))
    mixModel = "NewStaticMixingTable"; 
  else if (db->findBlock("ColdFlowMixingModel"))
    mixModel = "ColdFlowMixingModel"; 
  else if (db->findBlock("StandardTable"))
    mixModel = "StandardTable"; 
#if HAVE_TABPROPS
  else if (db->findBlock("TabProps"))
    mixModel = "TabProps";
#endif 
  else if (db->findBlock("ClassicTable"))
    mixModel = "ClassicTable";
	else if (db->findBlock("ColdFlow"))
		mixModel = "ColdFlow";
  else
    throw InvalidValue("ERROR!: No mixing/reaction table specified! If you are attempting to use the new TabProps interface, ensure that you configured properly with TabProps and Boost libs.",__FILE__,__LINE__);

  if (mixModel == "ColdFlowMixingModel") {
    d_mixingModel = scinew ColdflowMixingModel(d_calcReactingScalar,
                                               d_calcEnthalpy,
                                               d_calcVariance);
    d_reactingFlow = false;
  }
  else if (mixModel == "NewStaticMixingTable"){
    d_mixingModel = scinew NewStaticMixingTable(d_calcReactingScalar,
                                                d_calcEnthalpy,
                                                d_calcVariance,
                                                d_myworld);
    d_mixingModel->setNonAdiabPartBool(d_adiabGas_nonadiabPart); 
  }
  else if (mixModel == "StandardTable"){
    d_mixingModel = scinew StandardTable(d_calcReactingScalar,
                                         d_calcEnthalpy,
                                         d_calcVariance);
  }
#if HAVE_TABPROPS
  else if (mixModel == "TabProps") {
    // New TabPropsInterface stuff...
    d_mixingRxnTable = scinew TabPropsInterface( d_lab, d_MAlab );
    d_mixingRxnTable->problemSetup( db ); 
    d_reactingFlow    = d_mixingRxnTable->is_not_cold(); 
    // At this time, these all need to be false:
    d_co_output       = false;
    if (d_sulfur_chem) {
      proc0cout << "Warning!: The old sulfur_chem boolean is not compatible with TabProps.  I am going to set it to false. " << endl;
      d_sulfur_chem     = false;
    }
    if (d_soot_precursors) {
      proc0cout << "Warning!: The soot_precursors boolean is not compatible with TabProps.  I am going to set it to false. " << endl; 
      d_soot_precursors = false;
    }
    if (d_tabulated_soot) {
      proc0cout << "Warning!: The tabulated soot mechanism (tabulated_soot) is not active yet when using TabProps.  I am going to set it to false. " << endl;
      d_tabulated_soot  = false;
    }
  }
#endif 
  else if (mixModel == "ClassicTable") { 
    // New Classic interface
    d_mixingRxnTable = scinew ClassicTableInterface( d_lab, d_MAlab ); 
    d_mixingRxnTable->problemSetup( db ); 
    // At this time, these all need to be false:
    d_co_output       = false;
    d_reactingFlow    = d_mixingRxnTable->is_not_cold(); 

    if (d_sulfur_chem) {
      proc0cout << "Warning!: The old sulfur_chem boolean is not compatible with ClassicTable.  I am going to set it to false. " << endl;
      d_sulfur_chem     = false;
    }
    if (d_soot_precursors) {
      proc0cout << "Warning!: The soot_precursors boolean is not compatible with ClassicTable.  I am going to set it to false. " << endl; 
      d_soot_precursors = false;
    }
    if (d_tabulated_soot) {
      proc0cout << "Warning!: The tabulated soot mechanism (tabulated_soot) is not active yet when using ClassicTable.  I am going to set it to false. " << endl;
      d_tabulated_soot  = false;
    }
	} else if (mixModel == "ColdFlow") {
		d_mixingRxnTable = scinew ColdFlow( d_lab, d_MAlab ); 
		d_mixingRxnTable->problemSetup( db ); 
    d_reactingFlow = false;
  }
  else if (mixModel == "pdfMixingModel" || mixModel == "SteadyFlameletsTable"
        || mixModel == "flameletModel"  || mixModel == "StaticMixingTable"
        || mixModel == "meanMixingModel" ){
    throw InvalidValue("DEPRECATED: Mixing Model no longer supported: " + mixModel, __FILE__, __LINE__);
  }else{
    throw InvalidValue("Mixing Model not supported: " + mixModel, __FILE__, __LINE__);
  }
 
  if (mixModel != "TabProps" && mixModel != "ClassicTable" && mixModel != "ColdFlow") {
    d_mixingModel->problemSetup(db);

    if (d_calcEnthalpy){
      d_H_air = d_mixingModel->getAdiabaticAirEnthalpy();
    }
  
    if (d_reactingFlow) {
      d_f_stoich    = d_mixingModel->getFStoich();
      d_carbon_fuel = d_mixingModel->getCarbonFuel();
      d_carbon_air  = d_mixingModel->getCarbonAir();
    }

    d_co_output       = d_mixingModel->getCOOutput();
    d_sulfur_chem     = d_mixingModel->getSulfurChem();
    d_soot_precursors = d_mixingModel->getSootPrecursors();
    d_tabulated_soot  = d_mixingModel->getTabulatedSoot();  
    d_radiationCalc = false;
  } else { 

    if ( d_calcEnthalpy ) { 

      d_H_air = d_mixingRxnTable->get_ox_enthalpy(); 


    } 
  } 


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

    if ( db_enthalpy_solver->findBlock( "EnthalpySolver" ) ){ 
      db_enthalpy_solver = db_enthalpy_solver->findBlock("EnthalpySolver");

      if (db_enthalpy_solver->findBlock("DORadiationModel"))
        d_radiationCalc = true; 
      else 
        proc0cout << "ATTENTION: NO WORKING RADIATION MODEL TURNED ON!" << endl; 

      if (d_radiationCalc) {

        if (db_enthalpy_solver->findBlock("DORadiationModel"))
          d_DORadiationCalc = true; 

        d_opl = 0.0;

        if (!d_DORadiationCalc)
          db->require("optically_thin_model_opl",d_opl);
        if (d_tabulated_soot) {
          db->getWithDefault("empirical_soot",d_empirical_soot,false);
          if (d_empirical_soot)
            throw InvalidValue("Table has soot, do not use empirical soot model!",
                               __FILE__, __LINE__);
        }
        else {
          db->getWithDefault("empirical_soot",d_empirical_soot,true);
          if (d_empirical_soot) 
            db->getWithDefault("soot_factor", d_sootFactor, 1.0);

        }
      }
    }
  } else { 

    // allowance for other enthalpy solver
    ProblemSpecP params_non_constant = params;
    const ProblemSpecP params_root = params_non_constant->getRootNode();
    if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("newEnthalpySolver") ){ 

      d_newEnthalpySolver = true; 

    } 
  } 
}

//****************************************************************************
// compute density for inlet streams: only for cold streams
//****************************************************************************

void
Properties::computeInletProperties(const InletStream& inStream, 
                                   Stream& outStream, const string bc_type)
{
  if (dynamic_cast<const ColdflowMixingModel*>(d_mixingModel)){
    d_mixingModel->computeProps(inStream, outStream);
  } else if (dynamic_cast<const NewStaticMixingTable*>(d_mixingModel)) {
    d_mixingModel->computeProps(inStream, outStream);
  }
  else if (dynamic_cast<const StandardTable*>(d_mixingModel)) {
    d_mixingModel->computeProps(inStream, outStream);
  }
#if HAVE_TABPROPS
  else if ( mixModel == "TabProps"){
    d_mixingRxnTable->oldTableHack( inStream, outStream, d_calcEnthalpy, bc_type ); 
  }
#endif 
  else if ( mixModel == "ClassicTable"){
    d_mixingRxnTable->oldTableHack( inStream, outStream, d_calcEnthalpy, bc_type ); 
  } else if ( mixModel == "ColdFlow" ) {
		// nothing to do here -- put here as a place holder until Properties is deleted
  }
  else {
    throw InvalidValue("Mixing Model not supported", __FILE__, __LINE__);
  }
}

//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(SchedulerP& sched, 
                                 const PatchSet* patches,
                                 const MaterialSet* matls,
                                 const TimeIntegratorLabel* timelabels,
                                 bool modify_ref_density, 
                                 bool initialize)
{
  string md = "";
  if (!(modify_ref_density)) md += "RKSSP";
  string taskname =  "Properties::ReComputeProps" +
                     timelabels->integrator_step_name + md;
                     
  Task* tsk = scinew Task(taskname, this,
                          &Properties::reComputeProps,
                          timelabels, modify_ref_density, initialize);

  Ghost::GhostType  gn = Ghost::None;
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->modifies(d_lab->d_scalarSPLabel);

  if (d_calcVariance){
    tsk->requires(Task::NewDW, d_lab->d_normalizedScalarVarLabel, gn, 0);
  }
  
  //__________________________________
  if (d_calcReactingScalar)
  {
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel,  gn, 0);
  }  
  if (d_calcEnthalpy){
    tsk->modifies(d_lab->d_enthalpySPLabel);
  }
  
  //__________________________________
  if (d_MAlab && initialize) {
#ifdef ExactMPMArchesInitialize
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,    gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,  gn, 0);
#else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gn, 0);
#endif
  }
  else{
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gn, 0);
  }
  
  
  if (d_MAlab && !initialize) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,  gn, 0);
    
    if (d_DORadiationCalc && d_bc->getIfCalcEnergyExchange()){
      tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel,  gn, 0);
    }
  }

  tsk->modifies(d_lab->d_densityCPLabel);
  
// assuming ref_density is not changed by RK averaging
  if (modify_ref_density){
    tsk->computes(timelabels->ref_density);
  }
  
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_drhodfCPLabel);
    
    tsk->computes(d_lab->d_dummyTLabel);
    if (d_reactingFlow) {
      tsk->computes(d_lab->d_tempINLabel);
      tsk->computes(d_lab->d_cpINLabel);
      tsk->computes(d_lab->d_co2INLabel);
      tsk->computes(d_lab->d_h2oINLabel);
      tsk->computes(d_lab->d_heatLossLabel);
      tsk->computes(d_lab->d_enthalpyRXNLabel);
      tsk->computes(d_lab->d_mixMWLabel); 
      if (d_calcReactingScalar)
        tsk->computes(d_lab->d_reactscalarSRCINLabel);
    }

    if (d_co_output){
      tsk->computes(d_lab->d_coINLabel);
    }
    
    if (d_sulfur_chem) {
      tsk->computes(d_lab->d_h2sINLabel);
      tsk->computes(d_lab->d_so2INLabel);
      tsk->computes(d_lab->d_so3INLabel);
      tsk->computes(d_lab->d_sulfurINLabel);

      tsk->computes(d_lab->d_s2INLabel);
      tsk->computes(d_lab->d_shINLabel);
      tsk->computes(d_lab->d_soINLabel);
      tsk->computes(d_lab->d_hso2INLabel);

      tsk->computes(d_lab->d_hosoINLabel);
      tsk->computes(d_lab->d_hoso2INLabel);
      tsk->computes(d_lab->d_snINLabel);
      tsk->computes(d_lab->d_csINLabel);

      tsk->computes(d_lab->d_ocsINLabel);
      tsk->computes(d_lab->d_hsoINLabel);
      tsk->computes(d_lab->d_hosINLabel);
      tsk->computes(d_lab->d_hsohINLabel);

      tsk->computes(d_lab->d_h2soINLabel);
      tsk->computes(d_lab->d_hoshoINLabel);
      tsk->computes(d_lab->d_hs2INLabel);
      tsk->computes(d_lab->d_h2s2INLabel);
    }
    
    if (d_soot_precursors) {
      tsk->computes(d_lab->d_c2h2INLabel);
      tsk->computes(d_lab->d_ch4INLabel);
    }

    if (d_radiationCalc) {
      if (!d_DORadiationCalc){
        tsk->computes(d_lab->d_absorpINLabel);
      }
      tsk->computes(d_lab->d_sootFVINLabel);
    }

  }
  else {
    tsk->modifies(d_lab->d_drhodfCPLabel);

    tsk->modifies(d_lab->d_dummyTLabel);
    if (d_reactingFlow) {
      tsk->modifies(d_lab->d_tempINLabel);
      tsk->modifies(d_lab->d_cpINLabel);
      tsk->modifies(d_lab->d_co2INLabel);
      tsk->modifies(d_lab->d_h2oINLabel);
      tsk->modifies(d_lab->d_heatLossLabel);
      tsk->modifies(d_lab->d_enthalpyRXNLabel);
      tsk->modifies(d_lab->d_mixMWLabel); 
      if (d_calcReactingScalar)
        tsk->modifies(d_lab->d_reactscalarSRCINLabel);
    }

    if (d_co_output){
      tsk->modifies(d_lab->d_coINLabel);
    }
    
    if (d_sulfur_chem) {
      tsk->modifies(d_lab->d_h2sINLabel);
      tsk->modifies(d_lab->d_so2INLabel);
      tsk->modifies(d_lab->d_so3INLabel);
      tsk->modifies(d_lab->d_sulfurINLabel);

      tsk->modifies(d_lab->d_s2INLabel);
      tsk->modifies(d_lab->d_shINLabel);
      tsk->modifies(d_lab->d_soINLabel);
      tsk->modifies(d_lab->d_hso2INLabel);

      tsk->modifies(d_lab->d_hosoINLabel);
      tsk->modifies(d_lab->d_hoso2INLabel);
      tsk->modifies(d_lab->d_snINLabel);
      tsk->modifies(d_lab->d_csINLabel);

      tsk->modifies(d_lab->d_ocsINLabel);
      tsk->modifies(d_lab->d_hsoINLabel);
      tsk->modifies(d_lab->d_hosINLabel);
      tsk->modifies(d_lab->d_hsohINLabel);

      tsk->modifies(d_lab->d_h2soINLabel);
      tsk->modifies(d_lab->d_hoshoINLabel);
      tsk->modifies(d_lab->d_hs2INLabel);
      tsk->modifies(d_lab->d_h2s2INLabel);
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

    //tsk->modifies(d_lab->d_tabReactionRateLabel);
  }

  if (d_MAlab) {
    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) 
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
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<int> cellType;
    CCVariable<double> scalar;
    CCVariable<double> extrascalar;
    constCCVariable<double> normalizedScalarVar;
    constCCVariable<double> reactScalar;
    constCCVariable<double> scalarDisp;
    constCCVariable<double> voidFraction;
    CCVariable<double> new_density;
    CCVariable<double> dummytemperature;
    CCVariable<double> temperature;
    CCVariable<double> mixMW; 
    CCVariable<double> cp;
    CCVariable<double> co2;
    CCVariable<double> h2o;
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
    CCVariable<double> sulfur;

    CCVariable<double> s2;
    CCVariable<double> sh;
    CCVariable<double> so;
    CCVariable<double> hso2;

    CCVariable<double> hoso;
    CCVariable<double> hoso2;
    CCVariable<double> sn;
    CCVariable<double> cs;

    CCVariable<double> ocs;
    CCVariable<double> hso;
    CCVariable<double> hos;
    CCVariable<double> hsoh;

    CCVariable<double> h2so;
    CCVariable<double> hosho;
    CCVariable<double> hs2;
    CCVariable<double> h2s2;

    CCVariable<double> c2h2;
    CCVariable<double> ch4;

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    Ghost::GhostType  gn = Ghost::None;
    
    if (d_MAlab && initialize) {
#ifdef ExactMPMArchesInitialize
      new_dw->get(cellType,     d_lab->d_mmcellTypeLabel,   indx, patch, gn, 0);
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
#else
      new_dw->get(cellType, d_lab->d_cellTypeLabel,   indx, patch, gn, 0);
#endif
    }else{
      new_dw->get(cellType, d_lab->d_cellTypeLabel,   indx, patch, gn, 0);
    }

    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, indx, patch);

    if (d_calcVariance) {
      new_dw->get(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, indx, patch, gn, 0);
    }
    
    //__________________________________
    if (d_calcReactingScalar) {
      new_dw->get(reactScalar, d_lab->d_reactscalarSPLabel,  indx, patch, gn, 0);
    }
    
    //__________________________________
    if (d_calcEnthalpy){
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel,  indx, patch);
    }
    //__________________________________
    new_dw->getModifiable(new_density,   d_lab->d_densityCPLabel,  indx, patch);
    new_density.initialize(0.0);
    
    //__________________________________
    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
      new_dw->allocateAndPut(drhodf, d_lab->d_drhodfCPLabel, indx, patch);

      new_dw->allocateAndPut(dummytemperature, d_lab->d_dummyTLabel, indx, patch);
      if (d_reactingFlow) {
        new_dw->allocateAndPut(temperature, d_lab->d_tempINLabel,     indx, patch);
        new_dw->allocateAndPut(cp,          d_lab->d_cpINLabel,       indx, patch);
        new_dw->allocateAndPut(co2,         d_lab->d_co2INLabel,      indx, patch);
        new_dw->allocateAndPut(h2o,         d_lab->d_h2oINLabel,      indx, patch);
        new_dw->allocateAndPut(heatLoss,    d_lab->d_heatLossLabel,   indx, patch);
        new_dw->allocateAndPut(enthalpyRXN, d_lab->d_enthalpyRXNLabel,indx, patch);
        new_dw->allocateAndPut(mixMW,       d_lab->d_mixMWLabel,      indx, patch); 
        if (d_calcReactingScalar){
          new_dw->allocateAndPut(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
                                                                      indx, patch);
        }
      }

      if (d_co_output){
        new_dw->allocateAndPut(co, d_lab->d_coINLabel,indx, patch);
      }
      
      //__________________________________
      if (d_sulfur_chem) {
        new_dw->allocateAndPut(h2s,    d_lab->d_h2sINLabel,    indx, patch);
        new_dw->allocateAndPut(so2,    d_lab->d_so2INLabel,    indx, patch);
        new_dw->allocateAndPut(so3,    d_lab->d_so3INLabel,    indx, patch);
        new_dw->allocateAndPut(sulfur, d_lab->d_sulfurINLabel, indx, patch);

        new_dw->allocateAndPut(s2,     d_lab->d_s2INLabel,     indx, patch);
        new_dw->allocateAndPut(sh,     d_lab->d_shINLabel,     indx, patch);
        new_dw->allocateAndPut(so,     d_lab->d_soINLabel,     indx, patch);
        new_dw->allocateAndPut(hso2,   d_lab->d_hso2INLabel,   indx, patch);

        new_dw->allocateAndPut(hoso,   d_lab->d_hosoINLabel,   indx, patch);
        new_dw->allocateAndPut(hoso2,  d_lab->d_hoso2INLabel,  indx, patch);
        new_dw->allocateAndPut(sn,     d_lab->d_snINLabel,     indx, patch);
        new_dw->allocateAndPut(cs,     d_lab->d_csINLabel,     indx, patch);

        new_dw->allocateAndPut(ocs,    d_lab->d_ocsINLabel,    indx, patch);
        new_dw->allocateAndPut(hso,    d_lab->d_hsoINLabel,    indx, patch);
        new_dw->allocateAndPut(hos,    d_lab->d_hosINLabel,    indx, patch);
        new_dw->allocateAndPut(hsoh,   d_lab->d_hsohINLabel,   indx, patch);

        new_dw->allocateAndPut(h2so,   d_lab->d_h2soINLabel,   indx, patch);
        new_dw->allocateAndPut(hosho,  d_lab->d_hoshoINLabel,  indx, patch);
        new_dw->allocateAndPut(hs2,    d_lab->d_hs2INLabel,    indx, patch);
        new_dw->allocateAndPut(h2s2,   d_lab->d_h2s2INLabel,   indx, patch);
      }
      if (d_soot_precursors) {
        new_dw->allocateAndPut(c2h2,   d_lab->d_c2h2INLabel,   indx, patch);
        new_dw->allocateAndPut(ch4,    d_lab->d_ch4INLabel,    indx, patch);
      }

      if (d_radiationCalc) {
        if (!d_DORadiationCalc)
          new_dw->allocateAndPut(absorption, d_lab->d_absorpINLabel,
                                                               indx, patch);
        new_dw->allocateAndPut(sootFV, d_lab->d_sootFVINLabel, indx,patch);
      }

    }
    else {
      new_dw->getModifiable(drhodf, d_lab->d_drhodfCPLabel, indx, patch);

      new_dw->getModifiable(dummytemperature, d_lab->d_dummyTLabel, indx, patch);
      if (d_reactingFlow) {
        new_dw->getModifiable(temperature, d_lab->d_tempINLabel,      indx, patch);
        new_dw->getModifiable(cp,          d_lab->d_cpINLabel,        indx, patch);
        new_dw->getModifiable(co2,         d_lab->d_co2INLabel,       indx, patch);
        new_dw->getModifiable(h2o,         d_lab->d_h2oINLabel,       indx, patch);
        new_dw->getModifiable(heatLoss,    d_lab->d_heatLossLabel,    indx, patch);
        new_dw->getModifiable(enthalpyRXN, d_lab->d_enthalpyRXNLabel, indx, patch);
        new_dw->getModifiable(mixMW,       d_lab->d_mixMWLabel,       indx, patch); 
        if (d_calcReactingScalar)
          new_dw->getModifiable(reactscalarSRC, d_lab->d_reactscalarSRCINLabel,
                                                                      indx, patch);
      }

      if (d_co_output){
        new_dw->getModifiable(co, d_lab->d_coINLabel,   indx, patch);
      }
      //__________________________________
      if (d_sulfur_chem) {
        new_dw->getModifiable(h2s,    d_lab->d_h2sINLabel,   indx, patch);
        new_dw->getModifiable(so2,    d_lab->d_so2INLabel,   indx, patch);
        new_dw->getModifiable(so3,    d_lab->d_so3INLabel,   indx, patch);
        new_dw->getModifiable(sulfur, d_lab->d_sulfurINLabel,indx, patch);

        new_dw->getModifiable(s2,     d_lab->d_s2INLabel,    indx, patch);
        new_dw->getModifiable(sh,     d_lab->d_shINLabel,    indx, patch);
        new_dw->getModifiable(so,     d_lab->d_soINLabel,    indx, patch);
        new_dw->getModifiable(hso2,   d_lab->d_hso2INLabel,  indx, patch);

        new_dw->getModifiable(hoso,   d_lab->d_hosoINLabel,  indx, patch);
        new_dw->getModifiable(hoso2,  d_lab->d_hoso2INLabel, indx, patch);
        new_dw->getModifiable(sn,     d_lab->d_snINLabel,    indx, patch);
        new_dw->getModifiable(cs,     d_lab->d_csINLabel,    indx, patch);

        new_dw->getModifiable(ocs,    d_lab->d_ocsINLabel,   indx, patch);
        new_dw->getModifiable(hso,    d_lab->d_hsoINLabel,   indx, patch);
        new_dw->getModifiable(hos,    d_lab->d_hosINLabel,   indx, patch);
        new_dw->getModifiable(hsoh,   d_lab->d_hsohINLabel,  indx, patch);

        new_dw->getModifiable(h2so,   d_lab->d_h2soINLabel,  indx, patch);
        new_dw->getModifiable(hosho,  d_lab->d_hoshoINLabel, indx, patch);
        new_dw->getModifiable(hs2,    d_lab->d_hs2INLabel,   indx, patch);
        new_dw->getModifiable(h2s2,   d_lab->d_h2s2INLabel,  indx, patch);

      }
      //__________________________________
      if (d_soot_precursors) {
        new_dw->getModifiable(c2h2, d_lab->d_c2h2INLabel, indx, patch);
        new_dw->getModifiable(ch4, d_lab->d_ch4INLabel,   indx, patch);
      }
      //__________________________________
      if (d_radiationCalc) {
        if (!d_DORadiationCalc)
          new_dw->getModifiable(absorption, d_lab->d_absorpINLabel, indx, patch);
        new_dw->getModifiable(sootFV,       d_lab->d_sootFVINLabel, indx,patch);
      }

    }
    drhodf.initialize(0.0);
    
    dummytemperature.initialize(0.0);
    if (d_reactingFlow) {
      temperature.initialize(0.0); 
      cp.initialize(0.0);
      co2.initialize(0.0);
      h2o.initialize(0.0);
      heatLoss.initialize(0.0);
      enthalpyRXN.initialize(0.0);
      mixMW.initialize(0.0); 
      if (d_calcReactingScalar){
        reactscalarSRC.initialize(0.0);
      }
    }    

    if (d_co_output){
      co.initialize(0.0);
    }
    
    if (d_sulfur_chem) {
      h2s.initialize(0.0);
      so2.initialize(0.0);
      so3.initialize(0.0);
      sulfur.initialize(0.0);

      s2.initialize(0.0);
      sh.initialize(0.0);
      so.initialize(0.0);
      hso2.initialize(0.0);

      hoso.initialize(0.0);
      hoso2.initialize(0.0);
      sn.initialize(0.0);
      cs.initialize(0.0);

      ocs.initialize(0.0);
      hso.initialize(0.0);
      hos.initialize(0.0);
      hsoh.initialize(0.0);

      h2so.initialize(0.0);
      hosho.initialize(0.0);
      hs2.initialize(0.0);
      h2s2.initialize(0.0);
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
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel,  indx, patch, gn, 0);
      
      if (d_DORadiationCalc && d_bc->getIfCalcEnergyExchange())
        new_dw->get(solidTemp, d_MAlab->integTemp_CCLabel,   indx, patch, gn, 0);
    }

    if (d_MAlab) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        new_dw->allocateAndPut(denMicro, d_lab->d_densityMicroLabel, indx, patch);
      }else{
        new_dw->getModifiable(denMicro, d_lab->d_densityMicroLabel, indx, patch);
      }
    }

    IntVector indexLow = patch->getExtraCellLowIndex();
    IntVector indexHigh = patch->getExtraCellHighIndex();

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
          
          //Mixture fraction 
          inStream.d_mixVars[0] = scalar[currCell];
          
          if (d_calcVariance) {
            // Variance passed in has already been normalized !!!
            inStream.d_mixVarVariance[0] = normalizedScalarVar[currCell];
          }

          // currently not using any reaction progress variables
          if (d_calcReactingScalar) {
            inStream.d_rxnVars[0] = reactScalar[currCell];
          }

          if (d_calcEnthalpy)
              //              &&(cellType[currCell] != d_bc->getIntrusionID()))
            if (initialize || local_enthalpy_init)
              inStream.d_enthalpy = 0.0;
            else
              inStream.d_enthalpy = enthalpy[currCell];
          else
            inStream.d_enthalpy = 0.0;
           // This flag ensures properties for heatloss=0.0
           // during the initialization
           inStream.d_initEnthalpy = (initialize || local_enthalpy_init);

           inStream.cellvolume =  cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
  TAU_PROFILE_START(mixing);
           d_mixingModel->computeProps(inStream, outStream);
  TAU_PROFILE_STOP(mixing);

          double local_den = outStream.getDensity();
          drhodf[currCell] = outStream.getdrhodf();

          if (d_co_output) {
            co[currCell] = outStream.getCO();
          }
          //__________________________________
          if (d_sulfur_chem) {
            h2s[currCell]    = outStream.getH2S();
            so2[currCell]    = outStream.getSO2();
            so3[currCell]    = outStream.getSO3();
            sulfur[currCell] = outStream.getSULFUR();

            s2[currCell]     = outStream.getS2();
            sh[currCell]     = outStream.getSH();
            so[currCell]     = outStream.getSO();
            hso2[currCell]   = outStream.getHSO2();

            hoso[currCell]   = outStream.getHOSO();
            hoso2[currCell]  = outStream.getHOSO2();
            sn[currCell]     = outStream.getSN();
            cs[currCell]     = outStream.getCS();

            ocs[currCell]    = outStream.getOCS();
            hso[currCell]    = outStream.getHSO();
            hos[currCell]    = outStream.getHOS();
            hsoh[currCell]   = outStream.getHSOH();

            h2so[currCell]   = outStream.getH2SO();
            hosho[currCell]  = outStream.getHOSHO();
            hs2[currCell]    = outStream.getHS2();
            h2s2[currCell]   = outStream.getH2S2();
          }
          //__________________________________
          if (d_soot_precursors) {
            c2h2[currCell]   = outStream.getC2H2();
            ch4[currCell]    = outStream.getCH4();
          }
          
          //__________________________________s
          if (d_calcEnthalpy && (initialize || local_enthalpy_init)){
            enthalpy[currCell] = outStream.getEnthalpy();
          }
          
          //__________________________________
          dummytemperature[currCell] = outStream.getTemperature();
          if (d_reactingFlow) {
            temperature[currCell]   = outStream.getTemperature();
            cp[currCell]            = outStream.getCP();
            co2[currCell]           = outStream.getCO2();
            h2o[currCell]           = outStream.getH2O();
            heatLoss[currCell]      = outStream.getheatLoss();
            enthalpyRXN[currCell]   = outStream.getEnthalpy();
            mixMW[currCell]         = outStream.getMixMW(); 
// Uncomment the next line to check enthalpy transport in adiabatic case
            if (d_calcEnthalpy){
              enthalpyRXN[currCell] -= enthalpy[currCell];
            }
            if (d_calcReactingScalar) {
              reactscalarSRC[currCell] = outStream.getRxnSource();
              enthalpyRXN[currCell]    = scalar[currCell] - reactScalar[currCell];
            }
          }
          
          //__________________________________
          if (d_radiationCalc) {
            if ((d_calcReactingScalar)||(d_tabulated_soot)){ 
              sootFV[currCell] = outStream.getSootFV();
            }
            else {
              if (d_empirical_soot) {
                if (temperature[currCell] > 1000.0) {
                  double carbon_content = 
                           getCarbonContent(inStream.d_mixVars[0]);
                  double bc = carbon_content * local_den;
                  double c3      = 0.1;
                  double rhosoot = 1950.0;
                  double cmw     = 12.0;

                  if (inStream.d_mixVars[0] > d_f_stoich)
                    sootFV[currCell] = d_sootFactor * c3*bc*cmw/rhosoot;
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
          //__________________________________
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
    
    //__________________________________
    // Write the computed density to the new data warehouse
    if (modify_ref_density) {
      double den_ref = 0.0;
      if (patch->containsCell(d_denRef)) {
        den_ref = new_density[d_denRef];
        cerr << "density_ref " << den_ref << endl;
      }
      new_dw->put(sum_vartype(den_ref),timelabels->ref_density);
    }

    /*if ((d_bc->getIntrusionBC())&& d_reactingFlow )
      d_bc->intrusionTemperatureBC(pc, patch, cellType, temperature);*/

    if (d_MAlab && d_DORadiationCalc && !initialize) {
      bool d_energyEx = d_bc->getIfCalcEnergyExchange();
      d_bc->mmWallTemperatureBC(patch, cellType, solidTemp, temperature,
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
Properties::sched_computePropsFirst_mm(SchedulerP& sched, 
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  Task* tsk = scinew Task("Properties::mmComputePropsFirst",
                          this,
                          &Properties::computePropsFirst_mm);
                          
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW, d_lab->d_densityMicroINLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,       gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,   gn, 0);
  
  if (d_bc->getIfCalcEnergyExchange()) 
    if (d_DORadiationCalc)
      tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel, gn, 0);    

  //__________________________________
  if (d_reactingFlow) {
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_cpINLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_co2INLabel,    gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_heatLossLabel, gn, 0);
    //tsk->requires(Task::OldDW, d_lab->d_mixMWLabel,    gn, 0); 
    /*
    tsk->requires(Task::OldDW, d_lab->d_enthalpyRXNLabel, gn, 0);
    */
    if (d_calcReactingScalar)
      tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel, gn, 0);
  }

  //__________________________________
  if (d_radiationCalc) {
    if (!d_DORadiationCalc){
      tsk->requires(Task::OldDW, d_lab->d_absorpINLabel,        gn, 0);
    }
    tsk->requires(Task::OldDW, d_lab->d_sootFVINLabel,          gn, 0);
    if (d_DORadiationCalc) {
      tsk->requires(Task::OldDW, d_lab->d_h2oINLabel,           gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationSRCINLabel,  gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationFluxEINLabel,gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationFluxWINLabel,gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationFluxNINLabel,gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationFluxSINLabel,gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationFluxTINLabel,gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_radiationFluxBINLabel,gn, 0);
      tsk->requires(Task::OldDW, d_lab->d_abskgINLabel,         gn, 0);
    }
  }

  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_refDensity_label);
  tsk->computes(d_lab->d_densityMicroLabel);

  //__________________________________
  if (d_reactingFlow) {
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_cpINLabel);
    tsk->computes(d_lab->d_co2INLabel);
    tsk->computes(d_lab->d_heatLossLabel);
    tsk->computes(d_lab->d_enthalpyRXNLabel);
    tsk->computes(d_lab->d_mixMWLabel); 
    if (d_calcReactingScalar){
      tsk->computes(d_lab->d_reactscalarSRCINLabel);
    }
  }

  //__________________________________
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

  //__________________________________
  if (d_co_output) {
    tsk->requires(Task::OldDW, d_lab->d_coINLabel,     gn, 0);
    tsk->computes(d_lab->d_coINLabel);
  }

  //__________________________________
  if (d_sulfur_chem) {
    tsk->requires(Task::OldDW, d_lab->d_h2sINLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_so2INLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_so3INLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_sulfurINLabel,  gn, 0);

    tsk->requires(Task::OldDW, d_lab->d_s2INLabel,      gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_shINLabel,      gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_soINLabel,      gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hso2INLabel,    gn, 0);

    tsk->requires(Task::OldDW, d_lab->d_hosoINLabel,    gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hoso2INLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_snINLabel,      gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_csINLabel,      gn, 0);

    tsk->requires(Task::OldDW, d_lab->d_ocsINLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hsoINLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hosINLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hsohINLabel,    gn, 0);

    tsk->requires(Task::OldDW, d_lab->d_h2soINLabel,    gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hoshoINLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_hs2INLabel,     gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_h2s2INLabel,    gn, 0);

    tsk->computes(d_lab->d_h2sINLabel);
    tsk->computes(d_lab->d_so2INLabel);
    tsk->computes(d_lab->d_so3INLabel);
    tsk->computes(d_lab->d_sulfurINLabel);

    tsk->computes(d_lab->d_s2INLabel);
    tsk->computes(d_lab->d_shINLabel);
    tsk->computes(d_lab->d_soINLabel);
    tsk->computes(d_lab->d_hso2INLabel);

    tsk->computes(d_lab->d_hosoINLabel);
    tsk->computes(d_lab->d_hoso2INLabel);
    tsk->computes(d_lab->d_snINLabel);
    tsk->computes(d_lab->d_csINLabel);

    tsk->computes(d_lab->d_ocsINLabel);
    tsk->computes(d_lab->d_hsoINLabel);
    tsk->computes(d_lab->d_hosINLabel);
    tsk->computes(d_lab->d_hsohINLabel);

    tsk->computes(d_lab->d_h2soINLabel);
    tsk->computes(d_lab->d_hoshoINLabel);
    tsk->computes(d_lab->d_hs2INLabel);
    tsk->computes(d_lab->d_h2s2INLabel);
  }

  //__________________________________
  if (d_soot_precursors) {
    tsk->requires(Task::OldDW, d_lab->d_c2h2INLabel,gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_ch4INLabel, gn, 0);
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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> denMicro;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    CCVariable<double> density;
    CCVariable<double> denMicro_new;
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(denMicro,          d_lab->d_densityMicroINLabel,     indx, patch, gn, 0);
    new_dw->get(cellType,          d_lab->d_cellTypeLabel,           indx, patch, gn, 0);
    new_dw->get(voidFraction,      d_lab->d_mmgasVolFracLabel,       indx, patch, gn, 0);
    new_dw->getModifiable(density, d_lab->d_densityCPLabel,          indx, patch);
    new_dw->allocateAndPut(denMicro_new, d_lab->d_densityMicroLabel, indx, patch);
    denMicro_new.copyData(denMicro);

    constCCVariable<double> tempIN;
    constCCVariable<double> cpIN;
    constCCVariable<double> co2IN;
    constCCVariable<double> heatLoss;
    constCCVariable<double> mixMW; 

    constCCVariable<double> coIN;

    constCCVariable<double> h2sIN;
    constCCVariable<double> so2IN;
    constCCVariable<double> so3IN;
    constCCVariable<double> sulfurIN;

    constCCVariable<double> s2IN;
    constCCVariable<double> shIN;
    constCCVariable<double> soIN;
    constCCVariable<double> hso2IN;

    constCCVariable<double> hosoIN;
    constCCVariable<double> hoso2IN;
    constCCVariable<double> snIN;
    constCCVariable<double> csIN;

    constCCVariable<double> ocsIN;
    constCCVariable<double> hsoIN;
    constCCVariable<double> hosIN;
    constCCVariable<double> hsohIN;

    constCCVariable<double> h2soIN;
    constCCVariable<double> hoshoIN;
    constCCVariable<double> hs2IN;
    constCCVariable<double> h2s2IN;


    constCCVariable<double> c2h2IN;
    constCCVariable<double> ch4IN;

    CCVariable<double> coIN_new;

    CCVariable<double> h2sIN_new;
    CCVariable<double> so2IN_new;
    CCVariable<double> so3IN_new;
    CCVariable<double> sulfurIN_new;

    CCVariable<double> s2IN_new;
    CCVariable<double> shIN_new;
    CCVariable<double> soIN_new;
    CCVariable<double> hso2IN_new;

    CCVariable<double> hosoIN_new;
    CCVariable<double> hoso2IN_new;
    CCVariable<double> snIN_new;
    CCVariable<double> csIN_new;

    CCVariable<double> ocsIN_new;
    CCVariable<double> hsoIN_new;
    CCVariable<double> hosIN_new;
    CCVariable<double> hsohIN_new;

    CCVariable<double> h2soIN_new;
    CCVariable<double> hoshoIN_new;
    CCVariable<double> hs2IN_new;
    CCVariable<double> h2s2IN_new;

    CCVariable<double> c2h2IN_new;
    CCVariable<double> ch4IN_new;

    /*
    constCCVariable<double> enthalpyRXN; 
    */
    constCCVariable<double> reactScalarSrc;
    CCVariable<double> tempIN_new;
    CCVariable<double> cpIN_new;
    CCVariable<double> co2IN_new;
    CCVariable<double> heatLoss_new;
    CCVariable<double> enthalpyRXN_new;
    CCVariable<double> reactScalarSrc_new;
    CCVariable<double> mixMW_new; 
    constCCVariable<double> solidTemp;

    if (d_bc->getIfCalcEnergyExchange())
      if (d_DORadiationCalc)
        new_dw->get(solidTemp, d_MAlab->integTemp_CCLabel, indx, patch,gn, 0);

    if (d_reactingFlow) {
      old_dw->get(tempIN,   d_lab->d_tempINLabel,   indx, patch,gn, 0);
      old_dw->get(cpIN,     d_lab->d_cpINLabel,     indx, patch,gn, 0);
      old_dw->get(co2IN,    d_lab->d_co2INLabel,    indx, patch,gn, 0);
      old_dw->get(heatLoss, d_lab->d_heatLossLabel, indx, patch,gn, 0);
      //old_dw->get(mixMW,    d_lab->d_mixMWLabel,    indx, patch,gn, 0); 

      /*
      old_dw->get(enthalpyRXN, d_lab->d_enthalpyRXNLabel, indx, patch,
                  gn, 0);
      */
      if (d_calcReactingScalar) {
        old_dw->get(reactScalarSrc, d_lab->d_reactscalarSRCINLabel, indx, patch, gn, 0);
      }

      new_dw->allocateAndPut(tempIN_new,    d_lab->d_tempINLabel,   indx, patch);
      tempIN_new.copyData(tempIN);

      new_dw->allocateAndPut(cpIN_new,      d_lab->d_cpINLabel,     indx, patch);
      cpIN_new.copyData(cpIN);

      new_dw->allocateAndPut(co2IN_new,     d_lab->d_co2INLabel,    indx, patch);
      co2IN_new.copyData(co2IN);
      
      new_dw->allocateAndPut(heatLoss_new,  d_lab->d_heatLossLabel, indx, patch);
      heatLoss_new.copyData(heatLoss);

      new_dw->allocateAndPut(enthalpyRXN_new, d_lab->d_enthalpyRXNLabel, indx, patch);
      enthalpyRXN_new.initialize(0.0);

      new_dw->allocateAndPut(mixMW_new, d_lab->d_mixMWLabel, indx, patch); 
      mixMW_new.initialize(0.0); 
      //mixMW_new.copyData(mixMW); 

      if (d_calcReactingScalar) {
        new_dw->allocateAndPut(reactScalarSrc_new, d_lab->d_reactscalarSRCINLabel,indx, patch);
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
        old_dw->get(absorpIN, d_lab->d_absorpINLabel, indx, patch,
                    gn, 0);
        new_dw->allocateAndPut(absorpIN_new, d_lab->d_absorpINLabel, indx, patch);
        absorpIN_new.copyData(absorpIN);
      }


      old_dw->get(sootFVIN, d_lab->d_sootFVINLabel, indx, patch,
                  gn, 0);

      //      new_dw->allocateAndPut(abskgIN_new, d_lab->d_abskgINLabel, 
      //                       indx, patch);
      //      abskgIN_new.copyData(abskgIN);

      new_dw->allocateAndPut(sootFVIN_new, d_lab->d_sootFVINLabel, indx, patch);
      sootFVIN_new.copyData(sootFVIN);

      if (d_DORadiationCalc) {
        old_dw->get(h2oIN,            d_lab->d_h2oINLabel,           indx, patch,gn, 0);
        old_dw->get(radiationSRCIN,   d_lab->d_radiationSRCINLabel,  indx, patch,gn, 0);
        old_dw->get(radiationFluxEIN, d_lab->d_radiationFluxEINLabel,indx, patch,gn, 0);
        old_dw->get(radiationFluxWIN, d_lab->d_radiationFluxWINLabel,indx, patch,gn, 0);
        old_dw->get(radiationFluxNIN, d_lab->d_radiationFluxNINLabel,indx, patch,gn, 0);
        old_dw->get(radiationFluxSIN, d_lab->d_radiationFluxSINLabel,indx, patch,gn, 0);
        old_dw->get(radiationFluxTIN, d_lab->d_radiationFluxTINLabel,indx, patch,gn, 0);
        old_dw->get(radiationFluxBIN, d_lab->d_radiationFluxBINLabel,indx, patch,gn, 0);
        old_dw->get(abskg,            d_lab->d_abskgINLabel,         indx, patch,gn, 0);
        new_dw->allocateAndPut(h2oIN_new, d_lab->d_h2oINLabel,       indx, patch);
        h2oIN_new.copyData(h2oIN);
 
        new_dw->allocateAndPut(radiationSRCIN_new,
                               d_lab->d_radiationSRCINLabel, indx, patch);
        radiationSRCIN_new.copyData(radiationSRCIN);
 
        new_dw->allocateAndPut(radiationFluxEIN_new,
                               d_lab->d_radiationFluxEINLabel, indx, patch);
        radiationFluxEIN_new.copyData(radiationFluxEIN);
 
        new_dw->allocateAndPut(radiationFluxWIN_new,
                               d_lab->d_radiationFluxWINLabel, indx, patch);
        radiationFluxWIN_new.copyData(radiationFluxWIN);
 
        new_dw->allocateAndPut(radiationFluxNIN_new,
                               d_lab->d_radiationFluxNINLabel, indx, patch);
        radiationFluxNIN_new.copyData(radiationFluxNIN);
 
        new_dw->allocateAndPut(radiationFluxSIN_new,
                               d_lab->d_radiationFluxSINLabel, indx, patch);
        radiationFluxSIN_new.copyData(radiationFluxSIN);
 
        new_dw->allocateAndPut(radiationFluxTIN_new,
                               d_lab->d_radiationFluxTINLabel, indx, patch);
        radiationFluxTIN_new.copyData(radiationFluxTIN);
 
        new_dw->allocateAndPut(radiationFluxBIN_new,
                               d_lab->d_radiationFluxBINLabel, indx, patch);
        radiationFluxBIN_new.copyData(radiationFluxBIN);
 
        new_dw->allocateAndPut(abskg_new,
                               d_lab->d_abskgINLabel, indx, patch);
        abskg_new.copyData(abskg);
      }
    }

    if (d_co_output) {
      old_dw->get(coIN, d_lab->d_coINLabel, indx, patch,
                  gn, 0);
      new_dw->allocateAndPut(coIN_new, d_lab->d_coINLabel,indx, patch);
      coIN_new.copyData(coIN);
    }
    
    if (d_sulfur_chem) {
      old_dw->get(           h2sIN,     d_lab->d_h2sINLabel,indx, patch,gn, 0);
      new_dw->allocateAndPut(h2sIN_new, d_lab->d_h2sINLabel,indx, patch);
      h2sIN_new.copyData(h2sIN);      

      old_dw->get(           so2IN,     d_lab->d_so2INLabel, indx, patch,gn, 0);
      new_dw->allocateAndPut(so2IN_new, d_lab->d_so2INLabel, indx, patch);
      so2IN_new.copyData(so2IN);

      old_dw->get(           so3IN,     d_lab->d_so3INLabel, indx, patch,gn, 0);
      new_dw->allocateAndPut(so3IN_new, d_lab->d_so3INLabel, indx, patch);
      so3IN_new.copyData(so3IN);
      
      old_dw->get(           sulfurIN,   d_lab->d_sulfurINLabel,   indx, patch, gn, 0);
      new_dw->allocateAndPut(sulfurIN_new, d_lab->d_sulfurINLabel, indx, patch);
      sulfurIN_new.copyData(sulfurIN);

//
      old_dw->get(           s2IN,     d_lab->d_s2INLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(s2IN_new, d_lab->d_s2INLabel, indx, patch);
      s2IN_new.copyData(s2IN);

      old_dw->get(           shIN,     d_lab->d_shINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(shIN_new, d_lab->d_shINLabel, indx, patch);
      shIN_new.copyData(shIN);

      old_dw->get(           soIN,     d_lab->d_soINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(soIN_new, d_lab->d_soINLabel, indx, patch);
      soIN_new.copyData(soIN);

      old_dw->get(           hso2IN,     d_lab->d_hso2INLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hso2IN_new, d_lab->d_hso2INLabel, indx, patch);
      hso2IN_new.copyData(hso2IN);

//
      old_dw->get(           hosoIN,     d_lab->d_hosoINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hosoIN_new, d_lab->d_hosoINLabel, indx, patch);
      hosoIN_new.copyData(hosoIN);

      old_dw->get(           hoso2IN,     d_lab->d_hoso2INLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hoso2IN_new, d_lab->d_hoso2INLabel, indx, patch);
      hoso2IN_new.copyData(hoso2IN);

      old_dw->get(           snIN,     d_lab->d_snINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(snIN_new, d_lab->d_snINLabel, indx, patch);
      snIN_new.copyData(snIN);

      old_dw->get(           csIN,     d_lab->d_csINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(csIN_new, d_lab->d_csINLabel, indx, patch);
      csIN_new.copyData(csIN);

//
      old_dw->get(           ocsIN,     d_lab->d_ocsINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(ocsIN_new, d_lab->d_ocsINLabel, indx, patch);
      ocsIN_new.copyData(ocsIN);

      old_dw->get(           hsoIN,     d_lab->d_hsoINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hsoIN_new, d_lab->d_hsoINLabel, indx, patch);
      hsoIN_new.copyData(hsoIN);

      old_dw->get(           hosIN,     d_lab->d_hosINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hosIN_new, d_lab->d_hosINLabel, indx, patch);
      hosIN_new.copyData(hosIN);

      old_dw->get(           hsohIN,     d_lab->d_hsohINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hsohIN_new, d_lab->d_hsohINLabel, indx, patch);
      hsohIN_new.copyData(hsohIN);

//
      old_dw->get(           h2soIN,     d_lab->d_h2soINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(h2soIN_new, d_lab->d_h2soINLabel, indx, patch);
      h2soIN_new.copyData(h2soIN);

      old_dw->get(           hoshoIN,     d_lab->d_hoshoINLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hoshoIN_new, d_lab->d_hoshoINLabel, indx, patch);
      hoshoIN_new.copyData(hoshoIN);

      old_dw->get(           hs2IN,     d_lab->d_hs2INLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(hs2IN_new, d_lab->d_hs2INLabel, indx, patch);
      hs2IN_new.copyData(hs2IN);

      old_dw->get(           h2s2IN,    d_lab->d_h2s2INLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(h2s2IN_new, d_lab->d_h2s2INLabel,indx, patch);
      h2s2IN_new.copyData(h2s2IN);
    }
    
    if (d_soot_precursors) {
      old_dw->get(           c2h2IN,     d_lab->d_c2h2INLabel,indx, patch,gn, 0);
      new_dw->allocateAndPut(c2h2IN_new, d_lab->d_c2h2INLabel,indx, patch);
      c2h2IN_new.copyData(c2h2IN);      

      old_dw->get(           ch4IN,     d_lab->d_ch4INLabel,  indx, patch,gn, 0);
      new_dw->allocateAndPut(ch4IN_new, d_lab->d_ch4INLabel,  indx, patch);
      ch4IN_new.copyData(ch4IN);
    }

    // no need for if (d_MAlab),  since this routine is only 
    // called if d_MAlab
    IntVector indexLow = patch->getExtraCellLowIndex();
    IntVector indexHigh = patch->getExtraCellHighIndex();

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
            if (d_bc->getIfCalcEnergyExchange()) {
              if (d_DORadiationCalc) {

                if (fixTemp) 
                  tempIN_new[currCell] = 298.0;
                else
                  tempIN_new[currCell] = solidTemp[currCell];
              }
            }
          }
        }
      }
    }

    if (patch->containsCell(d_denRef)) {
      double den_ref = density[d_denRef];
      new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
    }
    else{
      new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);
    }
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
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, Ghost::None, 0);
  }

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_denRefArrayLabel);
  }else{
    tsk->modifies(d_lab->d_denRefArrayLabel);
  }
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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> denRefArray;
    constCCVariable<double> voidFraction;

    sum_vartype den_ref_var;
    new_dw->get(den_ref_var, timelabels->ref_density);

    double den_Ref = den_ref_var;

    if (d_MAlab) {
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch, Ghost::None, 0);
    }

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(denRefArray, d_lab->d_denRefArrayLabel,  indx, patch);
    }else{
      new_dw->getModifiable(denRefArray, d_lab->d_denRefArrayLabel,   indx, patch);
    }  
              
    denRefArray.initialize(den_Ref);

    if (d_MAlab) {
      for (CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
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
                          
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel,      gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,   gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalarFELabel,      gn, 0);
  
  tsk->modifies(d_lab->d_scalarSPLabel);
  tsk->modifies(d_lab->d_densityGuessLabel);
  
  
  if (d_calcReactingScalar){
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel,gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_reactscalarFELabel,gn, 0);
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  }
  
  if (d_calcEnthalpy){
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_enthalpyFELabel,   gn, 0);
    tsk->modifies(d_lab->d_enthalpySPLabel);
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
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
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

    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(old_density,  d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    old_dw->get(old_scalar,   d_lab->d_scalarSPLabel,     indx, patch, gn, 0);
    
    new_dw->get(rho1_density, d_lab->d_densityTempLabel,  indx, patch, gn, 0);
    new_dw->get(new_density,  d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    new_dw->get(fe_scalar,    d_lab->d_scalarFELabel,     indx, patch, gn, 0);
    
    new_dw->getModifiable(new_scalar,    d_lab->d_scalarSPLabel,     indx, patch);
    new_dw->getModifiable(density_guess, d_lab->d_densityGuessLabel, indx, patch);
    
    
    
    if (d_calcReactingScalar) {
      old_dw->get(old_reactScalar,d_lab->d_reactscalarSPLabel, indx, patch, gn, 0);
      new_dw->get(fe_reactScalar, d_lab->d_reactscalarFELabel, indx, patch, gn, 0);
      new_dw->getModifiable(new_reactScalar, d_lab->d_reactscalarSPLabel,indx, patch);
      
    }
    if (d_calcEnthalpy){
      old_dw->get(old_enthalpy,   d_lab->d_enthalpySPLabel,     indx, patch, gn, 0);
      new_dw->get(fe_enthalpy,    d_lab->d_enthalpyFELabel,     indx, patch, gn, 0);
      new_dw->getModifiable(new_enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
    }



    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;
    double epsilon = 1.0e-15;

    IntVector indexLow  = patch->getExtraCellLowIndex();
    IntVector indexHigh = patch->getExtraCellHighIndex();

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          
          // The following if statement is to eliminate Kumar's zero block density problem
          if (new_density[currCell] > 0.0) {
            double predicted_density;
            if (old_density[currCell] > 0.0) {
              //            predicted_density = rho1_density[currCell];
              if (d_inverse_density_average)
                predicted_density = 1.0/((factor_old/old_density[currCell] + factor_new/new_density[currCell])/factor_divide);
              else
                predicted_density = (factor_old*old_density[currCell] + factor_new*new_density[currCell])/factor_divide;
            }
            else {
              predicted_density = new_density[currCell];
            }

            bool average_failed = false;
            if (d_inverse_density_average) {
              new_scalar[currCell] = (factor_old*old_scalar[currCell] +
                                      factor_new*new_scalar[currCell])/factor_divide;
            }
            else {
              new_scalar[currCell] = (factor_old*old_density[currCell]*
                                      old_scalar[currCell] + factor_new*new_density[currCell]*
                                      new_scalar[currCell])/(factor_divide*predicted_density);
            }
            // Following lines to fix density delay problem for helium.
            // One would also need to edit fortran/explicit.F to use it.
            //            (new_scalar)[currCell] = (new_scalar)[currCell]*predicted_density;
            //            (new_scalar)[currCell] = (new_scalar)[currCell]*0.133/(
            //              0.133*1.184344+(new_scalar)[currCell]*(0.133-1.184344));
            if (new_scalar[currCell] > 1.0) {
              if (new_scalar[currCell] < 1.0 + epsilon) {
                new_scalar[currCell] = 1.0;
              }
              else {
                cout << "average failed with scalar > 1 at " << currCell << " , average value was " << new_scalar[currCell] << endl;
                new_scalar[currCell] = fe_scalar[currCell];
                average_failed = true;
              }
            }
            else if (new_scalar[currCell] < 0.0) {
              if (new_scalar[currCell] > - epsilon) {
                new_scalar[currCell] = 0.0;
              }
              else {
                cout << "average failed with scalar < 0 at " << currCell << " , average value was " << new_scalar[currCell] << endl;
                new_scalar[currCell] = fe_scalar[currCell];
                average_failed = true;
              }
            }
            
            if( d_calcEnthalpy ) {
              if( !average_failed ) {
                new_enthalpy[currCell] = (factor_old*old_density[currCell]*
                                          old_enthalpy[currCell] + factor_new*new_density[currCell]*
                                          new_enthalpy[currCell])/(factor_divide*predicted_density);
              }
              else {
                new_enthalpy[currCell] = fe_enthalpy[currCell];
              }
            }
            density_guess[currCell] = predicted_density;
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
Properties::sched_saveTempDensity(SchedulerP& sched, 
                                  const PatchSet* patches,
                                  const MaterialSet* matls,
                                  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::saveTempDensity" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &Properties::saveTempDensity,
                          timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None, 0);
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
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> temp_density;

    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,indx, patch);
    new_dw->copyOut(temp_density,      d_lab->d_densityCPLabel,   indx, patch);
  }
}
//****************************************************************************
// Schedule the computation of drhodt
//****************************************************************************
void 
Properties::sched_computeDrhodt(SchedulerP& sched, 
                                const PatchSet* patches,
                                const MaterialSet* matls, 
                                const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::computeDrhodt" +
                     timelabels->integrator_step_name;
  
  Task* tsk = scinew Task(taskname, this,
                          &Properties::computeDrhodt,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(parent_old_dw, d_lab->d_oldDeltaTLabel);
  tsk->requires(parent_old_dw, d_lab->d_densityOldOldLabel, gn,0);

  tsk->requires(Task::NewDW,    d_lab->d_densityCPLabel,   gn,0);
  tsk->requires(parent_old_dw, d_lab->d_densityCPLabel,   gn,0);
  

  if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
    tsk->computes(d_lab->d_filterdrhodtLabel);
    tsk->computes(d_lab->d_oldDeltaTLabel);
    tsk->computes(d_lab->d_densityOldOldLabel);
  }
  else{
    tsk->modifies(d_lab->d_filterdrhodtLabel);
  }
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
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
   parent_old_dw = old_dw;
  }
  
  int drhodt_1st_order = 1;
  int current_step = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  if (d_MAlab){ 
    drhodt_1st_order = 2;
  }
  delt_vartype delT, old_delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  
  
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    new_dw->put(delT, d_lab->d_oldDeltaTLabel);
  }
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;
  delta_t *= timelabels->time_position_multiplier_after_average;
  
  parent_old_dw->get(old_delT, d_lab->d_oldDeltaTLabel);
  double  old_delta_t = old_delT;

  //__________________________________
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> new_density;
    constCCVariable<double> old_density;
    constCCVariable<double> old_old_density;
    CCVariable<double> drhodt;
    CCVariable<double> filterdrhodt;
    CCVariable<double> density_oldold;
    Ghost::GhostType  gn = Ghost::None;
    
    parent_old_dw->get(old_density,     d_lab->d_densityCPLabel,     indx, patch,gn, 0);
    parent_old_dw->get(old_old_density, d_lab->d_densityOldOldLabel, indx, patch,gn, 0);
    
    
    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
      new_dw->allocateAndPut(density_oldold, d_lab->d_densityOldOldLabel, indx, patch);
      density_oldold.copyData(old_density);
    }

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(new_density, d_lab->d_densityCPLabel,  indx, patch, gn, 0);
    
    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
      new_dw->allocateAndPut(filterdrhodt, d_lab->d_filterdrhodtLabel, indx, patch);
    }else{
      new_dw->getModifiable(filterdrhodt, d_lab->d_filterdrhodtLabel,  indx, patch);
    }
    filterdrhodt.initialize(0.0);

    // Get the patch and variable indices
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    // compute drhodt and its filtered value
    drhodt.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    drhodt.initialize(0.0);

    //__________________________________
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
    d_filter->applyFilter<CCVariable<double> >(pc, patch, drhodt, filterdrhodt);
#else
    // filtering without petsc is not implemented
    // if it needs to be then drhodt will have to be computed with ghostcells
    filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
                      drhodt.getHighIndex());
#endif
    }else{
      filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
                      drhodt.getHighIndex());
    }

    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
          IntVector currcell(ii,jj,kk);

          double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
          if (Abs(filterdrhodt[currcell]/vol) < 1.0e-9){
              filterdrhodt[currcell] = 0.0;
          }
        }
      }
    }
  }
}

void
Properties::sched_reComputeProps_new( const LevelP& level,
                                      SchedulerP& sched,
                                      const TimeIntegratorLabel* time_labels, 
                                      const bool initialize, 
                                      const bool modify_ref_den )
{
  // this method is temporary while we get rid of properties.cc 
  if ( ! d_newEnthalpySolver ) { 
    d_mixingRxnTable->sched_computeHeatLoss( level, sched, initialize, d_calcEnthalpy );
  } else { 
    d_mixingRxnTable->sched_computeHeatLoss( level, sched, initialize, d_newEnthalpySolver );
  } 

  d_mixingRxnTable->sched_getState( level, sched, time_labels, initialize, d_calcEnthalpy, modify_ref_den ); 
}

void 
Properties::sched_initEnthalpy( const LevelP& level, SchedulerP& sched )
{
  d_mixingRxnTable->sched_computeFirstEnthalpy( level, sched ) ; 
}
void 
Properties::sched_doTPDummyInit( const LevelP& level, SchedulerP& sched )
{
  d_mixingRxnTable->sched_dummyInit( level, sched ); 
}
void 
Properties::addLookupSpecies( ){

  std::vector<std::string> sps; 
  sps = d_lab->model_req_species; 

  if ( mixModel == "ClassicTable"  || mixModel == "TabProps" ) { 
    for ( vector<string>::iterator i = sps.begin(); i != sps.end(); i++ ){
      d_mixingRxnTable->insertIntoMap( *i ); 
    }
  }
}

void 
Properties::doTableMatching(){ 

	d_mixingRxnTable->tableMatching(); 

}
