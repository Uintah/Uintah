/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


//----- Arches.cc ----------------------------------------------
#include <CCA/Components/Arches/DQMOM.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ExplicitSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/PicardNonlinearSolver.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/SmagorinskyModel.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/IncDynamicProcedure.h>
#include <CCA/Components/Arches/CompDynamicProcedure.h>
#include <CCA/Components/Arches/CompLocalDynamicProcedure.h>
#include <CCA/Components/Arches/ExtraScalarSolver.h>
#include <CCA/Components/Arches/OdtClosure.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/Parallel.h>

#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <fstream>

using std::endl;

using std::string;
using namespace Uintah;
using namespace SCIRun;
#ifdef PetscFilter
#include <CCA/Components/Arches/Filter.h>
#endif

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
  d_calcScalar = 0;
  d_calcEnthalpy =0;
  d_doingRestart = false; 
  d_newBC_on_Restart = false; 
#ifdef multimaterialform
  d_mmInterface = 0;
#endif
  nofTimeSteps = 0;
  init_timelabel_allocated = false;
  d_analysisModule = false;
  d_calcExtraScalars = false;
  d_extraScalarSolver = 0;
  d_set_initial_condition = false;

  DQMOMEqnFactory& dqmomfactory = DQMOMEqnFactory::self(); 
  dqmomfactory.set_quad_nodes(0);
  d_doDQMOM = false; 
  d_doMMS = false;
  
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
  if (d_analysisModule) {
    delete d_analysisModule;
  }
  if (d_calcExtraScalars)
    for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++)
      delete d_extraScalars[i];

  delete d_timeIntegrator; 

  if (d_doDQMOM) { 
    delete d_dqmomSolver; 
  }
}

// ****************************************************************************
// problem set up
// ****************************************************************************
void 
Arches::problemSetup(const ProblemSpecP& params, 
                     const ProblemSpecP& materials_ps, 
                     GridP& grid, SimulationStateP& sharedState)
{

  d_sharedState= sharedState;
  d_lab->setSharedState(sharedState);
  ArchesMaterial* mat= scinew ArchesMaterial();
  sharedState->registerArchesMaterial(mat);
  ProblemSpecP db = params->findBlock("CFD")->findBlock("ARCHES");

  // This will allow for changing the BC's on restart:
  if ( db->findBlock("new_BC_on_restart") )
    d_newBC_on_Restart = true; 
  else 
    d_newBC_on_Restart = false; 

  // not sure, do we need to reduce and put in datawarehouse
  if (db->findBlock("ExplicitSolver")){
    if (db->findBlock("ExplicitSolver")->findBlock("MixtureFractionSolver"))
      d_calcScalar = true;
      db->findBlock("ExplicitSolver")->findBlock("MixtureFractionSolver")->getWithDefault("initial_value",d_init_mix_frac,0.0); 
  } else if (db->findBlock("PicardSolver")){
    if (db->findBlock("PicardSolver")->findBlock("MixtureFractionSolver"))
      d_calcScalar = true;
  }
  if (!d_calcScalar)
    throw InvalidValue("Density being independent variable or equivalently mixture fraction transport disabled is not supported in current implementation. Please include the <MixtureFractionSolver> section as a child of <Arches>.", __FILE__, __LINE__);

  if (db->findBlock("set_initial_condition")) {
    d_set_initial_condition = true;
    db->findBlock("set_initial_condition")->getAttribute("inputfile",d_init_inputfile);
  }

//  db->getWithDefault("set_initial_condition",d_set_initial_condition,false);
//  if (d_set_initial_condition)
//    db->require("init_cond_input_file", d_init_inputfile);
//
  if (d_calcScalar) {
    if (d_calcReactingScalar) {
      throw InvalidValue("Transport of reacting scalar is being phased out.  Please email j.thornock@utah.edu if you have questions.", __FILE__, __LINE__);
      d_calcReactingScalar = false; 
    }

    if (db->findBlock("ExplicitSolver")){
      if (db->findBlock("ExplicitSolver")->findBlock("EnthalpySolver"))
        d_calcEnthalpy = true; 
    } else if (db->findBlock("PicardSolver")) {
      if (db->findBlock("PicardSolver")->findBlock("EnthalpySolver"))
        d_calcEnthalpy = true;
    }
    // Moved model_mixture_fraction_variance to properties
    db->findBlock("Properties")->require("use_mixing_model", d_calcVariance);
    // db->require("model_mixture_fraction_variance", d_calcVariance);
  }
  db->getWithDefault("turnonMixedModel",    d_mixedModel,false);
  db->getWithDefault("recompileTaskgraph",  d_recompile,false);

  // Shouldn't this block go in the nonlinear solver's problemSetup()?  - Charles
  string nlSolver;
  if (db->findBlock("ExplicitSolver")){
    nlSolver = "explicit";
    db->findBlock("ExplicitSolver")->getWithDefault("EKTCorrection", d_EKTCorrection,false);  
    db->findBlock("ExplicitSolver")->getWithDefault("scalarUnderflowCheck",d_underflow,false);
    db->findBlock("ExplicitSolver")->getWithDefault("extraProjection",     d_extraProjection,false);  
    db->findBlock("ExplicitSolver")->require("initial_dt", d_init_dt);
    db->findBlock("ExplicitSolver")->require("variable_dt", d_variableTimeStep);
    
  } else if (db->findBlock("PicardSolver")){
    nlSolver = "picard";
    db->findBlock("PicardSolver")->getWithDefault("EKTCorrection", d_EKTCorrection,false);  
    db->findBlock("PicardSolver")->getWithDefault("scalarUnderflowCheck",d_underflow,false);
    db->findBlock("PicardSolver")->getWithDefault("extraProjection",     d_extraProjection,false);  
    db->findBlock("PicardSolver")->require("initial_dt", d_init_dt);
    db->findBlock("PicardSolver")->require("variable_dt", d_variableTimeStep);
  }

  if(db->findBlock("MMS")) {

    d_useMMSScalar = false;
    d_useMMSTable = false;

    d_doMMS = true;
    ProblemSpecP db_mms = db->findBlock("MMS");
    if( !db_mms->getAttribute( "whichMMS", d_mms ) ) {
      throw ProblemSetupException( "whichMMS not specified", __FILE__, __LINE__);      
    }
    if (d_mms == "constantMMS") {
      ProblemSpecP db_mms0 = db_mms->findBlock("constantMMS");
      db_mms0->getWithDefault("cu",d_cu,0.0);
      db_mms0->getWithDefault("cv",d_cv,0.0);
      db_mms0->getWithDefault("cw",d_cw,0.0);
      db_mms0->getWithDefault("cp",d_cp,0.0);
      
      // Allow a choice to either specify mixture fraction or specify a table 
      if( db_mms0->findBlock("phi0") ) {
        db_mms0->getWithDefault("phi0",d_phi0,0.0);
        d_useMMSScalar = true;
      } else if( db_mms0->findBlock("useTable") ) {
        d_useMMSTable = true;
      } else {
        throw ProblemSetupException("ERROR: Arches: You must specify either an MMS scalar using <phi0>, or an MMS table using <useTable>, but you did not specify either!\n",__FILE__,__LINE__);
      }
      db_mms0->getWithDefault("esphi0",d_esphi0,0.0);
    }
    else if (d_mms == "almgrenMMS") {
      ProblemSpecP db_mms3 = db_mms->findBlock("almgrenMMS");
      db_mms3->require("amplitude",d_amp);
    }
    else {
      throw InvalidValue("current MMS not supported: " + d_mms, __FILE__, __LINE__);
    }
  }

  // physical constant
  d_physicalConsts = scinew PhysicalConstants();
  const ProblemSpecP db_root = db->getRootNode();
  d_physicalConsts->problemSetup(db_root);

  if (db->findBlock("ExtraScalars")) {
    d_calcExtraScalars = true; 
    ProblemSpecP extra_sc_db = db->findBlock("ExtraScalars");
    for (ProblemSpecP scalar_db = extra_sc_db->findBlock("scalar");
         scalar_db != 0; scalar_db = scalar_db->findNextBlock("scalar")) {
      d_extraScalarSolver = scinew ExtraScalarSolver(d_lab, d_MAlab,
                                                     d_physicalConsts);
      d_extraScalarSolver->problemSetup(scalar_db);
      d_extraScalars.push_back(d_extraScalarSolver);
    }
  }

  //create a time integrator.
  d_timeIntegrator = scinew ExplicitTimeInt(d_lab);
  ProblemSpecP time_db = db->findBlock("TimeIntegrator");
  if (time_db) {
    string time_order; 
    time_db->findBlock("ExplicitIntegrator")->getAttribute("order", time_order); 
    if (time_order == "first") {
      d_tOrder = 1;
    } else if (time_order == "second") {
      d_tOrder = 2; 
    } else if (time_order == "third") {
      d_tOrder = 3; 
    } else {
      throw InvalidValue("Explicit time integrator must be one of: first, second, third!  Please fix input file",__FILE__,__LINE__);
    }

    d_timeIntegrator->problemSetup(time_db);
  }

  // Do this always (factory initialization methods are run whether the factory 
  // is used or not, and without labels the initialization methods will crash)
  EqnFactory& eqnFactory = EqnFactory::self();
  eqnFactory.setArchesLabel( d_lab );
  eqnFactory.setTimeIntegrator(d_timeIntegrator);

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  dqmomFactory.setArchesLabel( d_lab );
  dqmomFactory.setTimeIntegrator(d_timeIntegrator);

  SourceTermFactory& srcFactory = SourceTermFactory::self();
  srcFactory.setArchesLabel( d_lab );

  CoalModelFactory& coalFactory = CoalModelFactory::self();
  coalFactory.setArchesLabel( d_lab );

  ProblemSpecP transportEqn_db = db->findBlock("TransportEqns");
  if (transportEqn_db) {
    // register transport eqns
    eqnFactory.problemSetup(transportEqn_db);

    // register source terms
    ProblemSpecP sources_db = transportEqn_db->findBlock("Sources");
    if (sources_db) {
      srcFactory.problemSetup(sources_db);
    }
  
  }

  // read properties
  // d_MAlab = multimaterial arches common labels
  d_props = scinew Properties(d_lab, d_MAlab, d_physicalConsts,
                              d_calcReactingScalar, 
                              d_calcEnthalpy, d_calcVariance, d_myworld);

  d_props->setCalcExtraScalars(d_calcExtraScalars);

  if (d_calcExtraScalars){
    d_props->setExtraScalars(&d_extraScalars);
  }
  
  d_props->problemSetup(db);

  coalFactory.setTabPropsInterface( d_props->getMixingRxnModel() );

  // read boundary condition information 
  d_boundaryCondition = scinew BoundaryCondition(d_lab, d_MAlab, d_physicalConsts,
                                                 d_props, d_calcReactingScalar,
                                                 d_calcEnthalpy, d_calcVariance);
  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->setMMS(d_doMMS);
  d_boundaryCondition->setCalcExtraScalars(d_calcExtraScalars);
  if (d_calcExtraScalars){
    d_boundaryCondition->setExtraScalars(&d_extraScalars);
  }
  d_boundaryCondition->problemSetup(db);

  d_carbon_balance_es = d_boundaryCondition->getCarbonBalanceES();
  d_sulfur_balance_es = d_boundaryCondition->getSulfurBalanceES();
  d_props->setCarbonBalanceES(d_carbon_balance_es);        
  d_props->setSulfurBalanceES(d_sulfur_balance_es);

  ProblemSpecP turb_db = db->findBlock("Turbulence");
  turb_db->getAttribute("model", d_whichTurbModel); 

  //db->require("turbulence_model", turbModel);
  if ( d_whichTurbModel == "smagorinsky"){ 
    d_turbModel = scinew SmagorinskyModel(d_lab, d_MAlab, d_physicalConsts,
                                          d_boundaryCondition);
  }else  if ( d_whichTurbModel == "dynamicprocedure"){ 
    d_turbModel = scinew IncDynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
                                          d_boundaryCondition);
  }else if ( d_whichTurbModel == "compdynamicprocedure"){
    d_turbModel = scinew CompDynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
                                          d_boundaryCondition);
  }else if ( d_whichTurbModel == "complocaldynamicprocedure") {
    d_initTurb = scinew CompLocalDynamicProcedure(d_lab, d_MAlab, d_physicalConsts, d_boundaryCondition); 
    d_turbModel = scinew CompLocalDynamicProcedure(d_lab, d_MAlab, d_physicalConsts, d_boundaryCondition);
  } else {
    throw InvalidValue("Turbulence Model not supported" + d_whichTurbModel, __FILE__, __LINE__);
  }

  d_turbModel->modelVariance(d_calcVariance);
  d_turbModel->problemSetup(db);
  d_dynScalarModel = d_turbModel->getDynScalarModel();
  if (d_dynScalarModel){
    d_turbModel->setCombustionSpecifics(d_calcScalar, d_calcEnthalpy,
                                        d_calcReactingScalar);
  }

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

  if (d_calcExtraScalars){
    for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++) {
      d_extraScalars[i]->setTurbulenceModel(d_turbModel);
      d_extraScalars[i]->setBoundaryCondition(d_boundaryCondition);
    }
  }

  if(nlSolver == "picard") {
    d_nlSolver = scinew PicardNonlinearSolver( d_lab, d_MAlab, d_props, 
                                               d_boundaryCondition,
                                               d_turbModel, d_physicalConsts,
                                               d_calcScalar,
                                               d_calcReactingScalar,
                                               d_calcEnthalpy,
                                               d_calcVariance,
                                               d_myworld);
    if (d_calcExtraScalars){
      throw InvalidValue("Transport of extra scalars by picard solver is not implemented", __FILE__, __LINE__);
    }
  } else if (nlSolver == "explicit") {
    d_nlSolver = scinew ExplicitSolver( d_lab, d_MAlab, d_props,
                                        d_boundaryCondition,
                                        d_turbModel, d_scaleSimilarityModel, 
                                        d_physicalConsts,
                                        d_calcScalar,
                                        d_calcReactingScalar,
                                        d_calcEnthalpy,
                                        d_calcVariance,
                                        d_myworld);

  } else{
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver, __FILE__, __LINE__);
  }

  d_nlSolver->setExtraProjection(d_extraProjection);
  d_nlSolver->setEKTCorrection(d_EKTCorrection);
  d_nlSolver->setMMS(d_doMMS);
  d_nlSolver->problemSetup(db);
  d_nlSolver->setCarbonBalanceES(d_carbon_balance_es);
  d_nlSolver->setSulfurBalanceES(d_sulfur_balance_es);
  d_timeIntegratorType = d_nlSolver->getTimeIntegratorType();
  d_nlSolver->setCalcExtraScalars(d_calcExtraScalars);
  if (d_calcExtraScalars) {
    d_nlSolver->setExtraScalars(&d_extraScalars);
  }


  //__________________
  // Init data analysis module(s) and run problemSetup
  // note we're just calling a child problemSetup with mostly
  // the same parameters

  
  //__________________
  //This is not the proper way to get our DA.  Scheduler should
  //pass us a DW pointer on every function call.  I don't think
  //AnalysisModule should retain the pointer in a field, IMHO.
  // - Steve
  Output* dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!dataArchiver){
    throw InternalError("ARCHES:couldn't get output port", __FILE__, __LINE__);
  }
 
  d_analysisModule = AnalysisModuleFactory::create(params, sharedState, dataArchiver);
  if (d_analysisModule) {
    d_analysisModule->problemSetup(params, grid, sharedState);
  }

  // ----- DQMOM STUFF:

  // If there is at least one DQMOM block, set up the DQMOM equation factory
  if ( db->findBlock("DQMOM") ) {
    d_doDQMOM = true;
    
    d_dqmomSolver = scinew DQMOM(d_lab);

    // assume there is only 1 <DQMOM> for now...
    ProblemSpecP dqmom_db = db->findBlock("DQMOM");
    dqmomFactory.problemSetup( dqmom_db );
		coalFactory.problemSetup(dqmom_db);

    d_dqmomSolver->problemSetup( dqmom_db ); 
    //d_dqmomSolvers.push_back(dqmomSolver);

    dqmomFactory.setDQMOMSolver( d_dqmomSolver );
  }

  // TODO
  // Looping over all <DQMOM> blocks will require changing DQMOMEqnFactory
  // (e.g. getting the number of quad nodes)
  //
  // Will also require changing all the classes that are using DQMOMEqnFactory...
  // (particularly those grabbing # quad nodes)

  // Loop through alll <DQMOM> blocks to create a model factory for each
  /*
  for( ProblemSpecP dqmom_db = db->findBlock("DQMOM"); 
       dqmom_db != 0; dqmom_db = dqmom_db->findNextBlock("DQMOM")) {

    if( !time_db ) {
      throw ProblemSetupException("ERROR: Arches: <DQMOM> block used without a <TimeIntegrator> block. You must add a time integrator to use DQMOM.",__FILE__,__LINE__);
    }

    d_doDQMOM = true;

    // Figure out what kind of DQMOM factory to create
    string dqmom_type;
    dqmom_db->getWithDefault("type",dqmom_type,"none");

    // Depending on the type, need to initialize different model factories
    if( dqmom_type == "coal" ) {
      // Set up coal model factory
      CoalModelFactory& model_factory = CoalModelFactory::self();
      model_factory.setArchesLabel( d_lab ); 
		  model_factory.problemSetup(dqmom_db);

    } else if (dqmom_type == "soot" ) {
      throw ProblemSetupException("ERROR: Arches: DQMOM for soot is not currently supported.",__FILE__,__LINE__);

    } else if (dqmom_type == "none" ) {
      // TODO 
      // add a generic DQMOM models folder
      // with a generic DQMOM models factory
    }

    // Set up the DQMOM linear solver:
    DQMOM* dqmomSolver = scinew DQMOM(d_lab);
    dqmomSolver->problemSetup( dqmom_db ); 
    d_dqmomSolvers.push_back(dqmomSolver);
  }
  */


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


  // -----------------------------------
  // Scalar equations (and source terms) initialization
  EqnFactory& eqnFactory = EqnFactory::self(); 
  eqnFactory.sched_scalarInit(level, sched);

  // check to make sure that all the scalar variables have BCs set. 
  EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns(); 
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
    EqnBase* eqn = ieqn->second; 
    eqn->sched_checkBCs( level, sched ); 
  }

  SourceTermFactory& sourceFactory = SourceTermFactory::self();
  sourceFactory.sched_sourceInit(level, sched);


  // ----------------------------------
  if (d_set_initial_condition) {
    sched_readCCInitialCondition(level, sched);
    sched_interpInitialConditionToStaggeredGrid(level, sched);
  }

  //mms initial condition
  if (d_doMMS) {
    sched_mmsInitialCondition(level, sched);
  }

  //========= MOM debugging ===========
  bool debug_mom = false;
  if (debug_mom){
    sched_blobInit(level, sched);
  }
  //===================================

  // schedule init of cell type
  // require : NONE
  // compute : cellType
  d_boundaryCondition->sched_cellTypeInit(sched, patches, matls);

  // computing flow inlet areas
  if (d_boundaryCondition->getInletBC()){
    d_boundaryCondition->sched_calculateArea(sched, patches, matls);
  }
  // Set the profile (output Varlabel have SP appended to them)
  // require : densityIN,[u,v,w]VelocityIN
  // compute : densitySP, [u,v,w]VelocitySP, scalarSP
  d_boundaryCondition->sched_setProfile(sched, patches, matls);
  d_boundaryCondition->sched_Prefill(sched, patches, matls);

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


  string mixmodel = d_props->getMixingModelType(); 
  if ( mixmodel != "TabProps") {
    d_props->sched_reComputeProps(sched, patches, matls,
                                init_timelabel, true, true, false,false);
  } else {
    bool initialize_it = true; 
    bool modify_ref_den = true; 
    if ( d_calcEnthalpy) 
      d_props->sched_initEnthalpy( level, sched ); 
    d_props->sched_reComputeProps_new( level, sched, init_timelabel, initialize_it, modify_ref_den ); 
  }

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

    if (d_whichTurbModel == "complocaldynamicprocedure")
      d_initTurb->sched_initializeSmagCoeff(sched, patches, matls, init_timelabel);
    else
      d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls, init_timelabel);
  }

  //______________________
  //Data Analysis
  if (d_analysisModule) {
    d_analysisModule->scheduleInitialize(sched, level);
  }


  //----------------------
  //DQMOM initialization 
  if(d_doDQMOM) {
    DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self(); 
    CoalModelFactory& model_factory = CoalModelFactory::self();

    // initialize DQMOM transport equations
    dqmom_factory.sched_weightInit(level, sched);
    dqmom_factory.sched_weightedAbscissaInit(level, sched);

    // initialize DQMOM models
    model_factory.sched_modelInit(level, sched);

    // check to make sure that all dqmom equations have BCs set. 
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns(); 
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
      EqnBase* eqn = ieqn->second; 
      eqn->sched_checkBCs( level, sched ); 
    }

  }

  // compute the cell area fraction 
  d_boundaryCondition->sched_setAreaFraction( sched, patches, matls ); 
    
}

void
Arches::restartInitialize()
{
  d_doingRestart = true; 
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

    tsk->computes(d_lab->d_cellInfoLabel);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
    tsk->computes(d_lab->d_wVelocitySPBCLabel);
    tsk->computes(d_lab->d_uVelRhoHatLabel);
    tsk->computes(d_lab->d_vVelRhoHatLabel);
    tsk->computes(d_lab->d_wVelRhoHatLabel);
    tsk->computes(d_lab->d_newCCVelocityLabel);
    tsk->computes(d_lab->d_newCCUVelocityLabel);
    tsk->computes(d_lab->d_newCCVVelocityLabel);
    tsk->computes(d_lab->d_newCCWVelocityLabel);
    tsk->computes(d_lab->d_pressurePSLabel);
    tsk->computes(d_lab->d_areaFractionLabel); 
    if ((d_extraProjection)||(d_EKTCorrection)){
      tsk->computes(d_lab->d_pressureExtraProjectionLabel);
    }
    if (!((d_timeIntegratorType == "FE")||(d_timeIntegratorType == "BE"))){
      tsk->computes(d_lab->d_pressurePredLabel);
    }
    if (d_timeIntegratorType == "RK3SSP"){
      tsk->computes(d_lab->d_pressureIntermLabel);
    }
    
    if (d_calcScalar){
      tsk->computes(d_lab->d_scalarSPLabel); // only work for 1 scalar
    }
    
    if (d_calcVariance) {
      tsk->computes(d_lab->d_scalarVarSPLabel); // only work for 1 scalarVar
      tsk->computes(d_lab->d_normalizedScalarVarLabel); // only work for 1 scalarVar
      tsk->computes(d_lab->d_scalarDissSPLabel); // only work for 1 scalarVar
    }
    
    if (d_calcReactingScalar){
      tsk->computes(d_lab->d_reactscalarSPLabel);
    }
    
    if (d_calcEnthalpy) {
      tsk->computes(d_lab->d_enthalpySPLabel); 
      tsk->computes(d_lab->d_radiationSRCINLabel);
      tsk->computes(d_lab->d_radiationFluxEINLabel); 
      tsk->computes(d_lab->d_radiationFluxWINLabel);
      tsk->computes(d_lab->d_radiationFluxNINLabel);
      tsk->computes(d_lab->d_radiationFluxSINLabel); 
      tsk->computes(d_lab->d_radiationFluxTINLabel);
      tsk->computes(d_lab->d_radiationFluxBINLabel);
      tsk->computes(d_lab->d_radiationVolqINLabel); 
      tsk->computes(d_lab->d_abskgINLabel); 
    }

    tsk->computes(d_lab->d_densityCPLabel);
    tsk->computes(d_lab->d_viscosityCTSLabel);
    if (d_dynScalarModel) {
      if (d_calcScalar){
        tsk->computes(d_lab->d_scalarDiffusivityLabel);
      }
      if (d_calcEnthalpy){
        tsk->computes(d_lab->d_enthalpyDiffusivityLabel);
      }
      if (d_calcReactingScalar){
        tsk->computes(d_lab->d_reactScalarDiffusivityLabel);
      }
    }
    tsk->computes(d_lab->d_oldDeltaTLabel);
    // for reacting flows save temperature and co2 
    if (d_MAlab) {
      tsk->computes(d_lab->d_pressPlusHydroLabel);
      tsk->computes(d_lab->d_mmgasVolFracLabel);
    }

    if (d_doMMS) {
      tsk->computes(d_lab->d_uFmmsLabel);
      tsk->computes(d_lab->d_vFmmsLabel);
      tsk->computes(d_lab->d_wFmmsLabel);
    }
    
    if (d_calcExtraScalars){
      for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++){
        tsk->computes(d_extraScalars[i]->getScalarLabel());
      }
    }
    if (d_carbon_balance_es){
      tsk->computes(d_lab->d_co2RateLabel);
    }
    if (d_sulfur_balance_es){
      tsk->computes(d_lab->d_so2RateLabel);                
    }
    tsk->computes(d_lab->d_scalarBoundarySrcLabel);
    tsk->computes(d_lab->d_enthalpyBoundarySrcLabel);
    tsk->computes(d_lab->d_umomBoundarySrcLabel);
    tsk->computes(d_lab->d_vmomBoundarySrcLabel);
    tsk->computes(d_lab->d_wmomBoundarySrcLabel);

    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

// ****************************************************************************
// Actual initialization
// ****************************************************************************
void
Arches::paramInit(const ProcessorGroup* pg,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw)
{
    double old_delta_t = 0.0;
    new_dw->put(delt_vartype(old_delta_t), d_lab->d_oldDeltaTLabel,getLevel(patches));

  // ....but will only compute for computational domain
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // Initialize cellInformation
    PerPatch<CellInformationP> cellInfoP;
    cellInfoP.setData(scinew CellInformation(patch));
    new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);

    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;
    CCVariable<double> uVelocityCC;
    CCVariable<double> vVelocityCC;
    CCVariable<double> wVelocityCC;
    CCVariable<Vector> ccVelocity; 
    CCVariable<double> pressure;
    CCVariable<double> pressureExtraProjection;
    CCVariable<double> pressurePred;
    CCVariable<double> pressureInterm;
    CCVariable<double> scalar;
    CCVariable<double> scalarVar_new;
    CCVariable<double> normalizedScalarVar_new;
    CCVariable<double> scalarDiss_new;
    CCVariable<double> enthalpy;
    CCVariable<double> density;
    CCVariable<double> viscosity;
    CCVariable<double> scalarDiffusivity;
    CCVariable<double> enthalpyDiffusivity;
    CCVariable<double> reactScalarDiffusivity;
    CCVariable<double> pPlusHydro;
    CCVariable<double> mmgasVolFrac;
    CCVariable<Vector> areaFraction; 

    new_dw->allocateAndPut( areaFraction, d_lab->d_areaFractionLabel, indx, patch ); 
    areaFraction.initialize(Vector(1.,1.,1.)); 
   
    if (d_calcExtraScalars){
      if (d_carbon_balance_es){ 
        CCVariable<double> co2Rate;
        new_dw->allocateAndPut(co2Rate, d_lab->d_co2RateLabel, indx, patch);
        co2Rate.initialize(0.0);
      }
      if (d_sulfur_balance_es){ 
        CCVariable<double> so2Rate;
        new_dw->allocateAndPut(so2Rate, d_lab->d_so2RateLabel, indx, patch);
        so2Rate.initialize(0.0);
      }
    }        

    CCVariable<double> scalarBoundarySrc;
    CCVariable<double> enthalpyBoundarySrc;
    SFCXVariable<double> umomBoundarySrc;
    SFCYVariable<double> vmomBoundarySrc;
    SFCZVariable<double> wmomBoundarySrc;

    new_dw->allocateAndPut(scalarBoundarySrc,   d_lab->d_scalarBoundarySrcLabel,   indx, patch);
    new_dw->allocateAndPut(enthalpyBoundarySrc, d_lab->d_enthalpyBoundarySrcLabel, indx, patch);
    new_dw->allocateAndPut(umomBoundarySrc,     d_lab->d_umomBoundarySrcLabel,     indx, patch);
    new_dw->allocateAndPut(vmomBoundarySrc,     d_lab->d_vmomBoundarySrcLabel,     indx, patch);
    new_dw->allocateAndPut(wmomBoundarySrc,     d_lab->d_wmomBoundarySrcLabel,     indx, patch);

    scalarBoundarySrc.initialize(0.0);
    enthalpyBoundarySrc.initialize(0.0);
    umomBoundarySrc.initialize(0.0);
    vmomBoundarySrc.initialize(0.0);
    wmomBoundarySrc.initialize(0.0);


    // Variables for mms analysis
    if (d_doMMS){
      //Force terms of convection + diffusion
      // These will be used in the MMS scirun module
      //  to get error convergence
      SFCXVariable<double> uFmms;
      SFCYVariable<double> vFmms;
      SFCZVariable<double> wFmms;

      new_dw->allocateAndPut(uFmms, d_lab->d_uFmmsLabel, indx, patch);
      new_dw->allocateAndPut(vFmms, d_lab->d_vFmmsLabel, indx, patch);
      new_dw->allocateAndPut(wFmms, d_lab->d_wFmmsLabel, indx, patch);

      uFmms.initialize(0.0);
      vFmms.initialize(0.0);
      wFmms.initialize(0.0);
    }
  
    new_dw->allocateAndPut(uVelocityCC, d_lab->d_newCCUVelocityLabel, indx, patch);
    new_dw->allocateAndPut(vVelocityCC, d_lab->d_newCCVVelocityLabel, indx, patch);
    new_dw->allocateAndPut(wVelocityCC, d_lab->d_newCCWVelocityLabel, indx, patch);
    new_dw->allocateAndPut(ccVelocity, d_lab->d_newCCVelocityLabel, indx, patch); 
    uVelocityCC.initialize(0.0);
    vVelocityCC.initialize(0.0);
    wVelocityCC.initialize(0.0);
    ccVelocity.initialize(Vector(0.,0.,0.));
    
    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    new_dw->allocateAndPut(uVelRhoHat, d_lab->d_uVelRhoHatLabel, indx, patch);
    new_dw->allocateAndPut(vVelRhoHat, d_lab->d_vVelRhoHatLabel, indx, patch);
    new_dw->allocateAndPut(wVelRhoHat, d_lab->d_wVelRhoHatLabel, indx, patch);
    uVelRhoHat.initialize(0.0);
    vVelRhoHat.initialize(0.0);
    wVelRhoHat.initialize(0.0);
    
    new_dw->allocateAndPut(pressure, d_lab->d_pressurePSLabel, indx, patch);
    if ((d_extraProjection)||(d_EKTCorrection)) {
      new_dw->allocateAndPut(pressureExtraProjection, d_lab->d_pressureExtraProjectionLabel, indx, patch);
      pressureExtraProjection.initialize(0.0);
    }
    if (!((d_timeIntegratorType == "FE")||(d_timeIntegratorType == "BE"))) {
      new_dw->allocateAndPut(pressurePred, d_lab->d_pressurePredLabel,indx, patch);
      pressurePred.initialize(0.0);
    }
    if (d_timeIntegratorType == "RK3SSP") {
      new_dw->allocateAndPut(pressureInterm, d_lab->d_pressureIntermLabel,indx, patch);
      pressureInterm.initialize(0.0);
    }

    if (d_MAlab) {
      new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel, indx, patch);
      pPlusHydro.initialize(0.0);
      new_dw->allocateAndPut(mmgasVolFrac, d_lab->d_mmgasVolFracLabel, indx, patch);
      mmgasVolFrac.initialize(1.0);
    }
    
    new_dw->allocateAndPut(scalar, d_lab->d_scalarSPLabel, indx, patch);
    
    if (d_calcVariance) {
      new_dw->allocateAndPut(scalarVar_new, d_lab->d_scalarVarSPLabel, indx, patch);
      scalarVar_new.initialize(0.0);
      new_dw->allocateAndPut(normalizedScalarVar_new, d_lab->d_normalizedScalarVarLabel, indx, patch);
      normalizedScalarVar_new.initialize(0.0);
      new_dw->allocateAndPut(scalarDiss_new, d_lab->d_scalarDissSPLabel, indx, patch);
      scalarDiss_new.initialize(0.0);  
    }
    
    CCVariable<double> reactscalar;
    if (d_calcReactingScalar) {
      new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarSPLabel,indx, patch);
      reactscalar.initialize(0.0);
    }

    if (d_calcEnthalpy) {
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
      enthalpy.initialize(0.0);

      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
      CCVariable<double> volq;
      CCVariable<double> abskg;
      CCVariable<double> radEnthalpySrc;;

      new_dw->allocateAndPut(radEnthalpySrc, d_lab->d_radiationSRCINLabel,indx, patch);
      radEnthalpySrc.initialize(0.0);

      new_dw->allocateAndPut(qfluxe, d_lab->d_radiationFluxEINLabel,indx, patch);
      qfluxe.initialize(0.0);
      
      new_dw->allocateAndPut(qfluxw, d_lab->d_radiationFluxWINLabel,indx, patch);
      qfluxw.initialize(0.0);
      
      new_dw->allocateAndPut(qfluxn, d_lab->d_radiationFluxNINLabel,indx, patch);
      qfluxn.initialize(0.0);
      
      new_dw->allocateAndPut(qfluxs, d_lab->d_radiationFluxSINLabel,indx, patch);
      qfluxs.initialize(0.0);
      
      new_dw->allocateAndPut(qfluxt, d_lab->d_radiationFluxTINLabel,indx, patch);
      qfluxt.initialize(0.0);
      
      new_dw->allocateAndPut(qfluxb, d_lab->d_radiationFluxBINLabel,indx, patch);
      qfluxb.initialize(0.0);
     
      new_dw->allocateAndPut(volq, d_lab->d_radiationVolqINLabel,indx, patch);
      volq.initialize(0.0);
 
      new_dw->allocateAndPut(abskg,   d_lab->d_abskgINLabel,        indx, patch);
      abskg.initialize(0.0);

    }
    new_dw->allocateAndPut(density,   d_lab->d_densityCPLabel,    indx, patch);
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, indx, patch);
    if (d_dynScalarModel) {
      if (d_calcScalar){
        new_dw->allocateAndPut(scalarDiffusivity,     d_lab->d_scalarDiffusivityLabel,     indx, patch);
      }
      if (d_calcEnthalpy){
        new_dw->allocateAndPut(enthalpyDiffusivity,   d_lab->d_enthalpyDiffusivityLabel,   indx, patch);
      }
      if (d_calcReactingScalar){
        new_dw->allocateAndPut(reactScalarDiffusivity,d_lab->d_reactScalarDiffusivityLabel,indx, patch);
      }
    }  

    uVelocity.initialize(0.0);
    vVelocity.initialize(0.0);
    wVelocity.initialize(0.0);
    density.initialize(0.0);
    pressure.initialize(0.0);
    double visVal = d_physicalConsts->getMolecularViscosity();
    viscosity.initialize(visVal);

    if (d_dynScalarModel) {
      if (d_calcScalar){
        scalarDiffusivity.initialize(visVal/0.4);
      }
      if (d_calcEnthalpy){
        enthalpyDiffusivity.initialize(visVal/0.4);
      }
      if (d_calcReactingScalar){
        reactScalarDiffusivity.initialize(visVal/0.4);
      }
    }

    scalar.initialize(0.0);
    if (d_init_mix_frac > 0.0) {
      //somewhat ugly...
      InletStream ffState; 
      Stream calculatedStream;
      double init_enthalpy  = 0.0;
      ffState.d_initEnthalpy = true; 
      ffState.d_scalarDisp   = 0.0;
      ffState.d_mixVarVariance.push_back(0.0);
      ffState.d_mixVars.push_back(d_init_mix_frac);

      string bc_type = "scalar_init"; 
      d_props->computeInletProperties( ffState, calculatedStream, bc_type ); 

      init_enthalpy = calculatedStream.getEnthalpy(); 

      for (CellIterator iter=patch->getCellIterator(); 
           !iter.done(); iter++){

        scalar[*iter] = d_init_mix_frac;  

      }
    }

    if (d_calcExtraScalars) {

      for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++) {
        CCVariable<double> extra_scalar;
        new_dw->allocateAndPut(extra_scalar,
                               d_extraScalars[i]->getScalarLabel(), indx, patch);
        extra_scalar.initialize(d_extraScalars[i]->getScalarInitValue());
      }
    }  
  } // patches
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
Arches::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "Arches::computeStableTimeStep",this, 
                           &Arches::computeStableTimeStep);
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;

  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,  gn,  0);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);

  tsk->computes(d_sharedState->get_delt_label(),level.get_rep());
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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> den;
    constCCVariable<double> visc;
    constCCVariable<int> cellType;


    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;
  
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(den, d_lab->d_densityCPLabel,           indx, patch, gac, 1);
    new_dw->get(visc, d_lab->d_viscosityCTSLabel,       indx, patch, gn,  0);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,       indx, patch, gac, 1);
  
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    
    int press_celltypeval = d_boundaryCondition->pressureCellType();
    int out_celltypeval = d_boundaryCondition->outletCellType();
    if ((xminus)&&((cellType[indexLow - IntVector(1,0,0)]==press_celltypeval)
                 ||(cellType[indexLow - IntVector(1,0,0)]==out_celltypeval))){
      indexLow = indexLow - IntVector(1,0,0);
    }
     
    if ((yminus)&&((cellType[indexLow - IntVector(0,1,0)]==press_celltypeval)
                 ||(cellType[indexLow - IntVector(0,1,0)]==out_celltypeval))){
      indexLow = indexLow - IntVector(0,1,0);
    }
    
    if ((zminus)&&((cellType[indexLow - IntVector(0,0,1)]==press_celltypeval)
                 ||(cellType[indexLow - IntVector(0,0,1)]==out_celltypeval))){
      indexLow = indexLow - IntVector(0,0,1);
    }
    
    if (xplus){
      indexHigh = indexHigh + IntVector(1,0,0);
    }
    if (yplus){
      indexHigh = indexHigh + IntVector(0,1,0);
    }
    if (zplus){
      indexHigh = indexHigh + IntVector(0,0,1);
    }

    double delta_t = d_init_dt; // max value allowed
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
            if (flag != 0){
              tmp_time=Abs(uvel)/(cellinfo->sew[colX])+
                Abs(vvel)/(cellinfo->sns[colY])+
                Abs(wvel)/(cellinfo->stb[colZ])+
                (visc[currCell]/den[currCell])* 
                (1.0/(cellinfo->sew[colX]*cellinfo->sew[colX]) +
                 1.0/(cellinfo->sns[colY]*cellinfo->sns[colY]) +
                 1.0/(cellinfo->stb[colZ]*cellinfo->stb[colZ])) +
                small_num;
            }
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
    
    if (d_underflow) {
      indexLow = patch->getFortranCellLowIndex();
      indexHigh = patch->getFortranCellHighIndex();

      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            IntVector yplusCell(colX, colY+1, colZ);
            IntVector zplusCell(colX, colY, colZ+1);
            IntVector xminusCell(colX-1, colY, colZ);
            IntVector yminusCell(colX, colY-1, colZ);
            IntVector zminusCell(colX, colY, colZ-1);
            double tmp_time;

            tmp_time = 0.5* (
            ((den[currCell]+den[xplusCell])*Max(uVelocity[xplusCell],0.0) -
             (den[currCell]+den[xminusCell])*Min(uVelocity[currCell],0.0)) /
            cellinfo->sew[colX] +
            ((den[currCell]+den[yplusCell])*Max(vVelocity[yplusCell],0.0) -
             (den[currCell]+den[yminusCell])*Min(vVelocity[currCell],0.0)) /
            cellinfo->sns[colY] +
            ((den[currCell]+den[zplusCell])*Max(wVelocity[zplusCell],0.0) -
             (den[currCell]+den[zminusCell])*Min(wVelocity[currCell],0.0)) /
            cellinfo->stb[colZ])+small_num;

            if (den[currCell] > 0.0){
              delta_t2=Min(den[currCell]/tmp_time, delta_t2);
            }
          }
        }
      }
    }


    if (d_variableTimeStep) {
      delta_t = delta_t2;
    }
    else {
      proc0cout << " Courant condition for time step: " << delta_t2 << endl;
    }

    //    proc0cout << "time step used: " << delta_t << endl;
    new_dw->put(delt_vartype(delta_t),  d_sharedState->get_delt_label(),getLevel(patches)); 
  }
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
Arches::scheduleTimeAdvance( const LevelP& level, 
                             SchedulerP& sched)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  nofTimeSteps++ ;

  if (d_MAlab) {
#ifndef ExactMPMArchesInitialize
    //    if (nofTimeSteps < 2) {
    if (time < 1.0E-10) {
      proc0cout << "Calculating at time step = " << nofTimeSteps << endl;
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

  if (d_analysisModule) {
    d_analysisModule->scheduleDoAnalysis(sched, level);
  }

  if (d_doingRestart) {

    if (d_newBC_on_Restart) {
      const PatchSet* patches= level->eachPatch();
      const MaterialSet* matls = d_sharedState->allArchesMaterials();

      //Reapply BC in case there was a modification to input.xml in the uda. 
      if (d_boundaryCondition->getInletBC()){
        d_boundaryCondition->sched_calculateArea(sched, patches, matls);
      }
      d_boundaryCondition->sched_setProfile(sched, patches, matls); 
      d_doingRestart = false;
      d_recompile = true; //This will be set to false after the first timestep.
    }
  }
}

// ****************************************************************************
// Function to return boolean for recompiling taskgraph
// ****************************************************************************
bool Arches::needRecompile(double time, double dt, 
                            const GridP& grid) {
  bool temp; 
  if ( d_recompile ) {
   temp = d_recompile;
   proc0cout << endl;
   proc0cout << "NOTICE!:Recompile Taskgraph is set to true.  Taskgraph will be recompiled once first timestep but not after." << endl; 
   proc0cout << endl;
   d_recompile = false; 
   return temp; 
  }
  else 
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
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<double> uVelocityCC;
    CCVariable<double> vVelocityCC;
    CCVariable<double> wVelocityCC;
    CCVariable<double> pressure;
    new_dw->getModifiable(uVelocityCC, d_lab->d_newCCUVelocityLabel, indx, patch);
    new_dw->getModifiable(vVelocityCC, d_lab->d_newCCVVelocityLabel, indx, patch);
    new_dw->getModifiable(wVelocityCC, d_lab->d_newCCWVelocityLabel, indx, patch);
    new_dw->getModifiable(pressure,    d_lab->d_pressurePSLabel,     indx, patch);

    ifstream fd(d_init_inputfile.c_str());
    if(fd.fail()) {
      ostringstream warn;
      warn << "ERROR Arches::readCCInitialCondition: \nUnable to open the given input file " << d_init_inputfile;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    int nx,ny,nz;
    fd >> nx >> ny >> nz;
    const Level* level = patch->getLevel();
    IntVector low, high;
    level->findCellIndexRange(low, high);
    IntVector range = high-low;//-IntVector(2,2,2);
    if (!(range == IntVector(nx,ny,nz))) {
      ostringstream warn;
      warn << "ERROR Arches::readCCInitialCondition: \nWrong grid size in input file " << range;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    double tmp;
    fd >> tmp >> tmp >> tmp;
    fd >> tmp >> tmp >> tmp;
    double uvel,vvel,wvel,pres;
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
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
//___________________________________________________________________________
//



//______________________________________________________________________
//
void 
Arches::sched_blobInit(const LevelP& level,
                       SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "Arches::blobInit",
                     this, &Arches::blobInit);

  tsk->modifies(d_extraScalars[0]->getScalarLabel());
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}
//______________________________________________________________________
//
void
Arches::blobInit(const ProcessorGroup* ,
                 const PatchSubset* patches,
                 const MaterialSubset*,
                 DataWarehouse* ,
                 DataWarehouse* new_dw)
{
  //WARNING: Hardcoded to 1 patch!
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    const VarLabel* extrascalarlabel;
    CCVariable<double> extrascalar;

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    extrascalarlabel = d_extraScalars[0]->getScalarLabel();
    new_dw->getModifiable(extrascalar, extrascalarlabel, indx, patch);
  
    proc0cout << "WARNING!  SETTING UP A BLOB IN YOUR DOMAIN!" << std::endl;
    proc0cout << "Turn off debug_mom in Arches.cc to stop this" << std::endl;
  
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
          
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          
          IntVector currCell(colX,colY,colZ);
                
          if (cellinfo->xx[colX] >= .4 && cellinfo->xx[colX] <= .6){
            if (cellinfo->yy[colY] >= .4 && cellinfo->yy[colY] <= .6){
              if (cellinfo->zz[colZ] >= .4 && cellinfo->zz[colZ] <= .6){
                      
                //for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++) {
                extrascalar[currCell] = .0004513;
                //}
                
              }
            }
          } //end if          
        }
      }
    } //end for
  }  //end patch loop
}


// ****************************************************************************
// schedule reading of initial condition for velocity and pressure
// ****************************************************************************
void 
Arches::sched_mmsInitialCondition(const LevelP& level,
                                  SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "Arches::mmsInitialCondition",
                          this, &Arches::mmsInitialCondition);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  tsk->modifies(d_lab->d_pressurePSLabel);
  //tsk->modifies(d_lab->d_scalarSPLabel);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  if (d_calcExtraScalars){
    for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++){
      tsk->modifies(d_extraScalars[i]->getScalarLabel());
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

// ****************************************************************************
// Actual read
// ****************************************************************************
void
Arches::mmsInitialCondition(const ProcessorGroup* ,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* ,
                            DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    CCVariable<double> pressure;
    //CCVariable<double> scalar;
    
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(pressure,  d_lab->d_pressurePSLabel,    indx, patch);
    //new_dw->getModifiable(scalar,    d_lab->d_scalarSPLabel,      indx, patch);
   
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    double pi = acos(-1.0);

    //CELL centered variables
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector currCell = *iter; 
    
      if (d_mms == "constantMMS") { 
        pressure[*iter] = d_cp;
        //scalar[*iter]   = d_phi0;
        if (d_calcExtraScalars) {
      
          for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++) {
           CCVariable<double> extra_scalar;
           new_dw->allocateAndPut(extra_scalar,
                                  d_extraScalars[i]->getScalarLabel(),indx, patch);
           extra_scalar.initialize(d_esphi0);
         }
        }
      } else if (d_mms == "almgrenMMS") {         
        pressure[*iter] = -d_amp*d_amp/4 * (cos(4.0*pi*cellinfo->xx[currCell.x()])
                          + cos(4.0*pi*cellinfo->yy[currCell.y()]));
        //scalar[*iter]   = 0.0;
      }
    }

    //X-FACE centered variables 
    for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
      IntVector currCell = *iter; 

      if (d_mms == "constantMMS") { 
        uVelocity[*iter] = d_cu; 
      } else if (d_mms == "almgrenMMS") { 
        // for mms in x-y plane
        uVelocity[*iter] = 1 - d_amp * cos(2.0*pi*cellinfo->xu[currCell.x()])
                           * sin(2.0*pi*cellinfo->yy[currCell.y()]);       
      }
    }

    //Y-FACE centered variables 
    for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){
      IntVector currCell = *iter; 

      if (d_mms == "constantMMS") { 
        vVelocity[*iter] = d_cv; 
      } else if (d_mms == "almgrenMMS") { 
        // for mms in x-y plane
        vVelocity[*iter] = 1 + d_amp * sin(2.0*pi*cellinfo->xx[currCell.x()])
                              * cos(2.0*pi*cellinfo->yv[currCell.y()]); 
        
      }
    }

    //Z-FACE centered variables 
    for (CellIterator iter=patch->getSFCZIterator(); !iter.done(); iter++){

      if (d_mms == "constantMMS") { 
        wVelocity[*iter] = d_cw; 
      } else if (d_mms == "almgrenMMS") { 
        // for mms in x-y plane
        wVelocity[*iter] =  0.0;
      }
    }

    // Previously, we had the boundaries initialized here (below this comment).  I have removed
    // this since a) it seemed incorrect and b) because it would fit better
    // where BC's were applied.  Note that b) implies that we have a better
    // BC abstraction.
    // -Jeremy

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
  Ghost::GhostType  gac = Ghost::AroundCells;   
                      
  tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel, gac, 1);
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
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<double> uVelocityCC;
    constCCVariable<double> vVelocityCC;
    constCCVariable<double> wVelocityCC;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(uVelocityCC, d_lab->d_newCCUVelocityLabel, indx, patch, gac, 1);
    new_dw->get(vVelocityCC, d_lab->d_newCCVVelocityLabel, indx, patch, gac, 1);
    new_dw->get(wVelocityCC, d_lab->d_newCCWVelocityLabel, indx, patch, gac, 1);
    
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    
    IntVector idxLo, idxHi;

    for(CellIterator iter = patch->getSFCXIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      IntVector L = c - IntVector(1,0,0);
      uVelocity[c] = 0.5 * (uVelocityCC[c] + uVelocityCC[L]);
    }
    
    for(CellIterator iter = patch->getSFCYIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      IntVector L = c - IntVector(0,1,0);
      vVelocity[c] = 0.5 * (vVelocityCC[c] + vVelocityCC[L]);
    }
    
    for(CellIterator iter = patch->getSFCXIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      IntVector L = c - IntVector(0,0,1);
      wVelocity[c] = 0.5 * (wVelocityCC[c] + wVelocityCC[L]);
    }
  }
}
// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC
// ****************************************************************************
void 
Arches::sched_getCCVelocities(const LevelP& level, SchedulerP& sched)
{
  Task* tsk = scinew Task("Arches::getCCVelocities", this, 
                          &Arches::getCCVelocities);
                          
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  tsk->modifies(d_lab->d_newCCUVelocityLabel);
  tsk->modifies(d_lab->d_newCCVVelocityLabel);
  tsk->modifies(d_lab->d_newCCWVelocityLabel);
  tsk->modifies(d_lab->d_newCCVelocityLabel); 
      
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

// ****************************************************************************
// interpolation from FC to CC Variable
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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;
    CCVariable<double> uvel_CC;
    CCVariable<double> vvel_CC;
    CCVariable<double> wvel_CC;
    CCVariable<Vector> vel_CC; 

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
      
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uvel_FC, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(vvel_FC, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(wvel_FC, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    
    new_dw->getModifiable(uvel_CC, d_lab->d_newCCUVelocityLabel,indx, patch);
    new_dw->getModifiable(vvel_CC, d_lab->d_newCCVVelocityLabel,indx, patch);
    new_dw->getModifiable(wvel_CC, d_lab->d_newCCWVelocityLabel,indx, patch);
    new_dw->getModifiable(vel_CC,  d_lab->d_newCCVelocityLabel, indx, patch); 
    uvel_CC.initialize(0.0);
    vvel_CC.initialize(0.0);
    wvel_CC.initialize(0.0);
    vel_CC.initialize(Vector(0.0,0.0,0.0)); 
   
    //__________________________________
    //  
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      int i = c.x();
      int j = c.y();
      int k = c.z();

      IntVector idxU(i+1,j,k);
      IntVector idxV(i,j+1,k);
      IntVector idxW(i,j,k+1);

      uvel_CC[c] = cellinfo->wfac[i] * uvel_FC[c] +
                   cellinfo->efac[i] * uvel_FC[idxU];
                     
      vvel_CC[c] = cellinfo->sfac[j] * vvel_FC[c] +
                   cellinfo->nfac[j] * vvel_FC[idxV];
                     
      wvel_CC[c] = cellinfo->bfac[k] * wvel_FC[c] +
                   cellinfo->tfac[k] * wvel_FC[idxW];

      vel_CC[c] = Vector(uvel_CC[c], vvel_CC[c], wvel_CC[c]); 
    }
    //__________________________________
    // Apply boundary conditions
    vector<Patch::FaceType> b_face;
    patch->getBoundaryFaces(b_face);
    vector<Patch::FaceType>::const_iterator itr;
    
    // Loop over boundary faces
    for( itr = b_face.begin(); itr != b_face.end(); ++itr ){
      Patch::FaceType face = *itr;
 
      IntVector f_dir = patch->getFaceDirection(face); 

      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
      CellIterator iter=patch->getFaceIterator(face, MEC);
      
      IntVector lo = iter.begin();
      int i = lo.x();
      int j = lo.y();
      int k = lo.z();
      
      Vector one_or_zero = Vector(1,1,1) - Abs(f_dir.asVector());
      // one_or_zero: faces x-+   (0,1,1)
      //                    y-+   (1,0,1)
      //                    z-+   (1,1,0)
        
      for(;!iter.done();iter++){                                 
        IntVector c = *iter;                                     
                                                                 
        IntVector idxU(i+1,j,k);                                 
        IntVector idxV(i,j+1,k);                                 
        IntVector idxW(i,j,k+1);                                 
                                                                 
        uvel_CC[c] = one_or_zero.x() *                         
                      (cellinfo->wfac[i] * uvel_FC[c] +          
                       cellinfo->efac[i] * uvel_FC[idxU]) +      
                      (1.0 - one_or_zero.x()) * uvel_FC[idxU];  
                                                                 
        vvel_CC[c] = one_or_zero.y() *                         
                       (cellinfo->sfac[j] * vvel_FC[c] +         
                        cellinfo->nfac[j] * vvel_FC[idxV]) +     
                       (1.0 - one_or_zero.y()) * vvel_FC[idxV];  
                                                                 
        wvel_CC[c] = one_or_zero.z() *                         
                      (cellinfo->bfac[k] * wvel_FC[c] +          
                       cellinfo->tfac[k] * wvel_FC[idxW] ) +     
                      (1.0 - one_or_zero.z()) * wvel_FC[idxW];   

        vel_CC[c] = Vector( uvel_CC[c], vvel_CC[c], wvel_CC[c] ); 
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

