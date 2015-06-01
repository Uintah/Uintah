/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- Arches.cc ----------------------------------------------
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <CCA/Components/Arches/DQMOM.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/IntrusionBC.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/CoalModels/ConstantModel.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/CoalModels/RichardsFletcherDevol.h>
#include <CCA/Components/Arches/CoalModels/FOWYDevol.h>
#include <CCA/Components/Arches/CoalModels/SimpleBirth.h>
#include <CCA/Components/Arches/CoalModels/YamamotoDevol.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/EnthalpyShaddix.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
//NOTE: new includes for CQMOM
#include <CCA/Components/Arches/CQMOM.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>

#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/PropertyModels/ConstProperty.h>
#include <CCA/Components/Arches/PropertyModels/ExtentRxn.h>
#include <CCA/Components/Arches/PropertyModels/TabStripFactor.h>
#include <CCA/Components/Arches/PropertyModels/EmpSoot.h>
#include <CCA/Components/Arches/PropertyModels/AlgebraicScalarDiss.h>
#include <CCA/Components/Arches/PropertyModels/HeatLoss.h>
#include <CCA/Components/Arches/PropertyModels/ScalarVarianceScaleSim.h>
#include <CCA/Components/Arches/PropertyModels/NormScalarVariance.h>
#include <CCA/Components/Arches/PropertyModels/ScalarDissipation.h>
#include <CCA/Components/Arches/PropertyModels/RadProperties.h>
#include <Core/IO/UintahZlibUtil.h>

#if HAVE_TABPROPS
#  include <CCA/Components/Arches/ChemMix/TabPropsInterface.h>
#endif

//NEW TASK INTERFACE STUFF
//factories
#include <CCA/Components/Arches/Utility/UtilityFactory.h>
#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/LagrangianParticles/LagrangianParticleFactory.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
//#include <CCA/Components/Arches/Task/SampleFactory.h>


//END NEW TASK INTERFACE STUFF


#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ExplicitSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/SmagorinskyModel.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <CCA/Components/Arches/TurbulenceModelPlaceholder.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/IncDynamicProcedure.h>
#include <CCA/Components/Arches/CompDynamicProcedure.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/Parallel.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/MemoryWindow.h>

#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <fstream>

using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::ostringstream;
using std::make_pair;
using std::cout;

using namespace Uintah;

static DebugStream dbg("ARCHES", false);

// Used to sync std::cout when output by multiple threads
extern SCIRun::Mutex coutLock;

const int Arches::NDIM = 3;

// ****************************************************************************
// Actual constructor for Arches
// ****************************************************************************
Arches::Arches(const ProcessorGroup* myworld, const bool doAMR) :
  UintahParallelComponent(myworld)
{
  d_lab =  scinew  ArchesLabel();
  d_MAlab                 =  0;      //will  be  set  by  setMPMArchesLabel
  d_props                 =  0;
  d_turbModel             =  0;
  d_scaleSimilarityModel  =  0;
  d_boundaryCondition     =  0;
  d_nlSolver              =  0;
  d_physicalConsts        =  0;
  d_doingRestart          =  false;

  nofTimeSteps                     =  0;
  init_timelabel_allocated         =  false;
  DQMOMEqnFactory&  dqmomfactory   =  DQMOMEqnFactory::self();
  dqmomfactory.set_quad_nodes(0);
  d_doDQMOM                        =  false;
  d_with_mpmarches                 =  false;
  d_do_dummy_solve                 =  false; 
  d_doAMR                          = doAMR;
  
  CQMOMEqnFactory& cqmomfactory    = CQMOMEqnFactory::self();
  cqmomfactory.set_number_moments(0);
  d_doCQMOM                        = false;

  //lagrangian particles: 
  _particlesHelper = scinew ArchesParticlesHelper(); 
  _particlesHelper->sync_with_arches(this); 

}

// ****************************************************************************
// Destructor
// ****************************************************************************
Arches::~Arches()
{
  delete d_lab;
  delete d_props;
  delete d_turbModel;
  delete d_scaleSimilarityModel;
  delete d_boundaryCondition;
  delete d_nlSolver;
  delete d_physicalConsts;

  Operators& opr = Operators::self();
  opr.delete_patch_set(); 

  if (init_timelabel_allocated)
    delete init_timelabel;

  if(d_analysisModules.size() != 0){
    std::vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      delete *iter;
    }
  }

  delete d_timeIntegrator;
  delete d_rad_prop_calc; 
  if (d_doDQMOM) {
    delete d_dqmomSolver;
    delete d_partVel;
  }
  
  if (d_doCQMOM) {
    delete d_cqmomSolver;
  }
  releasePort("solver");

  delete _particlesHelper; 

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
  _arches_spec = db; 
  d_lab->problemSetup( db );

  //Look for coal information 
  CoalHelper& coal_helper = CoalHelper::self(); 
  coal_helper.parse_for_coal_info( db ); 

  //==============NEW TASK STUFF
  //build the factories
  boost::shared_ptr<UtilityFactory> UtilF(scinew UtilityFactory()); 
  boost::shared_ptr<TransportFactory> TransF(scinew TransportFactory()); 
  boost::shared_ptr<InitializeFactory> InitF(scinew InitializeFactory()); 
  boost::shared_ptr<ParticleModelFactory> PartModF(scinew ParticleModelFactory()); 
  boost::shared_ptr<LagrangianParticleFactory> LagF(scinew LagrangianParticleFactory()); 
  boost::shared_ptr<PropertyModelFactoryV2> PropModels(scinew PropertyModelFactoryV2(sharedState)); 

  _boost_factory_map.clear(); 
  _boost_factory_map.insert(std::make_pair("utility_factory",UtilF)); 
  _boost_factory_map.insert(std::make_pair("transport_factory",TransF)); 
  _boost_factory_map.insert(std::make_pair("initialize_factory",InitF)); 
  _boost_factory_map.insert(std::make_pair("particle_model_factory",PartModF)); 
  _boost_factory_map.insert(std::make_pair("lagrangian_factory",LagF)); 
  _boost_factory_map.insert(std::make_pair("property_models_factory", PropModels)); 

  //==================== NEW STUFF ===============================
  typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
  proc0cout << "\n Registering Tasks For: " << std::endl;
  for ( BFM::iterator i = _boost_factory_map.begin(); i != _boost_factory_map.end(); i++ ){ 

    proc0cout << "   " << i->first << std::endl;
    i->second->register_all_tasks(db); 

  }
  proc0cout << "\n Building Tasks For: " << std::endl;
  for ( BFM::iterator i = _boost_factory_map.begin(); i != _boost_factory_map.end(); i++ ){ 

    proc0cout << "   " << i->first << std::endl;
    i->second->build_all_tasks(db); 

  }
  proc0cout << endl;

  //Checking for lagrangian particles:
  _doLagrangianParticles = _arches_spec->findBlock("LagrangianParticles"); 
  if ( _doLagrangianParticles ){ 
    _particlesHelper->problem_setup(params,_arches_spec->findBlock("LagrangianParticles"), sharedState);
  }

  //__________________________________
  //  Multi-level related
  d_archesLevelIndex = grid->numLevels()-1; // this is the finest level
  proc0cout << "ARCHES CFD level: " << d_archesLevelIndex << endl;

  // This will allow for changing the BC's on restart:
  if ( db->findBlock("new_BC_on_restart") )
    d_newBC_on_Restart = true;
  else
    d_newBC_on_Restart = false;

  db->getWithDefault("turnonMixedModel",    d_mixedModel,false);
  db->getWithDefault("recompileTaskgraph",  d_lab->recompile_taskgraph,false);

  string nlSolver;
  if (db->findBlock("ExplicitSolver")){
    nlSolver = "explicit";
    db->findBlock("ExplicitSolver")->getWithDefault("extraProjection",     d_extraProjection,false);
    d_underflow = false; 
    if ( db->findBlock("ExplicitSolver")->findBlock("scalarUnderflowCheck") ) d_underflow = true; 
    db->findBlock("ExplicitSolver")->getWithDefault("initial_dt", d_initial_dt, 1.0);

  }

  // physical constant
  d_physicalConsts = scinew PhysicalConstants();
  const ProblemSpecP db_root = db->getRootNode();
  d_physicalConsts->problemSetup(db_root);

  //create a time integrator.
  d_timeIntegrator = scinew ExplicitTimeInt(d_lab);
  ProblemSpecP time_db = db->findBlock("TimeIntegrator");
  if (time_db) {
    string time_order;
    time_db->findBlock("ExplicitIntegrator")->getAttribute("order", time_order);
    if (time_order == "first")
      d_tOrder = 1;
    else if (time_order == "second")
      d_tOrder = 2;
    else if (time_order == "third")
      d_tOrder = 3;
    else
      throw InvalidValue("Explicit time integrator must be one of: first, second, third!  Please fix input file",__FILE__,__LINE__);

    d_timeIntegrator->problemSetup(time_db);
  }

//------------------------------------------------------------------------------
//

  //Transport Eqns: 
  ProblemSpecP transportEqn_db = db->findBlock("TransportEqns");
  if (transportEqn_db) {

    // register source terms
    SourceTermFactory& src_factory = SourceTermFactory::self();
    ProblemSpecP sources_db = transportEqn_db->findBlock("Sources");
    if (sources_db)
      src_factory.registerUDSources(sources_db, d_lab, d_boundaryCondition, d_myworld);

    //register all equations
    Arches::registerTransportEqns(transportEqn_db);

    // Go through eqns and intialize all defined eqns and call their respective
    // problem setup
    EqnFactory& eqn_factory = EqnFactory::self();
    for (ProblemSpecP eqn_db = transportEqn_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){

      std::string eqnname;
      eqn_db->getAttribute("label", eqnname);
      d_scalarEqnNames.push_back(eqnname);
      if (eqnname == ""){
        throw InvalidValue( "Error: The label attribute must be specified for the eqns!", __FILE__, __LINE__);
      }
      EqnBase& an_eqn = eqn_factory.retrieve_scalar_eqn( eqnname );
      an_eqn.problemSetup( eqn_db );

    }


    // Now go through sources and initialize all defined sources and call
    // their respective problemSetup
    if (sources_db) {

      vector<string> used_sources; 

      SourceTermFactory& src_factory = SourceTermFactory::self();

      for (ProblemSpecP eqn_db = transportEqn_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){

        for (ProblemSpecP src_db = eqn_db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){ 

          std::string srcname;
          src_db->getAttribute("label", srcname);

          for ( ProblemSpecP found_src_db = transportEqn_db->findBlock("Sources")->findBlock("src"); found_src_db != 0; 
              found_src_db = found_src_db->findNextBlock("src")){

            string check_label; 
            found_src_db->getAttribute("label",check_label);

            if ( check_label == srcname ){

              vector<string>::iterator it = find( used_sources.begin(), used_sources.end(), srcname);

              if ( it == used_sources.end() ){
                used_sources.push_back( srcname );

                SourceTermBase& a_src = src_factory.retrieve_source_term( srcname );
                a_src.problemSetup( found_src_db );

                //Add any table lookup species to the table lookup list:                                      
                ChemHelper::TableLookup*  tbl_lookup = a_src.get_tablelookup_species();                        
                if ( tbl_lookup != NULL )
                  d_lab->add_species_struct( tbl_lookup ); 

              }
            }
          }
        }
      }
    }

  } else {

    proc0cout << "No defined transport equations found." << endl;

  }

  if ( db->findBlock("PropertyModels") ){

    ProblemSpecP propmodels_db = db->findBlock("PropertyModels");
    PropertyModelFactory& prop_factory = PropertyModelFactory::self();
    Arches::registerPropertyModels( propmodels_db );
    for ( ProblemSpecP prop_db = propmodels_db->findBlock("model");
        prop_db != 0; prop_db = prop_db->findNextBlock("model") ){

      std::string model_name;
      prop_db->getAttribute("label", model_name);
      if ( model_name == "" ){
        throw InvalidValue( "Error: The label attribute must be specified for the property models!", __FILE__, __LINE__);
      }
      PropertyModelBase& a_model = prop_factory.retrieve_property_model( model_name );
      a_model.problemSetup( prop_db );

    }
  }

  //Radiation Properties: 
  ProblemSpecP rad_properties_db = db->findBlock("RadiationProperties"); 
  int matl_index = d_sharedState->getArchesMaterial(0)->getDWIndex();
  d_rad_prop_calc = scinew RadPropertyCalculator(matl_index); 
  if ( rad_properties_db ){
    d_rad_prop_calc->problemSetup(rad_properties_db); 
  }

  // read properties
  // d_MAlab = multimaterial arches common labels
  d_props = scinew Properties(d_lab, d_MAlab, d_physicalConsts, d_myworld);

  d_props->problemSetup(db);

  //need to set bounds on heat loss as the values in the table itself
  PropertyModelFactory& propFactory = PropertyModelFactory::self();
  PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
  for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
      iprop != all_prop_models.end(); iprop++){

    PropertyModelBase* prop_model = iprop->second;
    if ( prop_model->getPropType() == "heat_loss" ){ 
      MixingRxnModel* mixing_table = d_props->getMixRxnModel();
      std::map<string,double> table_constants = mixing_table->getAllConstants(); 
      if (d_props->getMixingModelType() == "ClassicTable" ) {

        ClassicTableInterface* classic_table = dynamic_cast<ClassicTableInterface*>(mixing_table); 
        std::vector<double> hl_bounds;
        hl_bounds = classic_table->get_hl_bounds(); 

        HeatLoss* hl_prop_model = dynamic_cast<HeatLoss*>(prop_model); 
        hl_prop_model->set_hl_bounds(hl_bounds); 
        hl_prop_model->set_table_ref(classic_table); 

      }
    } 
  }

  EqnFactory& eqn_factory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqn_factory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); 
      ieqn != scalar_eqns.end(); ieqn++){

    EqnBase* eqn = ieqn->second;  
    eqn->assign_stage_to_sources(); 

  }

  // read boundary condition information
  d_boundaryCondition = scinew BoundaryCondition(d_lab, d_MAlab, d_physicalConsts,
                                                 d_props );

  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->problemSetup(db);

  d_whichTurbModel = "none"; 
  
  if ( db->findBlock("Turbulence") ) {
    db->findBlock("Turbulence")->getAttribute("model",d_whichTurbModel); 
  }

  if ( d_whichTurbModel == "smagorinsky"){
    d_turbModel = scinew SmagorinskyModel(d_lab, d_MAlab, d_physicalConsts,
                                          d_boundaryCondition);
  }else  if ( d_whichTurbModel == "dynamicprocedure"){
    d_turbModel = scinew IncDynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
                                          d_boundaryCondition);
  }else if ( d_whichTurbModel == "compdynamicprocedure"){
    d_turbModel = scinew CompDynamicProcedure(d_lab, d_MAlab, d_physicalConsts,
                                          d_boundaryCondition);
  } else if ( d_whichTurbModel == "none" ){ 
    proc0cout << "\n Notice: Turbulence model specificied as: none. Running without momentum closure. \n";
    d_turbModel = scinew TurbulenceModelPlaceholder(d_lab, d_MAlab, d_physicalConsts,
                                                    d_boundaryCondition);
  } else {
    proc0cout << "\n Notice: No Turbulence model found. \n" << endl;
  }

  d_turbModel->problemSetup(db);
    
  d_turbModel->setMixedModel(d_mixedModel);
  if (d_mixedModel) {
    d_scaleSimilarityModel=scinew ScaleSimilarityModel(d_lab, d_MAlab, d_physicalConsts,
                                                       d_boundaryCondition);
    d_scaleSimilarityModel->problemSetup(db);

    d_scaleSimilarityModel->setMixedModel(d_mixedModel);
  }

  d_props->setBC(d_boundaryCondition);

  //__________________________________
  SolverInterface* hypreSolver = dynamic_cast<SolverInterface*>(getPort("solver"));

  if(!hypreSolver) {
    throw InternalError("ARCHES:couldn't get hypreSolver port", __FILE__, __LINE__);
  }

  if (nlSolver == "explicit") {
    d_nlSolver = scinew ExplicitSolver(d_lab, d_MAlab, d_props,
                                       d_boundaryCondition,
                                       d_turbModel, d_scaleSimilarityModel,
                                       d_physicalConsts, 
                                       d_rad_prop_calc, 
                                       _boost_factory_map,
                                       d_myworld,
                                       hypreSolver);

  }
  else{
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver, __FILE__, __LINE__);
  }
  d_nlSolver->problemSetup(db,sharedState);
  d_timeIntegratorType = d_nlSolver->getTimeIntegratorType();
  //__________________
  //This is not the proper way to get our DA.  Scheduler should
  //pass us a DW pointer on every function call.  I don't think
  //AnalysisModule should retain the pointer in a field, IMHO.
  if(!d_with_mpmarches){
    Output* dataArchiver = dynamic_cast<Output*>(getPort("output"));
    if(!dataArchiver){
      throw InternalError("ARCHES:couldn't get output port", __FILE__, __LINE__);
    }

    d_analysisModules = AnalysisModuleFactory::create(params, sharedState, dataArchiver);

    if(d_analysisModules.size() != 0){
      vector<AnalysisModule*>::iterator iter;
      for( iter  = d_analysisModules.begin();
           iter != d_analysisModules.end(); iter++){
        AnalysisModule* am = *iter;
        am->problemSetup(params, materials_ps, grid, sharedState);
      }
    }
  }

  // ----- DQMOM STUFF:


  ProblemSpecP dqmom_db = db->findBlock("DQMOM");
  if (dqmom_db) {

    //turn on DQMOM
    d_doDQMOM = true;

    // require that we have weighted or unweighted explicitly specified as an attribute to DQMOM
    // type = "unweightedAbs" or type = "weighedAbs"
    dqmom_db->getAttribute( "type", d_which_dqmom );

    ProblemSpecP db_linear_solver = dqmom_db->findBlock("LinearSolver");
    if( db_linear_solver ) {
      string d_solverType;
      db_linear_solver->getWithDefault("type", d_solverType, "LU");

      // currently, unweighted abscissas only work with the optimized solver -- remove this check when other solvers work:
      if( d_which_dqmom == "unweightedAbs" && d_solverType != "Optimize" ) {
        throw ProblemSetupException("Error!: The unweighted abscissas only work with the optimized solver.", __FILE__, __LINE__);
      }
    }

    DQMOMEqnFactory& eqn_factory = DQMOMEqnFactory::self();

    //register all equations.
    eqn_factory.registerDQMOMEqns(dqmom_db, d_lab, d_timeIntegrator );

    //register all models
    CoalModelFactory& model_factory = CoalModelFactory::self();
    model_factory.problemSetup(dqmom_db);

    Arches::registerModels(dqmom_db);

    // Create a velocity model
    d_partVel = scinew PartVel( d_lab );
    d_partVel->problemSetup( dqmom_db );
    d_nlSolver->setPartVel( d_partVel );
    // Do through and initialze all DQMOM equations and call their respective problem setups.
    const int numQuadNodes = eqn_factory.get_quad_nodes();

    model_factory.setArchesLabel( d_lab );

    ProblemSpecP w_db = dqmom_db->findBlock("Weights");

    // do all weights
    for (int iqn = 0; iqn < numQuadNodes; iqn++){
      std::string weight_name = "w_qn";
      std::string node;
      std::stringstream out;
      out << iqn;
      node = out.str();
      weight_name += node;

      EqnBase& a_weight = eqn_factory.retrieve_scalar_eqn( weight_name );
      eqn_factory.set_weight_eqn( weight_name, &a_weight );
      DQMOMEqn& weight = dynamic_cast<DQMOMEqn&>(a_weight);
      weight.setAsWeight();
      weight.problemSetup( w_db );
    }

    // loop for all ic's
    for (ProblemSpecP ic_db = dqmom_db->findBlock("Ic"); ic_db != 0; ic_db = ic_db->findNextBlock("Ic")){
      std::string ic_name;
      ic_db->getAttribute("label", ic_name);
      //loop for all quad nodes for this internal coordinate
      for (int iqn = 0; iqn < numQuadNodes; iqn++){

        std::string final_name = ic_name + "_qn";
        std::string node;
        std::stringstream out;
        out << iqn;
        node = out.str();
        final_name += node;

        EqnBase& an_ic = eqn_factory.retrieve_scalar_eqn( final_name );
        eqn_factory.set_abscissa_eqn( final_name, &an_ic );
        an_ic.problemSetup( ic_db );

      }
    }

    // Now go through models and initialize all defined models and call
    // their respective problemSetup
    ProblemSpecP models_db = dqmom_db->findBlock("Models");
    if (models_db) {
      for (ProblemSpecP m_db = models_db->findBlock("model"); m_db != 0; m_db = m_db->findNextBlock("model")){
        std::string model_name;
        m_db->getAttribute("label", model_name);
        for (int iqn = 0; iqn < numQuadNodes; iqn++){
          std::string temp_model_name = model_name;
          std::string node;
          std::stringstream out;
          out << iqn;
          node = out.str();
          temp_model_name += "_qn";
          temp_model_name += node;

          ModelBase& a_model = model_factory.retrieve_model( temp_model_name );
          a_model.problemSetup( m_db, iqn );
        }
      }
    }

    // set up the linear solver:
    d_dqmomSolver = scinew DQMOM( d_lab, d_which_dqmom );
    d_dqmomSolver->problemSetup( dqmom_db );

    // now pass it off to the nonlinear solver:
    d_nlSolver->setDQMOMSolver( d_dqmomSolver );

  }

  
  // ----- CQMOM STUFF:

  ProblemSpecP cqmom_db = db->findBlock("CQMOM");
  if (cqmom_db) {
    d_doCQMOM = true;
    // require that we have weighted or unweighted explicitly specified as an attribute to CQMOM
    cqmom_db->getAttribute( "type", d_which_cqmom );
   
    //register all equations.
    Arches::registerCQMOMEqns(cqmom_db);
    
    //register all models/srcs
//    CoalModelFactory& model_factory = CoalModelFactory::self();
//    model_factory.problemSetup(cqmom_db);
//    Arches::registerModels(cqmom_db);
  
    // initialze all CQMOM equations and call their respective problem setups.
    CQMOMEqnFactory& eqn_factory = CQMOMEqnFactory::self();
    const int numMoments = eqn_factory.get_number_moments();
    proc0cout << "Feeding these " << numMoments << " eqns into CQMOM Eqn Factory" << endl;
//    model_factory.setArchesLabel( d_lab );
    
    int M;
    cqmom_db->get("NumberInternalCoordinates",M);
    vector<int> temp_moment_index;
    for ( ProblemSpecP db_moments = cqmom_db->findBlock("Moment");
         db_moments != 0; db_moments = db_moments->findNextBlock("Moment") ) {
      temp_moment_index.resize(0);
      db_moments->get("m", temp_moment_index);
      proc0cout << "Index " << temp_moment_index << " ";
      int index_length = temp_moment_index.size();
      if (index_length != M) {
        std::cout << "Index for moment " << temp_moment_index << " does not have same number of indexes as internal coordinate #" << M << std::endl;
      }
      
      std::string moment_name = "m_";
      std::string mIndex;
      std::stringstream out;
      for (int i = 0; i<M ; i++) {
        out << temp_moment_index[i];
        mIndex = out.str();
      }
      moment_name += mIndex;
      
      EqnBase& a_moment = eqn_factory.retrieve_scalar_eqn( moment_name );
      eqn_factory.set_moment_eqn( moment_name, &a_moment );
      CQMOMEqn& moment = dynamic_cast<CQMOMEqn&>(a_moment);
      moment.problemSetup( db_moments );
    }
    
    // Now go through models and initialize all defined models and call
    // their respective problemSetup
//    ProblemSpecP models_db = cqmom_db->findBlock("Models");
//    if (models_db) {
//      for (ProblemSpecP m_db = models_db->findBlock("model"); m_db != 0; m_db = m_db->findNextBlock("model")){
//        std::string model_name;
//        m_db->getAttribute("label", model_name);
//        for (int i = 0; i < nMoments; i++){
//        }
//      }
//    }
    
    // set up the linear solver:
    d_cqmomSolver = scinew CQMOM( d_lab, d_which_cqmom );
    d_cqmomSolver->problemSetup( cqmom_db );
    
    //pass it to the explicit solver
    d_nlSolver->setCQMOMSolver( d_cqmomSolver );
  }

  
  // register any other source terms:
  SourceTermFactory& src_factory = SourceTermFactory::self();
  src_factory.registerSources( d_lab, d_doDQMOM, d_which_dqmom );


  // do any last setup operations on the active source terms: 
  src_factory.extraSetup( grid, d_boundaryCondition ); 

  // Add extra species to table lookup as required by models
  d_props->addLookupSpecies();

  // Add new intrusion stuff:
  // get a reference to the intrusions
  IntrusionBC* intrusion_ref = d_boundaryCondition->get_intrusion_ref();
  bool using_new_intrusions = d_boundaryCondition->is_using_new_intrusion();

  if(d_doDQMOM)
  {
    // check to make sure that all dqmom equations have BCs set.
    DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns();
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
      EqnBase* eqn = ieqn->second;
      eqn->set_intrusion( intrusion_ref );
      eqn->set_intrusion_bool( using_new_intrusions );
    }
  }

  if(d_doCQMOM) //just copied from dqmom BC lock, this should work the same
  {
    // check to make sure that all cqmom equations have BCs set.
    CQMOMEqnFactory& cqmom_factory = CQMOMEqnFactory::self();
    CQMOMEqnFactory::EqnMap& cqmom_eqns = cqmom_factory.retrieve_all_eqns();
    for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin(); ieqn != cqmom_eqns.end(); ieqn++){
      EqnBase* eqn = ieqn->second;
      eqn->set_intrusion( intrusion_ref );
      eqn->set_intrusion_bool( using_new_intrusions );
    }
  }
  
  // check to make sure that all the scalar variables have BCs set and set intrusions:
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
    EqnBase* eqn = ieqn->second;
    eqn->set_intrusion( intrusion_ref );
    eqn->set_intrusion_bool( using_new_intrusions );

    //send a reference of the mixing/rxn table to the eqn for initializiation
    MixingRxnModel* d_mixingTable = d_props->getMixRxnModel();
    eqn->set_table( d_mixingTable ); 

    //look for an set any tabulated bc's
    eqn->extraProblemSetup( db ); 
  }
  
  if(d_doAMR && !sharedState->isLockstepAMR()){
    ostringstream msg;
    msg << "\n ERROR: You must add \n"
        << " <useLockStep> true </useLockStep> \n"
        << " inside of the <AMR> section for multi-level ARCHES & MPMARCHES. \n"; 
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void
Arches::scheduleInitialize(const LevelP& level,
                           SchedulerP& sched)
{
 if( level->getIndex() != d_archesLevelIndex )
  return;

  const MaterialSet* matls = d_sharedState->allArchesMaterials();

  Operators& opr = Operators::self(); 
  opr.set_my_world( d_myworld ); 
  opr.create_patch_operators( level, sched, matls ); 

  if ( _doLagrangianParticles ){ 
    _particlesHelper->set_materials(d_sharedState->allArchesMaterials()); 
    _particlesHelper->schedule_initialize(level, sched);
  }

  //Initialize several parameters
  sched_paramInit(level, sched);

  //Check for hand-off momentum BCs and perform mapping
  d_nlSolver->checkMomBCs( sched, level, matls ); 

  //initialize cell type
  d_boundaryCondition->sched_cellTypeInit( sched, level, matls );

  // compute the cell area fraction
  d_boundaryCondition->sched_setAreaFraction( sched, level, matls, 0, true );

  // setup intrusion cell type 
  d_boundaryCondition->sched_setupNewIntrusionCellType( sched, level, matls, false );

  //AF must be called again to account for intrusions (can this be the ONLY call?) 
  d_boundaryCondition->sched_setAreaFraction( sched, level, matls, 1, true ); 

  d_turbModel->sched_computeFilterVol( sched, level, matls ); 

  //Particle models are initialized after DQMOM/CQMOM initialization 

  //=========== NEW TASK INTERFACE ==============================
  typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
  BFM::iterator i_util_fac = _boost_factory_map.find("utility_factory"); 
  BFM::iterator i_trans_fac = _boost_factory_map.find("transport_factory"); 
  BFM::iterator i_init_fac = _boost_factory_map.find("initialize_factory"); 
  BFM::iterator i_partmod_fac = _boost_factory_map.find("particle_model_factory"); 
  BFM::iterator i_lag_fac = _boost_factory_map.find("lagrangian_factory"); 
  BFM::iterator i_property_models_fac = _boost_factory_map.find("property_models_factory"); 

  bool is_restart = false; 
  //utility factory
  TaskFactoryBase::TaskMap all_tasks = i_util_fac->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    i->second->schedule_init(level, sched, matls, is_restart); 
  }

  //transport factory
  all_tasks.clear();
  all_tasks = i_trans_fac->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    i->second->schedule_init(level, sched, matls, is_restart); 
  }

  //initialize factory
  all_tasks.clear();
  all_tasks = i_init_fac->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    if ( i->first == "Lx" || i->first == "Lvel" || i->first == "Ld"){ 
      std::cout << "Delaying particle calc..." << std::endl;
    } else { 
      i->second->schedule_init(level, sched, matls, is_restart); 
    }
  }
  //have to delay and order these specific tasks...clean this up later...
  TaskFactoryBase::TaskMap::iterator iLX = all_tasks.find("Lx");
  if ( iLX != all_tasks.end() ) iLX->second->schedule_init(level, sched, matls, is_restart); 
  TaskFactoryBase::TaskMap::iterator iLD = all_tasks.find("Ld"); 
  if ( iLD != all_tasks.end() ) iLD->second->schedule_init(level, sched, matls, is_restart); 
  TaskFactoryBase::TaskMap::iterator iLV = all_tasks.find("Lvel");
  if ( iLV != all_tasks.end() ) iLV->second->schedule_init(level, sched, matls, is_restart); 

  //lagrangian particles
  all_tasks.clear();
  all_tasks = i_lag_fac->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    i->second->schedule_init(level, sched, matls, is_restart ); 
  }

  sched_scalarInit(level, sched);
  //property models
  all_tasks.clear();
  all_tasks = i_property_models_fac->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    i->second->schedule_init(level, sched, matls, is_restart ); 
  }

  //===============================================================

  // base initialization of all scalars
 // sched_scalarInit(level, sched);

  //pass some periodic stuff around.
  IntVector periodic_vector = level->getPeriodicBoundaries();
  bool d_3d_periodic = (periodic_vector == IntVector(1,1,1));
  d_turbModel->set3dPeriodic(d_3d_periodic);
  d_props->set3dPeriodic(d_3d_periodic);

  init_timelabel = scinew TimeIntegratorLabel(d_lab, TimeIntegratorStepType::FE);
  init_timelabel_allocated = true;

  // Property model initialization
  PropertyModelFactory& propFactory = PropertyModelFactory::self();
  PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
  for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
      iprop != all_prop_models.end(); iprop++){

    PropertyModelBase* prop_model = iprop->second;
    prop_model->sched_initialize( level, sched );
  }

  // Table Lookup
  bool initialize_it = true;
  bool modify_ref_den = true;
  int time_substep = 0; //no meaning here, but is required to be zero for 
                        //variables to be properly allocated. 
                        //
  d_props->doTableMatching();
  d_props->sched_computeProps( level, sched, initialize_it, modify_ref_den, time_substep );

  for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
      iprop != all_prop_models.end(); iprop++){
    PropertyModelBase* prop_model = iprop->second;
        if ( prop_model->initType()=="physical" )
          prop_model->sched_computeProp( level, sched, 1 );
  }

  //Setup BC areas
  d_boundaryCondition->sched_computeBCArea__NEW( sched, level, matls );

  //For debugging
  //d_boundaryCondition->printBCInfo();
  
  //Setup initial inlet velocities 
  d_boundaryCondition->sched_setupBCInletVelocities__NEW( sched, level, matls, d_doingRestart );

  //Set the initial profiles
  d_boundaryCondition->sched_setInitProfile__NEW( sched, level, matls );

  //Setup the intrusions. 
  d_boundaryCondition->sched_setupNewIntrusions( sched, level, matls );

  sched_getCCVelocities(level, sched);

  if (!d_MAlab) {
    
    if (d_mixedModel) {
      d_scaleSimilarityModel->sched_reComputeTurbSubmodel(sched, level, matls,
                                                          init_timelabel);
    }
    
    d_turbModel->sched_reComputeTurbSubmodel(sched, level, matls, init_timelabel);
    
  }

  //______________________
  //Data Analysis
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }

  //----------------------
  //DQMOM initialization
  if(d_doDQMOM)
  {
    sched_weightInit(level, sched);
    sched_weightedAbsInit(level, sched);

    // check to make sure that all dqmom equations have BCs set.
    DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns();
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
      EqnBase* eqn = ieqn->second;
      eqn->sched_checkBCs( level, sched );
      //as needed for the coal propery models
      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(ieqn->second); 
      dqmom_eqn->sched_getUnscaledValues( level, sched ); 
    }
    d_partVel->schedInitPartVel(level, sched);
  }


  //----------------------
  //CQMOM initialization
  if(d_doCQMOM)
  {
    sched_momentInit( level, sched );
    
    // check to make sure that all cqmom equations have BCs set.
    CQMOMEqnFactory& cqmom_factory = CQMOMEqnFactory::self();
    CQMOMEqnFactory::EqnMap& cqmom_eqns = cqmom_factory.retrieve_all_eqns();
    for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin(); ieqn != cqmom_eqns.end(); ieqn++){
      EqnBase* eqn = ieqn->second;
      eqn->sched_checkBCs( level, sched );
    }
    //call the cqmom inversion so weights and abscissas are calculated at the start
    d_cqmomSolver->sched_solveCQMOMInversion( level, sched, 0 );
  }

  //=================================================================================
  //NEW TASK INTERFACE 
  //
  //particle models
  all_tasks.clear(); 
  all_tasks = i_partmod_fac->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    i->second->schedule_init(level, sched, matls, is_restart ); 
  }
  //=================================================================================

  
  // check to make sure that all the scalar variables have BCs set and set intrusions:
  EqnFactory& eqnFactory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
    EqnBase* eqn = ieqn->second;
    eqn->sched_checkBCs( level, sched );

    // also, do table initialization here since all scalars should be initialized by now 
    if (eqn->does_table_initialization()){
      eqn->sched_tableInitialization( level, sched ); 
    }
  }

  d_boundaryCondition->sched_setIntrusionTemperature( sched, level, matls );
  
  d_boundaryCondition->sched_create_radiation_temperature( sched, level, matls, false );

  if ( _doLagrangianParticles ){ 
    _particlesHelper->schedule_sync_particle_position(level,sched,true);
  }

  //finally set the momentum (velocity) initial condition
  d_nlSolver->sched_setInitVelCond( level, sched, matls );

  //d_rad_prop_calc->sched_compute_radiation_properties( level, sched, matls, 0, true );
}


// ****************************************************************************
// 
// ****************************************************************************
void 
Arches::scheduleRestartInitialize(const LevelP& level,
                                     SchedulerP& sched)
{

  bool is_restart = true; 
  const MaterialSet* matls = d_sharedState->allArchesMaterials();

  typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
  BFM::iterator i_property_models_fac = _boost_factory_map.find("property_models_factory"); 
  TaskFactoryBase::TaskMap all_tasks = i_property_models_fac->second->retrieve_all_tasks(); 

  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){ 
    i->second->schedule_init(level, sched, matls, is_restart ); 
  }

}

void
Arches::restartInitialize()
{
  d_doingRestart = true;
  d_lab->recompile_taskgraph = true; //always recompile on restart...
}

// ****************************************************************************
// schedule the initialization of parameters
// ****************************************************************************
void
Arches::sched_paramInit(const LevelP& level,
                        SchedulerP& sched)
{
    // primitive variable initialization
    Task* tsk = scinew Task( "Arches::paramInit", this, &Arches::paramInit);

    printSchedule(level,dbg,"Arches::paramInit");

    tsk->computes(d_lab->d_cellInfoLabel);
    tsk->computes(d_lab->d_uVelocitySPBCLabel);
    tsk->computes(d_lab->d_vVelocitySPBCLabel);
    tsk->computes(d_lab->d_wVelocitySPBCLabel);
    tsk->computes(d_lab->d_uVelRhoHatLabel);
    tsk->computes(d_lab->d_vVelRhoHatLabel);
    tsk->computes(d_lab->d_wVelRhoHatLabel);
    tsk->computes(d_lab->d_CCVelocityLabel);
    tsk->computes(d_lab->d_CCUVelocityLabel);
    tsk->computes(d_lab->d_CCVVelocityLabel);
    tsk->computes(d_lab->d_CCWVelocityLabel);
    tsk->computes(d_lab->d_pressurePSLabel);
    tsk->computes(d_lab->d_densityGuessLabel);
    tsk->computes(d_lab->d_totalKineticEnergyLabel); 
    tsk->computes(d_lab->d_kineticEnergyLabel); 
    if ( VarLabel::find("true_wall_temperature"))
      tsk->computes(VarLabel::find("true_wall_temperature")); 

    if (!((d_timeIntegratorType == "FE")||(d_timeIntegratorType == "BE"))){
      tsk->computes(d_lab->d_pressurePredLabel);
    }
    if (d_timeIntegratorType == "RK3SSP"){
      tsk->computes(d_lab->d_pressureIntermLabel);
    }

    tsk->computes(d_lab->d_densityCPLabel);
    tsk->computes(d_lab->d_viscosityCTSLabel);
    tsk->computes(d_lab->d_turbViscosLabel);
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
Arches::paramInit(const ProcessorGroup* pg,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw)
{
  double old_delta_t = 0.0;
  new_dw->put(delt_vartype(old_delta_t), d_lab->d_oldDeltaTLabel);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    // Initialize cellInformation
    PerPatch<CellInformationP> cellInfoP;
    cellInfoP.setData(scinew CellInformation(patch));
    new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);

    CCVariable<double> density_guess;
    new_dw->allocateAndPut( density_guess, d_lab->d_densityGuessLabel, indx, patch );
    density_guess.initialize(0.0);

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
    CCVariable<double> density;
    CCVariable<double> viscosity;
    CCVariable<double> turb_viscosity; 
    CCVariable<double> pPlusHydro;
    CCVariable<double> mmgasVolFrac;
    CCVariable<double> ccUVelocity;
    CCVariable<double> ccVVelocity;
    CCVariable<double> ccWVelocity;

    if ( VarLabel::find("true_wall_temperature")){

      CCVariable<double> true_wall_temperature; 
      new_dw->allocateAndPut( true_wall_temperature, VarLabel::find("true_wall_temperature"), indx, patch ); 
      true_wall_temperature.initialize(0.0);

    }

    CCVariable<double> ke; 
    new_dw->allocateAndPut( ke, d_lab->d_kineticEnergyLabel, indx, patch ); 
    ke.initialize(0.0); 
    new_dw->put( sum_vartype(0.0), d_lab->d_totalKineticEnergyLabel ); 

    new_dw->allocateAndPut(ccVelocity, d_lab->d_CCVelocityLabel, indx, patch);
    ccVelocity.initialize(Vector(0.,0.,0.));
    
    new_dw->allocateAndPut( ccUVelocity, d_lab->d_CCUVelocityLabel , indx, patch );
    new_dw->allocateAndPut( ccVVelocity, d_lab->d_CCVVelocityLabel , indx, patch );
    new_dw->allocateAndPut( ccWVelocity, d_lab->d_CCWVelocityLabel , indx, patch );
    ccUVelocity.initialize(0.0);
    ccVVelocity.initialize(0.0);
    ccWVelocity.initialize(0.0);

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

    new_dw->allocateAndPut(density,   d_lab->d_densityCPLabel,    indx, patch);
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, indx, patch);
    new_dw->allocateAndPut(turb_viscosity,    d_lab->d_turbViscosLabel,       indx, patch);

    uVelocity.initialize(0.0);
    vVelocity.initialize(0.0);
    wVelocity.initialize(0.0);
    density.initialize(0.0);
    pressure.initialize(0.0);
    double visVal = d_physicalConsts->getMolecularViscosity();
    viscosity.initialize(visVal);
    turb_viscosity.initialize(0.0);

  } // patches
}

// ****************************************************************************
// schedule computation of stable time step
//    You must compute a delT on every level
// ****************************************************************************
void
Arches::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{  
  // primitive variable initialization
  Task* tsk = scinew Task( "Arches::computeStableTimeStep",this,
                           &Arches::computeStableTimeStep);

  printSchedule(level,dbg, "Arches::computeStableTimeStep");
  
  if(level->getIndex() == d_archesLevelIndex) {
  
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
  }

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
  const Level* level = getLevel(patches);
  // You have to compute it on every level but
  // only computethe real delT on the archesLevel
  if( level->getIndex() == d_archesLevelIndex ) {      
  
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

      double delta_t = d_initial_dt;
      double small_num = 1e-30;
      double delta_t2 = delta_t;

      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            double tmp_time;

//            if (d_MAlab) {
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
 //           }
 //           else
 //             tmp_time=Abs(uVelocity[currCell])/(cellinfo->sew[colX])+
 //               Abs(vVelocity[currCell])/(cellinfo->sns[colY])+
 //               Abs(wVelocity[currCell])/(cellinfo->stb[colZ])+
 //               (visc[currCell]/den[currCell])*
 //               (1.0/(cellinfo->sew[colX]*cellinfo->sew[colX]) +
 //                1.0/(cellinfo->sns[colY]*cellinfo->sns[colY]) +
 //                1.0/(cellinfo->stb[colZ]*cellinfo->stb[colZ])) +
 //               small_num;

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


      delta_t = delta_t2; 
      new_dw->put(delt_vartype(delta_t),  d_sharedState->get_delt_label(), level);

    }
  } else {  // if not on the arches level

    new_dw->put(delt_vartype(9e99),  d_sharedState->get_delt_label(),level);

  }
}

void 
Arches::MPMArchesIntrusionSetupForResart( const LevelP& level, SchedulerP& sched, bool& recompile, bool doing_restart )
{ 
  if ( doing_restart ) { 
    recompile = true; 
  }
} 

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void
Arches::scheduleTimeAdvance( const LevelP& level,
                             SchedulerP& sched)
{
 // Only schedule
 if(level->getIndex() != d_archesLevelIndex)
  return;
  
  printSchedule(level,dbg, "Arches::scheduleTimeAdvance");

  nofTimeSteps++ ;
  
  if( d_sharedState->isRegridTimestep() ){  // needed for single level regridding on restarts
    d_doingRestart = true;                  // this task is called twice on a regrid.
  }

  if (d_doingRestart  ) {

    const MaterialSet* matls = d_sharedState->allArchesMaterials();

    d_boundaryCondition->sched_computeBCArea__NEW( sched, level, matls );
    d_boundaryCondition->sched_setupBCInletVelocities__NEW( sched, level, matls, d_doingRestart );

    EqnFactory& eqnFactory = EqnFactory::self();
    EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
    for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
      EqnBase* eqn = ieqn->second;
      eqn->sched_checkBCs( level, sched );
    }

    d_nlSolver->checkMomBCs( sched, level, matls ); 

    d_boundaryCondition->sched_setupNewIntrusionCellType( sched, level, matls, d_doingRestart );
    d_boundaryCondition->sched_setupNewIntrusions( sched, level, matls );

    Operators& opr = Operators::self();
    opr.set_my_world( d_myworld ); 
    opr.create_patch_operators( level, sched, matls );
  }
  
  d_nlSolver->nonlinearSolve(level, sched);

  if ( _doLagrangianParticles ){ 

    typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
    BFM::iterator i_lag_fac = _boost_factory_map.find("lagrangian_factory"); 
    TaskFactoryBase::TaskMap all_tasks = i_lag_fac->second->retrieve_all_tasks(); 

    TaskFactoryBase::TaskMap::iterator i_part_size_update = all_tasks.find("update_particle_size");  
    TaskFactoryBase::TaskMap::iterator i_part_pos_update = all_tasks.find("update_particle_position");  
    TaskFactoryBase::TaskMap::iterator i_part_vel_update = all_tasks.find("update_particle_velocity");  

    //UPDATE SIZE
    i_part_size_update->second->schedule_task( level, sched, d_sharedState->allArchesMaterials(), TaskInterface::STANDARD_TASK, 0); 
    //UPDATE POSITION 
    i_part_pos_update->second->schedule_task( level, sched, d_sharedState->allArchesMaterials(), TaskInterface::STANDARD_TASK, 0); 
    //UPDATE VELOCITY
    i_part_vel_update->second->schedule_task( level, sched, d_sharedState->allArchesMaterials(), TaskInterface::STANDARD_TASK, 0); 

    _particlesHelper->schedule_sync_particle_position(level,sched);
    _particlesHelper->schedule_transfer_particle_ids(level,sched);
    _particlesHelper->schedule_relocate_particles(level,sched);
    _particlesHelper->schedule_add_particles(level, sched); 

  }

  //__________________________________
  //  on the fly analysis
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis( sched, level);
    }
  }

  if (d_doingRestart) {
    d_doingRestart = false;
    d_lab->recompile_taskgraph = true;

  }
}

// ****************************************************************************
// Function to return boolean for recompiling taskgraph
// ****************************************************************************
bool Arches::needRecompile(double time, double dt,
                            const GridP& grid)
{
  bool temp;
  if ( d_lab->recompile_taskgraph ) {
    //Currently turning off recompile after.
    temp = d_lab->recompile_taskgraph;
    proc0cout << "\n NOTICE: Recompiling task graph. \n \n";
    d_lab->recompile_taskgraph = false;
    return temp;
  }
  else
    return d_lab->recompile_taskgraph;
}

//___________________________________________________________________________
//
void
Arches::sched_scalarInit( const LevelP& level,
                          SchedulerP& sched )
{
  Task* tsk = scinew Task( "Arches::scalarInit",
                           this, &Arches::scalarInit);

  printSchedule(level,dbg,"Arches::scalarInit");

  EqnFactory& eqnFactory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
    EqnBase* eqn = ieqn->second;

    const VarLabel* tempVar = eqn->getTransportEqnLabel();
    tsk->computes( tempVar );
  }
  
  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

  //__________________________________
  //  initialize src terms
  SourceTermFactory& srcFactory = SourceTermFactory::self();
  SourceTermFactory::SourceMap& sources = srcFactory.retrieve_all_sources();
  for (SourceTermFactory::SourceMap::iterator isrc=sources.begin(); isrc !=sources.end(); isrc++){
    SourceTermBase* src = isrc->second;
    src->sched_initialize(level, sched);
  }
}

//______________________________________________________________________
//
void
Arches::scalarInit( const ProcessorGroup* ,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw )
{
  coutLock.lock();
  proc0cout << "Initializing all scalar equations and sources..." << std::endl;
  for (int p = 0; p < patches->size(); p++){
    //assume only one material for now
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    EqnFactory& eqnFactory = EqnFactory::self();
    EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
    for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){

      EqnBase* eqn = ieqn->second;
      std::string eqn_name = ieqn->first;
      const VarLabel* phiLabel = eqn->getTransportEqnLabel();

      CCVariable<double> phi;
      CCVariable<double> oldPhi;
      constCCVariable<double> eps_v;
      new_dw->allocateAndPut( phi, phiLabel, matlIndex, patch );
      new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, Ghost::None, 0 );

      phi.initialize(0.0);

      // initialize to something other than zero if desired.
      eqn->initializationFunction( patch, phi, eps_v );

      //do Boundary conditions
      eqn->computeBCsSpecial( patch, eqn_name, phi );

    }
  }
  coutLock.unlock();
}
//___________________________________________________________________________
//
void
Arches::sched_weightInit( const LevelP& level,
                         SchedulerP& sched )
{
  Task* tsk = scinew Task( "Arches::weightInit",
                           this, &Arches::weightInit);

  printSchedule(level,dbg,"Arches::weightInit");

  // DQMOM weight transport vars
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
  for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
    EqnBase* temp_eqn = ieqn->second;
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(temp_eqn);

    if (eqn->weight()){
      const VarLabel* tempVar = eqn->getTransportEqnLabel();
      const VarLabel* tempVar_icv = eqn->getUnscaledLabel();
      const VarLabel* tempSource = eqn->getSourceLabel();

      tsk->computes( tempVar );
      tsk->computes( tempVar_icv );
      tsk->computes( tempSource );
    }
  }

  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}
//______________________________________________________________________
//
void
Arches::weightInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw )
{
  // ***************************************
  // QUESTION: WHY DOES THIS FUNCTION GET CALLED ON THE FIRST TIMESTEP RATHER THAN THE ZEROTH TIMESTEP???
  // This causes several problems:
  // - The RHS/unweighted variables are unavailable until the first timestep
  //      (not available at zeroth timestep because they're initialized here, and this isn't called until the first timestep)
  // - DQMOM & other scalars are not initialized until the first timestep, so at the first timestep (if you output it) everything is 0.0!!!
  //      (this means that if you initialize your scalar to be a step function, you never see the scalar as the actual step function;
  //        at the zeroth timestep the scalar is 0.0 everywhere, and at the first timestep the step function has already been convected/diffused)
  // ***************************************

  proc0cout << "Initializing all DQMOM weight equations..." << endl;
  for (int p = 0; p < patches->size(); p++){
    //assume only one material for now.
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    CCVariable<Vector> partVel;
    constCCVariable<double> eps_v;

    new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, Ghost::None, 0 );

    DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();

    // --- DQMOM EQNS
    // do only weights
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin();
         ieqn != dqmom_eqns.end(); ieqn++){

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(ieqn->second);
      string eqn_name = ieqn->first;

      if (eqn->weight()){
        // This is a weight equation
        const VarLabel* sourceLabel  = eqn->getSourceLabel();
        const VarLabel* phiLabel     = eqn->getTransportEqnLabel();
        const VarLabel* phiLabel_icv = eqn->getUnscaledLabel();

        CCVariable<double> source;
        CCVariable<double> phi;
        CCVariable<double> phi_icv;

        new_dw->allocateAndPut( source,  sourceLabel,  matlIndex, patch );
        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch );
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch );

        source.initialize(0.0);
        phi.initialize(0.0);
        phi_icv.initialize(0.0);

        // initialize phi
        eqn->initializationFunction( patch, phi, eps_v );

        // do boundary conditions
        eqn->computeBCs( patch, eqn_name, phi );
      }
    }
  proc0cout << endl;
  }
}

//___________________________________________________________________________
//
void
Arches::sched_weightedAbsInit( const LevelP& level,
                               SchedulerP& sched )
{
  Task* tsk = scinew Task( "Arches::weightedAbsInit",
                           this, &Arches::weightedAbsInit);
  // DQMOM transport vars
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
  for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){

    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(ieqn->second); 

    if (!eqn->weight()) {
      const VarLabel* tempVar = eqn->getTransportEqnLabel();
      const VarLabel* tempVar_icv = eqn->getUnscaledLabel();
      const VarLabel* tempSource = eqn->getSourceLabel();

      tsk->computes( tempVar );
      tsk->computes( tempVar_icv );
      tsk->computes( tempSource );
    } else {
      const VarLabel* tempVar = eqn->getTransportEqnLabel();
      tsk->requires( Task::NewDW, tempVar, Ghost::None, 0 );
    }
  }

  // Particle Velocities

  // Models
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  CoalModelFactory::ModelMap models = modelFactory.retrieve_all_models();
  for ( CoalModelFactory::ModelMap::iterator imodel=models.begin(); imodel != models.end(); imodel++){
    ModelBase* model = imodel->second;
    const VarLabel* modelLabel = model->getModelLabel();
    const VarLabel* gasmodelLabel = model->getGasSourceLabel();

    tsk->computes( modelLabel );
    tsk->computes( gasmodelLabel );

    vector<const VarLabel*> extraLocalLabels = model->getExtraLocalLabels();
    for (vector<const VarLabel*>::iterator iexmodel = extraLocalLabels.begin(); iexmodel != extraLocalLabels.end(); iexmodel++){
      tsk->computes( *iexmodel );
    }

    model->sched_initVars( level, sched );

  }

  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}
//______________________________________________________________________
//
void
Arches::weightedAbsInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw )
{

  string msg = "Initializing all DQMOM weighted abscissa equations...";
  proc0cout << msg << std::endl;

  for (int p = 0; p < patches->size(); p++){
    //assume only one material for now.
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    CCVariable<Vector> partVel;
    constCCVariable<double> eps_v;

    Ghost::GhostType  gn = Ghost::None;

    new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, gn, 0 );

    DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();

    // --- DQMOM EQNS
    // do weights first because we need them later for the weighted abscissas
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin();
         ieqn != dqmom_eqns.end(); ieqn++){

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(ieqn->second);
      string eqn_name = ieqn->first;
      int qn = eqn->getQuadNode();

      if (!eqn->weight()) {
        // This is a weighted abscissa
        const VarLabel* sourceLabel  = eqn->getSourceLabel();
        const VarLabel* phiLabel     = eqn->getTransportEqnLabel();
        const VarLabel* phiLabel_icv = eqn->getUnscaledLabel();
        std::string weight_name;
        std::string node;
        std::stringstream out;
        out << qn;
        node = out.str();
        weight_name = "w_qn";
        weight_name += node;
        EqnBase& w_eqn = dqmomFactory.retrieve_scalar_eqn(weight_name);
        const VarLabel* weightLabel = w_eqn.getTransportEqnLabel();

        CCVariable<double> source;
        CCVariable<double> phi;
        CCVariable<double> phi_icv;
        constCCVariable<double> weight;

        new_dw->allocateAndPut( source,  sourceLabel,  matlIndex, patch );
        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch );
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch );
        new_dw->get( weight, weightLabel, matlIndex, patch, gn, 0 );

        source.initialize(0.0);
        phi.initialize(0.0);
        phi_icv.initialize(0.0);

        // initialize phi
        if( d_which_dqmom == "unweightedAbs" ){
          eqn->initializationFunction( patch, phi, eps_v);
        } else {
          eqn->initializationFunction( patch, phi, weight, eps_v );
        }

        // do boundary conditions
        eqn->computeBCs( patch, eqn_name, phi );

      }
    }

     // --- PARTICLE VELS


    // --- MODELS VALUES
    CoalModelFactory& modelFactory = CoalModelFactory::self();
    CoalModelFactory::ModelMap models = modelFactory.retrieve_all_models();
    for ( CoalModelFactory::ModelMap::iterator imodel=models.begin();
          imodel != models.end(); imodel++){

      ModelBase* model = imodel->second;
      const VarLabel* modelLabel = model->getModelLabel();
      CCVariable<double> model_value;
      new_dw->allocateAndPut( model_value, modelLabel, matlIndex, patch );
      model_value.initialize(0.0);

      const VarLabel* gasModelLabel = model->getGasSourceLabel();
      CCVariable<double> gas_source;
      new_dw->allocateAndPut( gas_source, gasModelLabel, matlIndex, patch );
      gas_source.initialize(0.0);


      vector<const VarLabel*> extraLocalLabels = model->getExtraLocalLabels();
      for (vector<const VarLabel*>::iterator iexmodel = extraLocalLabels.begin(); iexmodel != extraLocalLabels.end(); iexmodel++){
        CCVariable<double> extraVar;
        new_dw->allocateAndPut( extraVar, *iexmodel, matlIndex, patch );
        extraVar.initialize(0.0);
      }

    }
  }
  proc0cout << endl;
}
//___________________________________________________________________________
//
void
Arches::sched_momentInit( const LevelP& level,
                          SchedulerP& sched )
{
  Task* tsk = scinew Task( "Arches::momentInit",
                          this, &Arches::momentInit);
  
  printSchedule(level,dbg,"Arches::momentInit");
  
  // CQMOM moment transport vars
  CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
  CQMOMEqnFactory::EqnMap& cqmom_eqns = cqmomFactory.retrieve_all_eqns();
  for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin(); ieqn != cqmom_eqns.end(); ieqn++){
    EqnBase* temp_eqn = ieqn->second;
    CQMOMEqn* eqn = dynamic_cast<CQMOMEqn*>(temp_eqn);
    
    const VarLabel* tempVar = eqn->getTransportEqnLabel();
    const VarLabel* tempSource = eqn->getSourceLabel();
      
    tsk->computes( tempVar );
    tsk->computes( tempSource );
  }
  
  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );
  
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}
//______________________________________________________________________
//
void
Arches::momentInit( const ProcessorGroup* ,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw )
{
  
  proc0cout << "Initializing all CQMOM moment equations..." << endl;
  for (int p = 0; p < patches->size(); p++){
    //assume only one material for now.
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);
    
    constCCVariable<double> eps_v;
    
    new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, Ghost::None, 0 );
    
    CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
    CQMOMEqnFactory::EqnMap& cqmom_eqns = cqmomFactory.retrieve_all_eqns();
    
    // --- CQMOM EQNS
    for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin();
         ieqn != cqmom_eqns.end(); ieqn++){
      
      CQMOMEqn* eqn = dynamic_cast<CQMOMEqn*>(ieqn->second);
      string eqn_name = ieqn->first;
      
      const VarLabel* sourceLabel  = eqn->getSourceLabel();
      const VarLabel* phiLabel     = eqn->getTransportEqnLabel();
      
      CCVariable<double> source;
      CCVariable<double> phi;
      
      new_dw->allocateAndPut( source,  sourceLabel,  matlIndex, patch );
      new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch );
        
      source.initialize(0.0);
      phi.initialize(0.0);
      
      // initialize phi
      eqn->initializationFunction( patch, phi, eps_v );
        
      // do boundary conditions
      eqn->computeBCs( patch, eqn_name, phi );
      
    }
    proc0cout << endl;
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

  printSchedule(level,dbg,"Arches::getCCVelocities");

  Ghost::GhostType  gaf = Ghost::AroundFaces;
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  tsk->modifies(d_lab->d_CCVelocityLabel);
  tsk->modifies(d_lab->d_CCUVelocityLabel);
  tsk->modifies(d_lab->d_CCVVelocityLabel);
  tsk->modifies(d_lab->d_CCWVelocityLabel);

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
    CCVariable<Vector> vel_CC;
    CCVariable<double> uVel_CC;
    CCVariable<double> vVel_CC;
    CCVariable<double> wVel_CC;

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

    new_dw->getModifiable(vel_CC,  d_lab->d_CCVelocityLabel, indx, patch);
    vel_CC.initialize(Vector(0.0,0.0,0.0));
    
    new_dw->getModifiable(uVel_CC,  d_lab->d_CCUVelocityLabel, indx, patch);
    new_dw->getModifiable(vVel_CC,  d_lab->d_CCVVelocityLabel, indx, patch);
    new_dw->getModifiable(wVel_CC,  d_lab->d_CCWVelocityLabel, indx, patch);
    uVel_CC.initialize( 0.0 );
    vVel_CC.initialize( 0.0 );
    wVel_CC.initialize( 0.0 );
    
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

      double u,v,w;

      u = cellinfo->wfac[i] * uvel_FC[c] +
          cellinfo->efac[i] * uvel_FC[idxU];

      v = cellinfo->sfac[j] * vvel_FC[c] +
          cellinfo->nfac[j] * vvel_FC[idxV];

      w = cellinfo->bfac[k] * wvel_FC[c] +
          cellinfo->tfac[k] * wvel_FC[idxW];

      vel_CC[c] = Vector(u, v, w);
      //NOTE: this function could probably be nebo-ized with interp later
      uVel_CC[c] = u;
      vVel_CC[c] = v;
      wVel_CC[c] = w;
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

        double u,v,w;

        u = one_or_zero.x() *
            (cellinfo->wfac[i] * uvel_FC[c] +
             cellinfo->efac[i] * uvel_FC[idxU]) +
            (1.0 - one_or_zero.x()) * uvel_FC[idxU];

        v = one_or_zero.y() *
            (cellinfo->sfac[j] * vvel_FC[c] +
             cellinfo->nfac[j] * vvel_FC[idxV]) +
            (1.0 - one_or_zero.y()) * vvel_FC[idxV];

        w = one_or_zero.z() *
            (cellinfo->bfac[k] * wvel_FC[c] +
             cellinfo->tfac[k] * wvel_FC[idxW] ) +
            (1.0 - one_or_zero.z()) * wvel_FC[idxW];

        vel_CC[c] = Vector( u, v, w );
        uVel_CC[c] = u;
        vVel_CC[c] = v;
        wVel_CC[c] = w;
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

//---------------------------------------------------------------------------
// Method: Register Models
//---------------------------------------------------------------------------
void Arches::registerModels(ProblemSpecP& db)
{
  //ProblemSpecP models_db = db->findBlock("DQMOM")->findBlock("Models");
  ProblemSpecP models_db = db->findBlock("Models");

  // Get reference to the model factory
  CoalModelFactory& model_factory = CoalModelFactory::self();
  // Get reference to the dqmom factory
  DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();

  proc0cout << "\n";
  proc0cout << "******* Model Registration ********" << endl;

  // There are three kind of variables to worry about:
  // 1) internal coordinates
  // 2) other "extra" scalars
  // 3) standard flow variables
  // We want the model to have access to all three.
  // Thus, for 1) you just set the internal coordinate name and the "_qn#" is attached.  This means the models are reproduced qn times
  // for 2) you specify this in the <scalarVars> tag
  // for 3) you specify this in the implementation of the model itself (ie, no user input)

  // perhaps we should have an <ArchesVars> tag... for variables in Arches but
  // not in the DQMOM or Scalar equation factory
  // (this would probably need some kind of error-checking for minor typos,
  //  like "tempin" instead of "tempIN"...
  //  which would require some way of searching label names in ArchesLabel.h...
  //  which would require some kind of map in ArchesLabel.h...)

  if (models_db) {
    for (ProblemSpecP model_db = models_db->findBlock("model"); model_db != 0; model_db = model_db->findNextBlock("model")){
      std::string model_name;
      model_db->getAttribute("label", model_name);
      std::string model_type;
      model_db->getAttribute("type", model_type);

      proc0cout << endl;
      proc0cout << "Found  a model: " << model_name << endl;

      // The model must be reproduced for each quadrature node.
      const int numQuadNodes = dqmom_factory.get_quad_nodes();

      vector<string> requiredICVarLabels;
      ProblemSpecP icvar_db = model_db->findBlock("ICVars");

      if ( icvar_db ) {
        proc0cout << "Requires the following internal coordinates: " << endl;
        // These variables are only those that are specifically defined from the input file
        for (ProblemSpecP var = icvar_db->findBlock("variable");
             var !=0; var = var->findNextBlock("variable")){

          std::string label_name;
          var->getAttribute("label", label_name);

          proc0cout << "label = " << label_name << endl;
          // This map hold the labels that are required to compute this model term.
          requiredICVarLabels.push_back(label_name);
        }
      } else {
        proc0cout << "Model does not require any internal coordinates. " << endl;
      }

      // This is not immediately useful...
      // However, if we use the new TransportEqn mechanism to add extra scalars,
      //  we could potentially track flow variables (temperature, mixture frac, etc.)
      //  using the new TransportEqn mechanism, which would make it necessary to
      // be able to make our models depend on these scalars
      vector<string> requiredScalarVarLabels;
      ProblemSpecP scalarvar_db = model_db->findBlock("scalarVars");

      if ( scalarvar_db ) {
        proc0cout << "Requires the following scalar variables: " << endl;
        for (ProblemSpecP var = scalarvar_db->findBlock("variable");
             var != 0; var = var->findNextBlock("variable") ) {
          std::string label_name;
          var->getAttribute("label", label_name);

          proc0cout << "label = " << label_name << endl;
          // This map holds the scalar labels required to compute this model term
          requiredScalarVarLabels.push_back(label_name);
        }
      } else {
        proc0cout << "Model does not require any scalar variables. " << endl;
      }

      // --- looping over quadrature nodes ---
      // This will make a model for each quadrature node.
      for (int iqn = 0; iqn < numQuadNodes; iqn++){
        std::string temp_model_name = model_name;
        std::string node;
        std::stringstream out;
        out << iqn;
        node = out.str();
        temp_model_name += "_qn";
        temp_model_name += node;

        if ( model_type == "ConstantModel" ) {
          // Model term G = constant (G = 1)
          ModelBuilder* modelBuilder = scinew ConstantModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "KobayashiSarofimDevol" ) {
          // Kobayashi Sarofim devolatilization model
          ModelBuilder* modelBuilder = scinew KobayashiSarofimDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "RichardsFletcherDevol" ) {
          // Richards Fletcher devolatilization model
          ModelBuilder* modelBuilder = scinew RichardsFletcherDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "FOWYDevol" ) {
          // Biagini Tognotti devolatilization model
          ModelBuilder* modelBuilder = scinew FOWYDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "YamamotoDevol" ) {
          ModelBuilder* modelBuilder = scinew YamamotoDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "CharOxidationShaddix" ) {
          ModelBuilder* modelBuilder = scinew CharOxidationShaddixBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "EnthalpyShaddix" ) {
          ModelBuilder* modelBuilder = scinew EnthalpyShaddixBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, d_props, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "Drag" ) {
          ModelBuilder* modelBuilder = scinew DragModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "SimpleBirth" ) {
          ModelBuilder* modelBuilder = scinew SimpleBirthBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else {
          proc0cout << "For model named: " << temp_model_name << endl;
          proc0cout << "with type: " << model_type << endl;
          std::string errmsg;
          errmsg = model_type + ": This model type not recognized or not supported.";
          throw InvalidValue(errmsg, __FILE__, __LINE__);
        }
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Register Eqns
//---------------------------------------------------------------------------
void Arches::registerTransportEqns(ProblemSpecP& db)
{
  ProblemSpecP eqns_db = db;

  // Get reference to the source factory
  EqnFactory& eqnFactory = EqnFactory::self();

  if (eqns_db) {

    proc0cout << "\n";
    proc0cout << "******* Equation Registration ********" << endl;

    for (ProblemSpecP eqn_db = eqns_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){
      std::string eqn_name;
      eqn_db->getAttribute("label", eqn_name);
      std::string eqn_type;
      eqn_db->getAttribute("type", eqn_type);

      proc0cout << "Found  an equation: " << eqn_name << endl;

      // Here we actually register the equations based on their types.
      // This is only done once and so the "if" statement is ok.
      // Equations are then retrieved from the factory when needed.
      // The keys are currently strings which might be something we want to change if this becomes inefficient
      if ( eqn_type == "CCscalar" ) {

        EqnBuilder* scalarBuilder = scinew CCScalarEqnBuilder( d_lab, d_timeIntegrator, eqn_name );
        eqnFactory.register_scalar_eqn( eqn_name, scalarBuilder );

      // ADD OTHER OPTIONS HERE if ( eqn_type == ....

      } else {
        proc0cout << "For equation named: " << eqn_name << endl;
        proc0cout << "with type: " << eqn_type << endl;
        throw InvalidValue("This equation type not recognized or not supported! ", __FILE__, __LINE__);
      }
    }
  }
}
//---------------------------------------------------------------------------
// Method: Register Property Models
//---------------------------------------------------------------------------
void Arches::registerPropertyModels(ProblemSpecP& db)
{
  ProblemSpecP propmodels_db = db;
  PropertyModelFactory& prop_factory = PropertyModelFactory::self();

  if ( propmodels_db ) {

    proc0cout << "\n";
    proc0cout << "******* Property Model Registration *******" << endl;

    for ( ProblemSpecP prop_db = propmodels_db->findBlock("model");
        prop_db != 0; prop_db = prop_db->findNextBlock("model") ){

      std::string prop_name;
      prop_db->getAttribute("label", prop_name);
      std::string prop_type;
      prop_db->getAttribute("type", prop_type);

      proc0cout << "Found a property model: " << prop_name << endl;

      if ( prop_type == "cc_constant" ) {

        // An example of a constant CC variable property
        PropertyModelBase::Builder* the_builder = new ConstProperty<CCVariable<double>, constCCVariable<double> >::Builder( prop_name, d_sharedState );
        prop_factory.register_property_model( prop_name, the_builder );

      }  else if ( prop_type == "extent_rxn" ) {

        // Scalar dissipation rate calculation
        PropertyModelBase::Builder* the_builder = new ExtentRxn::Builder( prop_name, d_sharedState );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "tab_strip_factor" ) {

        // Scalar dissipation rate calculation
        PropertyModelBase::Builder* the_builder = new TabStripFactor::Builder( prop_name, d_sharedState );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "fx_constant" ) {

        // An example of a constant FCX variable property
        PropertyModelBase::Builder* the_builder = new ConstProperty<SFCXVariable<double>, constCCVariable<double> >::Builder( prop_name, d_sharedState );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "empirical_soot" ){ 
        
        // emperical soot model (computes soot volume fraction and abskp) 
        PropertyModelBase::Builder* the_builder = new EmpSoot::Builder( prop_name, d_sharedState ); 
        prop_factory.register_property_model( prop_name, the_builder ); 

      } else if ( prop_type == "algebraic_scalar_diss" ){ 

        // Algebraic scalar dissipation rate 
        PropertyModelBase::Builder* the_builder = new AlgebraicScalarDiss::Builder( prop_name, d_sharedState ); 
        prop_factory.register_property_model( prop_name, the_builder ); 

      } else if ( prop_type == "heat_loss" ){ 

        // Heat loss
        PropertyModelBase::Builder* the_builder = new HeatLoss::Builder( prop_name, d_sharedState ); 
        prop_factory.register_property_model( prop_name, the_builder ); 

      } else if ( prop_type == "scalsim_variance" ){ 

        //Scalar variance using a scale similarity concept
        PropertyModelBase::Builder* the_builder = new ScalarVarianceScaleSim::Builder( prop_name, d_sharedState ); 
        prop_factory.register_property_model( prop_name, the_builder ); 
        
      } else if ( prop_type == "norm_scalar_var") {
        
        //Normalized scalar variance based on second mixfrac moment
        PropertyModelBase::Builder* the_builder = new NormScalarVariance::Builder( prop_name, d_sharedState );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "scalar_diss") {

        //Scalar dissipation based on the transported squared gradient of mixture fraction for 2-eqn scalar var model
        PropertyModelBase::Builder* the_builder = new ScalarDissipation::Builder( prop_name, d_sharedState );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "radiation_properties" ){ 

        //Radiation properties as computed through the RadPropertyCalculator
        PropertyModelBase::Builder* the_builder = new RadProperties::Builder( prop_name, d_sharedState ); 
        prop_factory.register_property_model( prop_name, the_builder );

      } else {

        proc0cout << endl;
        proc0cout << "For property model named: " << prop_name << endl;
        proc0cout << "with type: " << prop_type << endl;
        throw InvalidValue("This property model is not recognized or supported! ", __FILE__, __LINE__);

      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Register CQMOM Eqns
//---------------------------------------------------------------------------
void Arches::registerCQMOMEqns(ProblemSpecP& db)
{
  // Now do the same for CQMOM equations.
  ProblemSpecP cqmom_db = db;
  
  // Get reference to the cqmom eqn factory
  CQMOMEqnFactory& cqmom_eqnFactory = CQMOMEqnFactory::self();
  
  if (cqmom_db) {
    
    int nMoments = 0;
    int M;
    cqmom_db->require("NumberInternalCoordinates",M);
    cqmom_db->get("NumberInternalCoordinates",M);
    
    proc0cout << "# IC = " << M << endl;
    proc0cout << "******* CQMOM Equation Registration ********" << endl;
    // Make the moment transport equations
    vector<int> temp_moment_index;
    for ( ProblemSpecP db_moments = cqmom_db->findBlock("Moment");
         db_moments != 0; db_moments = db_moments->findNextBlock("Moment") ) {
      temp_moment_index.resize(0);
      db_moments->get("m", temp_moment_index);
      
      proc0cout << "creating a moment equation for: " << temp_moment_index << " as " ;
      // put moment index into vector of moment indices:
 //     momentIndexes.push_back(temp_moment_index);
      
      int index_length = temp_moment_index.size();
      if (index_length != M) {
        proc0cout << "\nIndex for moment " << temp_moment_index << " does not have same number of indexes as internal coordinate #" << M << endl;
        throw InvalidValue("All specified moment must have same number of internal coordinates", __FILE__, __LINE__);
      }
      
      //register eqns - this should make varLabel for each moment eqn
      std::string moment_name = "m_";
      std::string mIndex;
      std::stringstream out;
      for (int i = 0; i<M ; i++) {
        out << temp_moment_index[i];
        mIndex = out.str();
      }
      moment_name += mIndex;
      proc0cout << moment_name << endl;
      
      CQMOMEqnBuilderBase* eqnBuilder = scinew CQMOMEqnBuilder( d_lab, d_timeIntegrator, moment_name );
      cqmom_eqnFactory.register_scalar_eqn( moment_name, eqnBuilder );
      nMoments++;
    }
    
    cqmom_eqnFactory.set_number_moments( nMoments );
    
    //register internal coordinate names in a way to know which are velocities
  }
}

//------------------------------------------------------------------
