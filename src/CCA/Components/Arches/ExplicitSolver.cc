/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

//----- ExplicitSolver.cc ----------------------------------------------
#include <CCA/Components/Arches/ArchesParticlesHelper.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctors.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/DQMOM.h>
#include <CCA/Components/Arches/CQMOM.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/CQMOM_Convection.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>
#include <CCA/Components/Arches/TurbulenceModelPlaceholder.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/CompDynamicProcedure.h>
#include <CCA/Components/Arches/SmagorinskyModel.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>

//------------------ New Task Interface (start) --------------------------------------------------
#include <CCA/Components/Arches/Task/TaskController.h>
#include <CCA/Components/Arches/Task/TaskFactoryHelper.h>
#include <CCA/Components/Arches/TurbulenceModels/TurbulenceModelFactory.h>
#include <CCA/Components/Arches/Utility/UtilityFactory.h>
#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/LagrangianParticles/LagrangianParticleFactory.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryConditionFactory.h>
#include <CCA/Components/Arches/SourceTermsV2/SourceTermFactoryV2.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//------------------ New Task Interface (end) --------------------------------------------------

#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/CoalModels/ConstantModel.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/CoalModels/RichardsFletcherDevol.h>
#include <CCA/Components/Arches/CoalModels/FOWYDevol.h>
#include <CCA/Components/Arches/CoalModels/BirthDeath.h>
#include <CCA/Components/Arches/CoalModels/YamamotoDevol.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/EnthalpyShaddix.h>
#include <CCA/Components/Arches/CoalModels/MaximumTemperature.h>
#include <CCA/Components/Arches/CoalModels/Thermophoresis.h>
#include <CCA/Components/Arches/CoalModels/LinearSwelling.h>
#include <CCA/Components/Arches/CoalModels/ShrinkageRate.h>
#include <CCA/Components/Arches/CoalModels/Deposition.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationSmith.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationSmithConstLv0.h>
#include <CCA/Components/Arches/CoalModels/DragModel.h>
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
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/CQMOM_Convection.h>
#include <CCA/Components/Arches/ParticleModels/CQMOMSourceWrapper.h>

#include <CCA/Components/Arches/ExplicitSolver.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ArchesStatsEnum.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/MomentumSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/WallHTModels/WallModelDriver.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Timers/Timers.hpp>
#include <CCA/Components/Arches/Filter.h>

#include <sci_defs/visit_defs.h>

#include <cmath>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

// ****************************************************************************
// Default constructor for ExplicitSolver
// ****************************************************************************
ExplicitSolver::
ExplicitSolver(MaterialManagerP     & materialManager,
               PhysicalConstants    * physConst,
               const ProcessorGroup * myworld,
               ArchesParticlesHelper* particle_helper,
               SolverInterface      * hypreSolver,
               ApplicationCommon    * arches )
:
  NonlinearSolver( myworld, arches ),
  d_materialManager( materialManager ),
  d_physicalConsts( physConst ),
  _particlesHelper( particle_helper ),
  d_hypreSolver( hypreSolver ),
  d_arches( arches )
{

  d_lab  = scinew ArchesLabel();
  d_lab->setMaterialManager(materialManager);
  d_props = 0;
  d_turbModel = 0;
  d_pressSolver = 0;
  d_scaleSimilarityModel = 0;
  d_boundaryCondition = 0;
  d_momSolver = 0;
  nosolve_timelabels_allocated = false;
  d_printTotalKE = false;
  d_wall_ht_models = 0;
  d_doDQMOM = false;
  d_doCQMOM = false;
  d_init_timelabel = 0;
  d_timeIntegrator = 0;
  d_doDQMOM =  false;
  d_mixedModel = false;

  DQMOMEqnFactory&  dqmomfactory   =  DQMOMEqnFactory::self();
  dqmomfactory.set_quad_nodes(0);

  CQMOMEqnFactory& cqmomfactory    = CQMOMEqnFactory::self();
  cqmomfactory.set_number_moments(0);
  d_doCQMOM                        = false;

  //eulerian particles:
  d_partVel = 0;
  d_dqmomSolver = 0;
  d_cqmomSource = 0;
  d_cqmomSolver = 0;
  d_cqmomConvect = 0;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
ExplicitSolver::~ExplicitSolver()
{

  delete d_lab;
  delete d_props;
  delete d_tabulated_properties;
  delete d_turbModel;
  delete d_scaleSimilarityModel;
  delete d_boundaryCondition;
  delete d_pressSolver;
  delete d_momSolver;
  delete d_timeIntegrator;
  if (d_doDQMOM) {
    delete d_dqmomSolver;
    delete d_partVel;
  }
  if (d_doCQMOM) {
    delete d_cqmomSolver;
    delete d_cqmomConvect;
    delete d_cqmomSource;
  }
  if ( d_init_timelabel != 0 )
    delete d_init_timelabel;
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
    delete d_timeIntegratorLabels[curr_level];
  if (nosolve_timelabels_allocated)
    delete nosolve_timelabels;
  if ( d_wall_ht_models != 0 ){
    delete d_wall_ht_models;
  }

  for (auto i = m_bcHelper.begin(); i != m_bcHelper.end(); i++){
    delete i->second;
  }
  m_bcHelper.clear();
}

// ****************************************************************************
// Problem Setup
// ****************************************************************************
void
ExplicitSolver::problemSetup( const ProblemSpecP & params,
                              MaterialManagerP & materialManager,
                              GridP& grid )
{

  ArchesCore::TaskController& tsk_controller = ArchesCore::TaskController::self();
  tsk_controller.parse_task_controller(params);

  // check to see what radiation calc frequency is (default is 1)
  ProblemSpecP transport_ps = params->findBlock("TransportEqns");
  if (transport_ps) {
    ProblemSpecP sources_ps = transport_ps->findBlock("Sources");
    if (sources_ps) {
      ProblemSpecP rad_src_ps = sources_ps->findBlock("src");
      // find the "divQ" src block for the radiation calculation frequency
      while ( rad_src_ps !=nullptr) {
        std::string src_type = "";
        rad_src_ps->getAttribute("type", src_type);
        if (src_type == "do_radiation" || src_type == "rmcrt_radiation" ){

          // Tell the infrastructure that two tasksgraphs will be used.
          d_perform_radiation = true;
          m_arches->getScheduler()->setNumTaskGraphs( 2 );

          rad_src_ps->require("calc_frequency", d_rad_calc_frequency);

          // Check to see if the dynamic frequency radiation solve
          // should be used.
          if(1 < d_rad_calc_frequency &&
             rad_src_ps->findBlock("use_dynamic_frequency") != nullptr) {
            d_dynamicSolveFrequency = true;
          }

          if (src_type == "do_radiation" ) {
            m_arches->getApplicationStats().insert( (ApplicationCommon::ApplicationStatsEnum) DORadiationTime,   std::string("DO_Radiation_Time"),   "seconds" );
            m_arches->getApplicationStats().insert( (ApplicationCommon::ApplicationStatsEnum) DORadiationSweeps, std::string("DO_Radiation_Sweeps"), "sweeps"  );
            m_arches->getApplicationStats().insert( (ApplicationCommon::ApplicationStatsEnum) DORadiationBands,  std::string("DO_Radiation_Bands"),  "bands"   );
          }

  break;
        }
        rad_src_ps = rad_src_ps->findNextBlock("src");
      }
    }
  }

  ProblemSpecP db_es = params->findBlock("ExplicitSolver");
  ProblemSpecP db = params;

  commonProblemSetup( db_es );

  d_archesLevelIndex = grid->numLevels()-1; // this is the finest level

  //------------------------------------------------------------------------------------------------
  //Looking to see if we are using the turb models abstracted into the task Interface
  if ( db->findBlock("TurbulenceModels")){
    if ( db->findBlock("TurbulenceModels")->findBlock("model")){
      d_using_TI_turb = true;
    }
  }

  //------------------------------------------------------------------------------------------------
  //Look for coal information
  if( db->findBlock("ParticleProperties") ) {
    string particle_type;
    db->findBlock("ParticleProperties")->getAttribute("type", particle_type);
    if ( particle_type == "coal" ) {
      CoalHelper& coal_helper = CoalHelper::self();
      coal_helper.parse_for_coal_info( db );
    } else {
      throw InvalidValue("Error: Particle type not recognized. Current types supported: coal",__FILE__,__LINE__);
    }
  }

  //------------------ New Task Interface (start) --------------------------------------------------

  //build the factories
  std::shared_ptr<UtilityFactory> UtilF(scinew UtilityFactory(m_arches));
  std::shared_ptr<TransportFactory> TransF(scinew TransportFactory(m_arches));
  std::shared_ptr<InitializeFactory> InitF(scinew InitializeFactory(m_arches));
  std::shared_ptr<ParticleModelFactory> PartModF(scinew ParticleModelFactory(m_arches));
  std::shared_ptr<LagrangianParticleFactory> LagF(scinew LagrangianParticleFactory(m_arches));
  std::shared_ptr<PropertyModelFactoryV2> PropModelsF(scinew PropertyModelFactoryV2(m_arches));
  std::shared_ptr<TurbulenceModelFactory> TurbModelF(scinew TurbulenceModelFactory(m_arches));
  std::shared_ptr<BoundaryConditionFactory> BCF(scinew BoundaryConditionFactory(m_arches));
  std::shared_ptr<SourceTermFactoryV2> SourceTermV2F(scinew SourceTermFactoryV2(m_arches));

  _task_factory_map.clear();
  _task_factory_map.insert(std::make_pair("utility_factory",UtilF));
  _task_factory_map.insert(std::make_pair("transport_factory",TransF));
  _task_factory_map.insert(std::make_pair("initialize_factory",InitF));
  _task_factory_map.insert(std::make_pair("particle_model_factory",PartModF));
  _task_factory_map.insert(std::make_pair("lagrangian_factory",LagF));
  _task_factory_map.insert(std::make_pair("turbulence_model_factory", TurbModelF));
  _task_factory_map.insert(std::make_pair("boundary_condition_factory", BCF));
  _task_factory_map.insert(std::make_pair("property_models_factory", PropModelsF));
  _task_factory_map.insert(std::make_pair("source_term_factory",SourceTermV2F));

  typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
  proc0cout << "\n Registering and Building Tasks For: " << std::endl;
  for ( BFM::iterator i = _task_factory_map.begin(); i != _task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->set_materialManager(d_materialManager);
    i->second->register_all_tasks(db);

  }
  proc0cout << endl;

  //Checking for lagrangian particles:
  _doLagrangianParticles = m_arches_spec->findBlock("LagrangianParticles");

  //------------------ New Task Interface (end) ----------------------------------------------------


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

  //------------------------------------------------------------------------------------------------
  //Transport Eqns:
  ProblemSpecP transportEqn_db = db->findBlock("TransportEqns");
  if (transportEqn_db) {

    // register source terms
    SourceTermFactory& src_factory = SourceTermFactory::self();
    ProblemSpecP sources_db = transportEqn_db->findBlock("Sources");
    if (sources_db)
      src_factory.registerUDSources(sources_db, d_lab, d_boundaryCondition, d_myworld);

    //register all equations
    ExplicitSolver::registerTransportEqns(transportEqn_db);

    // Go through eqns and intialize all defined eqns and call their respective
    // problem setup
    EqnFactory& eqn_factory = EqnFactory::self();
    for (ProblemSpecP eqn_db = transportEqn_db->findBlock("Eqn"); eqn_db != nullptr; eqn_db = eqn_db->findNextBlock("Eqn")) {

      std::string eqnname;
      eqn_db->getAttribute("label", eqnname);
      if (eqnname == "") {
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

      for ( ProblemSpecP src_db = sources_db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src") ){
        std::string label;
        src_db->getAttribute("label", label);
        SourceTermBase& the_src = src_factory.retrieve_source_term( label );
        the_src.problemSetup( src_db );
      }

      src_factory.commonSrcProblemSetup( sources_db );

    }

  } else {

    proc0cout << "No defined transport equations found." << endl;

  }

  //The RMCRT radiatometer model (not RMCRT as divQ) was implemeted originally as a src,
  // which has an abstraction problem from the user point of view and creates an implementation
  // issue, specifically with some inputs overiding others due to input file ordering of specs.
  // As such, to avoid clashes with two radiation models (say DO specifying
  // the divQ and RMCRT acting as a radiometer) we need to look for it separately
  // here and put it into the src factory. This avoids a potential bug where
  // one radiation model may cancel out settings with the other. It also preserves how the code
  // actually operates without a rewrite of the model.
  ProblemSpecP db_radiometer = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Radiometer");
  if ( db_radiometer != nullptr ){
    SourceTermFactory& src_factory = SourceTermFactory::self();
    std::string label;
    db_radiometer->getAttribute("label", label);
    SourceTermBase& the_src = src_factory.retrieve_source_term( label );
    the_src.problemSetup( db_radiometer );
  }


  if ( db->findBlock("PropertyModels") ) {

    ProblemSpecP propmodels_db = db->findBlock("PropertyModels");
    PropertyModelFactory& prop_factory = PropertyModelFactory::self();
    ExplicitSolver::registerPropertyModels( propmodels_db );
    for ( ProblemSpecP prop_db = propmodels_db->findBlock("model"); prop_db != nullptr; prop_db = prop_db->findNextBlock("model") ) {

      std::string model_name;
      prop_db->getAttribute("label", model_name);
      if ( model_name == "" ) {
        throw InvalidValue( "Error: The label attribute must be specified for the property models!", __FILE__, __LINE__);
      }
      PropertyModelBase& a_model = prop_factory.retrieve_property_model( model_name );
      a_model.problemSetup( prop_db );

    }
  }

  // read properties
  d_props = scinew Properties(d_lab, d_physicalConsts, d_myworld);
  d_tabulated_properties = scinew TableLookup( d_lab->d_materialManager );
  d_tabulated_properties->problemSetup( db );

  d_props->problemSetup(db);

  //need to set bounds on heat loss as the values in the table itself
  PropertyModelFactory& propFactory = PropertyModelFactory::self();
  PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
  for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
        iprop != all_prop_models.end(); iprop++) {

    PropertyModelBase* prop_model = iprop->second;
    if ( prop_model->getPropType() == "heat_loss" ) {
      MixingRxnModel* mixing_table = d_tabulated_properties->get_table();
      std::map<string,double> table_constants = mixing_table->getAllConstants();
      if ( d_tabulated_properties->get_table_type() == TableLookup::CLASSIC ) {

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
       ieqn != scalar_eqns.end(); ieqn++) {

    EqnBase* eqn = ieqn->second;
    eqn->assign_stage_to_sources();

  }

  // read boundary condition information
  d_boundaryCondition = scinew BoundaryCondition(d_lab, d_physicalConsts,
                                                 d_props, d_tabulated_properties, m_arches->getOutput() );

  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->problemSetup(db,  grid);

  std::string whichTurbModel = "none";
  if ( db->findBlock("Turbulence") ) {
    db->findBlock("Turbulence")->getAttribute("model",whichTurbModel);
  }

  if ( whichTurbModel == "smagorinsky") {
    d_turbModel = scinew SmagorinskyModel(d_lab, d_physicalConsts,
                                          d_boundaryCondition);
  }else if ( whichTurbModel == "compdynamicprocedure") {
    d_turbModel = scinew CompDynamicProcedure(d_lab, d_physicalConsts,
                                              d_boundaryCondition);
  } else if ( whichTurbModel == "none" ) {
    proc0cout << "\n Notice: Turbulence model specificied as: none. Running without momentum closure. \n";
    d_turbModel = scinew TurbulenceModelPlaceholder(d_lab, d_physicalConsts,
                                                    d_boundaryCondition);
  } else {
    proc0cout << "\n Notice: No (old interface) Turbulence model was found. \n" << endl;
  }

  d_turbModel->problemSetup(db);

  d_turbModel->setMixedModel(d_mixedModel);
  if (d_mixedModel) {
    d_scaleSimilarityModel=scinew ScaleSimilarityModel(d_lab, d_physicalConsts,
                                                       d_boundaryCondition);
    d_scaleSimilarityModel->problemSetup(db);

    d_scaleSimilarityModel->setMixedModel(d_mixedModel);
  }

  d_props->setBC(d_boundaryCondition);

  // ----- DQMOM STUFF:
  d_kokkos_dqmom_Translate = false;
  ProblemSpecP dqmom_db = db->findBlock("DQMOM");
  if (dqmom_db) {

    //turn on DQMOM
    d_doDQMOM = true;

    if ( dqmom_db->findBlock("kokkos_translate")){
      d_kokkos_dqmom_Translate = true;
    }

    // require that we have weighted or unweighted explicitly specified as an attribute to DQMOM
    // type = "unweightedAbs" or type = "weighedAbs"
    //dqmom_db->getAttribute( "type", d_which_dqmom );
    d_which_dqmom = "weighedAbs";

    m_DQMOMSolverType = "No_Inversion";

    ProblemSpecP db_linear_solver = dqmom_db->findBlock("LinearSolver");
    if ( db_linear_solver ){

      db_linear_solver->getWithDefault("type", m_DQMOMSolverType, "LU");

      // currently, unweighted abscissas only work with the optimized solver -- remove this check when other solvers work:
      if( d_which_dqmom == "unweightedAbs" && m_DQMOMSolverType != "Optimize" ) {
        throw ProblemSetupException("Error!: The unweighted abscissas only work with the optimized solver.", __FILE__, __LINE__);
      }
    }

    DQMOMEqnFactory& eqn_factory = DQMOMEqnFactory::self();

    //register all equations.
    eqn_factory.registerDQMOMEqns(dqmom_db, d_lab, d_timeIntegrator );

    //register all models
    CoalModelFactory& model_factory = CoalModelFactory::self();
    model_factory.problemSetup(dqmom_db);

    ExplicitSolver::registerModels(dqmom_db);

    // Create a velocity model
    d_partVel = scinew PartVel( d_lab );
    d_partVel->problemSetup( dqmom_db );
    // Do through and initialze all DQMOM equations and call their respective problem setups.
    const int numQuadNodes = eqn_factory.get_quad_nodes();

    model_factory.setArchesLabel( d_lab );

    ProblemSpecP w_db = dqmom_db->findBlock("Weights");

    // do all weights
    for (int iqn = 0; iqn < numQuadNodes; iqn++) {
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
    for (ProblemSpecP ic_db = dqmom_db->findBlock("Ic"); ic_db != nullptr; ic_db = ic_db->findNextBlock("Ic")) {
      std::string ic_name;
      ic_db->getAttribute("label", ic_name);
      //loop for all quad nodes for this internal coordinate
      for (int iqn = 0; iqn < numQuadNodes; iqn++) {

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
      for (ProblemSpecP m_db = models_db->findBlock("model"); m_db != nullptr; m_db = m_db->findNextBlock("model")) {
        std::string model_name;
        m_db->getAttribute("label", model_name);
        for (int iqn = 0; iqn < numQuadNodes; iqn++) {
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

  }

  // ----- CQMOM STUFF:

  ProblemSpecP cqmom_db = db->findBlock("CQMOM");
  if (cqmom_db) {
    d_doCQMOM = true;
    bool usePartVel = false;
    cqmom_db->getAttribute( "partvel", usePartVel );

    //set up source terms and pass to explicit solver
    d_cqmomSource = scinew CQMOMSourceWrapper( d_lab );
    d_cqmomSource->problemSetup( cqmom_db );

    //register all equations.
    ExplicitSolver::registerCQMOMEqns(cqmom_db);

    // initialze all CQMOM equations and call their respective problem setups.
    CQMOMEqnFactory& eqn_factory = CQMOMEqnFactory::self();
    const int numMoments = eqn_factory.get_number_moments();
    proc0cout << "Feeding these " << numMoments << " eqns into CQMOM Eqn Factory" << endl;

    int M;
    cqmom_db->get("NumberInternalCoordinates",M);
    vector<int> temp_moment_index;
    for ( ProblemSpecP db_moments = cqmom_db->findBlock("Moment"); db_moments != nullptr; db_moments = db_moments->findNextBlock("Moment") ) {
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
      for (int i = 0; i<M; i++) {
        out << temp_moment_index[i];
        mIndex = out.str();
      }
      moment_name += mIndex;

      EqnBase& a_moment = eqn_factory.retrieve_scalar_eqn( moment_name );
      eqn_factory.set_moment_eqn( moment_name, &a_moment );
      CQMOMEqn& moment = dynamic_cast<CQMOMEqn&>(a_moment);
      moment.problemSetup( db_moments );
    }

    // set up the linear solver:
    d_cqmomSolver = scinew CQMOM( d_lab, usePartVel );
    d_cqmomSolver->problemSetup( cqmom_db );

    // set up convection
    d_cqmomConvect = scinew CQMOM_Convection( d_lab );
    if (usePartVel ) {
      d_cqmomConvect->problemSetup( cqmom_db );
    }
  }

  // register any other source terms:
  SourceTermFactory& src_factory = SourceTermFactory::self();
  src_factory.registerSources( d_lab, d_doDQMOM, d_which_dqmom );

  // do any last setup operations on the active source terms:
  src_factory.extraSetup( grid, d_boundaryCondition, d_tabulated_properties );

  // Add extra species to table lookup as required by models
  //d_props->addLookupSpecies();
  d_tabulated_properties->addLookupSpecies();

  // Add new intrusion stuff:
  // get a reference to the intrusions
  const std::map<int, IntrusionBC*> intrusion_ref = d_boundaryCondition->get_intrusion_ref();
  bool using_new_intrusions = d_boundaryCondition->is_using_new_intrusion();

  if(d_doDQMOM)
  {
    // check to make sure that all dqmom equations have BCs set.
    DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns();
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++) {
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
    for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin(); ieqn != cqmom_eqns.end(); ieqn++) {
      EqnBase* eqn = ieqn->second;
      eqn->set_intrusion( intrusion_ref );
      eqn->set_intrusion_bool( using_new_intrusions );
    }
  }

  // check to make sure that all the scalar variables have BCs set and set intrusions:
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++) {
    EqnBase* eqn = ieqn->second;
    eqn->set_intrusion( intrusion_ref );
    eqn->set_intrusion_bool( using_new_intrusions );

    //send a reference of the mixing/rxn table to the eqn for initializiation
    MixingRxnModel* d_mixingTable = d_tabulated_properties->get_table();
    eqn->set_table( d_mixingTable );

    //look for an set any tabulated bc's
    eqn->extraProblemSetup( db );
  }

  //------------------------------------------------------------------------------------------------

  if ( db_es->findBlock( "print_total_ke" ) ){
    d_printTotalKE = true;
  }

  db_es->getWithDefault( "max_ke_allowed", d_ke_limit, 1.0e99 );

  if ( db->findBlock("BoundaryConditions") ){
    if ( db->findBlock("BoundaryConditions")->findBlock( "WallHT" ) ){
      ProblemSpecP db_wall_ht = db->findBlock("BoundaryConditions")->findBlock( "WallHT" );
      d_wall_ht_models = scinew WallModelDriver( d_lab->d_materialManager );
      d_wall_ht_models->problemSetup( db_wall_ht );
    }
  }


  d_pressSolver = scinew PressureSolver( d_lab,
                                         d_boundaryCondition,
                                         d_physicalConsts, d_myworld,
                                         d_hypreSolver );
  d_pressSolver->problemSetup( db_es, materialManager );

  d_momSolver = scinew MomentumSolver(d_lab,
                                      d_turbModel, d_boundaryCondition,
                                      d_physicalConsts,
                                      &_task_factory_map );

  d_momSolver->problemSetup(db_es, materialManager);

  const ProblemSpecP params_root = db->getRootNode();
  std::string t_order;
  ProblemSpecP db_time_int = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator");
  db_time_int->findBlock("ExplicitIntegrator")->getAttribute("order", t_order);

  ProblemSpecP db_vars  = params_root->findBlock("DataArchiver");
  for( ProblemSpecP db_dv = db_vars->findBlock("save"); db_dv != nullptr; db_dv = db_dv->findNextBlock("save") ){

    std::string var_name;
    db_dv->getAttribute( "label", var_name );

    if ( var_name == "kineticEnergy" || var_name == "totalKineticEnergy" ){
      d_printTotalKE = true;
    }
  }

  //translate order to the older code:
  if ( t_order == "first" ){
    d_timeIntegratorType = "FE";
  }
  else if ( t_order == "second" ){
    d_timeIntegratorType = "RK2SSP";
  }
  else if ( t_order == "third" ) {
    d_timeIntegratorType = "RK3SSP";
  }
  else {
    throw InvalidValue("Error: <ExplicitIntegrator> order attribute must be one of: first, second, third!  Please fix input file.",__FILE__, __LINE__);
  }

  if (d_timeIntegratorType == "FE") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::FE));
    numTimeIntegratorLevels = 1;
  }
  else if (d_timeIntegratorType == "RK2") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::OldPredictor));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::OldCorrector));
    numTimeIntegratorLevels = 2;
  }
  else if (d_timeIntegratorType == "RK2SSP") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Predictor));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Corrector));
    numTimeIntegratorLevels = 2;
  }
  else if (d_timeIntegratorType == "RK3SSP") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Predictor));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Intermediate));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::CorrectorRK3));
    numTimeIntegratorLevels = 3;
  }
  else if (d_timeIntegratorType == "BEEmulation") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BEEmulation1));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BEEmulation2));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BEEmulation3));
    numTimeIntegratorLevels = 3;
  }
  else {
    throw ProblemSetupException("Integrator type is not defined "+d_timeIntegratorType,
                                __FILE__, __LINE__);
  }

  db_es->getWithDefault("turbModelCalcFreq",d_turbModelCalcFreq,1);
  db_es->getWithDefault("turbModelCalcForAllRKSteps",d_turbModelRKsteps,true);

  db_es->getWithDefault("restartOnNegativeDensityGuess",
                     d_recompute_on_negative_density_guess,false);
  db_es->getWithDefault("NoisyDensityGuess",
                     d_noisyDensityGuess, false);
  db_es->getWithDefault("kineticEnergy_fromFC",d_KE_fromFC,false);
  db_es->getWithDefault("maxDensityLag",d_maxDensityLag,0.0);

  d_extra_table_lookup = false;
  if ( db_es->findBlock("extra_table_lookup")) d_extra_table_lookup = true;

  d_props->setFilter(d_turbModel->getFilter());
  d_momSolver->setDiscretizationFilter(d_turbModel->getFilter());

  d_mixedModel=d_turbModel->getMixedModel();

  //__________________________________
  // allow for addition of mass source terms
  if (db_es->findBlock("PressureSolver")->findBlock("src")){
    ProblemSpecP db_p = db_es->findBlock("PressureSolver");
    string srcname;
    for (ProblemSpecP src_db = db_p->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      d_mass_sources.push_back( srcname );
    }
  }
  d_solvability = false;
  if ( db_es->findBlock("PressureSolver")->findBlock("enforce_solvability")){
    d_solvability = true;
  }

#ifdef HAVE_VISIT

#define IGNORE_LEVEL -2

    static bool initialized = false;

    if( m_arches->getVisIt() && !initialized ) {
      ApplicationInterface::analysisVar aVar;
      aVar.component = "Arches/ExplicitSolver";
      aVar.name  = d_lab->d_totalKineticEnergyLabel->getName();
      aVar.matl  = -1; // Ignore the material
      aVar.level = IGNORE_LEVEL;
      aVar.labels.push_back( d_lab->d_totalKineticEnergyLabel );
      m_arches->getAnalysisVars().push_back(aVar);

      initialized = true;
    }
#endif
}

//--------------------------------------------------------------------------------------------------
void
ExplicitSolver::computeTimestep(const LevelP& level, SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "ExplicitSolver::computeStableTimeStep",this,
                           &ExplicitSolver::computeStableTimeStep);

  //printSchedule(level,dbg, "ExplicitSolver::computeStableTimeStep");

  if(level->getIndex() == d_archesLevelIndex) {

    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gaf = Ghost::AroundFaces;
    Ghost::GhostType gn = Ghost::None;

    //NOTE: Hardcoding the labels for now. In the future, these can be made generic.
    d_x_vel_label = VarLabel::find("uVelocitySPBC");
    d_y_vel_label = VarLabel::find("vVelocitySPBC");
    d_z_vel_label = VarLabel::find("wVelocitySPBC");
    d_rho_label = VarLabel::find("densityCP");
    d_viscos_label = VarLabel::find("viscosityCTS");
    d_celltype_label = VarLabel::find("cellType");

    tsk->requiresVar(Task::NewDW, d_x_vel_label, gaf, 1);
    tsk->requiresVar(Task::NewDW, d_y_vel_label, gaf, 1);
    tsk->requiresVar(Task::NewDW, d_z_vel_label, gaf, 1);
    tsk->requiresVar(Task::NewDW, d_rho_label,     gac, 1);
    tsk->requiresVar(Task::NewDW, d_viscos_label,  gn,  0);
    tsk->requiresVar(Task::NewDW, d_celltype_label,  gac, 1);
  }

  tsk->computesVar(d_lab->d_delTLabel,level.get_rep());

  // This method is called at both initialization and
  // otherwise. At initialization the old DW, i.e. DW(0) will not
  // exist so require the value from the new DW.  Otherwise for a
  // normal time step require the time step from the old DW.
  if(sched->get_dw(0) ) {
    tsk->requiresVar( Task::OldDW, VarLabel::find(timeStep_name) );
  }
  else if(sched->get_dw(1) ) {
    tsk->requiresVar( Task::NewDW, VarLabel::find(timeStep_name) );
  }

#ifdef OUTPUT_WITH_BAD_DELTA_T
  // For when delta T goes below a value output and checkpoint
  tsk->computesVar( VarLabel::find(    outputTimeStep_name) );
  tsk->computesVar( VarLabel::find(checkpointTimeStep_name) );
  tsk->computesVar( VarLabel::find(     endSimulation_name) );
#endif

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

  // When restarting with radiation compute the next task graph index
  // using a fixed time stepping. Not used when doing dynamic time
  // stepping.
  if( d_perform_radiation && !d_dynamicSolveFrequency ) {
    sched_computeTaskGraphIndex( level, sched );
  }
}

//--------------------------------------------------------------------------------------------------
void
ExplicitSolver::computeStableTimeStep(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // This method is called at both initialization and otherwise. At
  // initialization the old DW will not exist so get the value from
  // the new DW.  Otherwise for a normal time step get the time step
  // from the old DW.
  timeStep_vartype timeStep(0);

  if( old_dw && old_dw->exists( VarLabel::find(timeStep_name) ) )
    old_dw->get(timeStep, VarLabel::find(timeStep_name) );
  else if( new_dw && new_dw->exists( VarLabel::find(timeStep_name) ) )
    new_dw->get(timeStep, VarLabel::find(timeStep_name) );

  const Level* level = getLevel(patches);
  // You have to compute it on every level but
  // only computethe real delT on the archesLevel
  if( level->getIndex() == d_archesLevelIndex ) {

    for (int p = 0; p < patches->size(); p++) {

      const Patch* patch = patches->get(p);
      int archIndex = 0; // only one arches material
      int indx = d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

      constSFCXVariable<double> uVelocity;
      constSFCYVariable<double> vVelocity;
      constSFCZVariable<double> wVelocity;
      constCCVariable<double> den;
      constCCVariable<double> visc;
      constCCVariable<int> cellType;

      Ghost::GhostType gac = Ghost::AroundCells;
      Ghost::GhostType gaf = Ghost::AroundFaces;
      Ghost::GhostType gn = Ghost::None;

      new_dw->get(uVelocity, d_x_vel_label, indx, patch, gaf, 1);
      new_dw->get(vVelocity, d_y_vel_label, indx, patch, gaf, 1);
      new_dw->get(wVelocity, d_z_vel_label, indx, patch, gaf, 1);
      new_dw->get(den, d_rho_label,           indx, patch, gac, 1);
      new_dw->get(visc, d_viscos_label,       indx, patch, gn,  0);
      new_dw->get(cellType, d_celltype_label, indx, patch, gac, 1);

      Vector DX = patch->dCell();

      IntVector indexLow = patch->getFortranCellLowIndex();
      IntVector indexHigh = patch->getFortranCellHighIndex();
      bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
      bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
      bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
      bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

      int press_celltypeval = BoundaryCondition::PRESSURE;
      int out_celltypeval = BoundaryCondition::OUTLET;
      if ((xminus)&&((cellType[indexLow - IntVector(1,0,0)]==press_celltypeval)
                     ||(cellType[indexLow - IntVector(1,0,0)]==out_celltypeval))) {
        indexLow = indexLow - IntVector(1,0,0);
      }

      if ((yminus)&&((cellType[indexLow - IntVector(0,1,0)]==press_celltypeval)
                     ||(cellType[indexLow - IntVector(0,1,0)]==out_celltypeval))) {
        indexLow = indexLow - IntVector(0,1,0);
      }

      if ((zminus)&&((cellType[indexLow - IntVector(0,0,1)]==press_celltypeval)
                     ||(cellType[indexLow - IntVector(0,0,1)]==out_celltypeval))) {
        indexLow = indexLow - IntVector(0,0,1);
      }

      if (xplus) {
        indexHigh = indexHigh + IntVector(1,0,0);
      }
      if (yplus) {
        indexHigh = indexHigh + IntVector(0,1,0);
      }
      if (zplus) {
        indexHigh = indexHigh + IntVector(0,0,1);
      }

      double delta_t = d_initial_dt;
      double small_num = 1e-30;
      double delta_t2 = delta_t;

      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX++) {
            IntVector currCell(colX, colY, colZ);
            double tmp_time;

            int flag = true;
            int colXm = colX - 1;
            int colXp = colX + 1;
            int colYm = colY - 1;
            int colYp = colY + 1;
            int colZm = colZ - 1;
            int colZp = colZ + 1;
            if (colXm < indexLow.x()) colXm = indexLow.x();
            if (colXp > indexHigh.x()) colXp = indexHigh.x();
            if (colYm < indexLow.y()) colYm = indexLow.y();
            if (colYp > indexHigh.y()) colYp = indexHigh.y();
            if (colZm < indexLow.z()) colZm = indexLow.z();
            if (colZp > indexHigh.z()) colZp = indexHigh.z();
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
            if (den[currCell] < 1.0e-12) flag = false;
            if ((den[xMinusCell] < 1.0e-12)&&(den[xPlusCell] < 1.0e-12)) flag = false;
            if ((den[yMinusCell] < 1.0e-12)&&(den[yPlusCell] < 1.0e-12)) flag = false;
            if ((den[zMinusCell] < 1.0e-12)&&(den[zPlusCell] < 1.0e-12)) flag = false;

            tmp_time=1.0;
            if (flag != false) {
              tmp_time= Abs(uvel)/(DX.x())+
                        Abs(vvel)/(DX.y())+
                        Abs(wvel)/(DX.z())+
                        (visc[currCell]/den[currCell])*
                        (1.0/(DX.x()*DX.x()) +
                         1.0/(DX.y()*DX.y()) +
                         1.0/(DX.z()*DX.z()) ) +
                        small_num;
            }

            delta_t2=Min(1.0/tmp_time, delta_t2);

          }
        }
      }

      //Check intrusion velocities:
      Vector max_intrusion_vel = d_boundaryCondition->getMaxIntrusionVelocity(level);
      double tmp_time= Abs(max_intrusion_vel.x())/(DX.x())+
                       Abs(max_intrusion_vel.y())/(DX.y())+
                       Abs(max_intrusion_vel.z())/(DX.z())+small_num;
      if ( tmp_time > 0. ){
        delta_t2=Min(1.0/tmp_time, delta_t2);
      }

      if (d_underflow) {

        indexLow = patch->getFortranCellLowIndex();
        indexHigh = patch->getFortranCellHighIndex();

        for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ++) {
          for (int colY = indexLow.y(); colY <= indexHigh.y(); colY++) {
            for (int colX = indexLow.x(); colX <= indexHigh.x(); colX++) {
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
                DX.x() +
                ((den[currCell]+den[yplusCell])*Max(vVelocity[yplusCell],0.0) -
                 (den[currCell]+den[yminusCell])*Min(vVelocity[currCell],0.0)) /
                DX.y() +
                ((den[currCell]+den[zplusCell])*Max(wVelocity[zplusCell],0.0) -
                 (den[currCell]+den[zminusCell])*Min(wVelocity[currCell],0.0)) /
                DX.z());

              if (den[currCell] > 0.0) {
                delta_t2=Min(den[currCell]/tmp_time, delta_t2);
              }
            }
          }
        }
      }

      delta_t = delta_t2;
      new_dw->put(delt_vartype(delta_t), d_lab->d_delTLabel, level);

#ifdef OUTPUT_WITH_BAD_DELTA_T
      // If after the first 10 time steps delta T goes below this
      // value immediately output and checkpoint. Then end the simulation.
      if( 10 < timeStep && delta_t <= 0.01 )
      {
        new_dw->put( bool_or_vartype(true), VarLabel::find(    outputTimeStep_name) );
        new_dw->put( bool_or_vartype(true), VarLabel::find(checkpointTimeStep_name) );
        new_dw->put( bool_or_vartype(true), VarLabel::find(     endSimulation_name) );
      }
#endif
    }
  } else { // if not on the arches level

    new_dw->put(delt_vartype(9e99), d_lab->d_delTLabel, level);

  }
}

//--------------------------------------------------------------------------------------------------
void
ExplicitSolver::sched_computeTaskGraphIndex(const LevelP& level, SchedulerP& sched)
{
  // This is done ONLY for the finest level as to put a fake
  // dependency on the new delta T.
  if ( !level->hasFinerLevel() ) {
    Task* task = scinew Task( "ExplicitSolver::computeTaskGraphIndex",this,
                              &ExplicitSolver::computeTaskGraphIndex);

    task->setType(Task::OncePerProc);

    // This method is called at both initialization and otherwise. At
    // initialization the old DW, i.e. DW(0) will not exist so require
    // the value from the new DW.  Otherwise for a normal time step
    // require the time step from the old DW.
    if(sched->get_dw(0) ) {
      task->requiresVar( Task::OldDW, VarLabel::find(timeStep_name) );
    }
    else if(sched->get_dw(1) ) {
      task->requiresVar( Task::NewDW, VarLabel::find(timeStep_name) );
    }

    task->requiresVar(Task::NewDW, d_lab->d_delTLabel, level.get_rep());

    const PatchSet* perProcPatchSet = sched->getLoadBalancer()->getPerProcessorPatchSet( level->getGrid() );

    sched->addTask(task, perProcPatchSet, d_materialManager->allMaterials());
  }
}

//--------------------------------------------------------------------------------------------------
void
ExplicitSolver::computeTaskGraphIndex(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // This method is called at both initialization and otherwise. At
  // initialization the old DW will not exist so get the value from
  // the new DW.  Otherwise for a normal time step get the time step
  // from the old DW.
  timeStep_vartype timeStep(0);

  if( old_dw && old_dw->exists( VarLabel::find(timeStep_name) ) )
    old_dw->get(timeStep, VarLabel::find(timeStep_name) );
  else if( new_dw && new_dw->exists( VarLabel::find(timeStep_name) ) )
    new_dw->get(timeStep, VarLabel::find(timeStep_name) );

  // Calculate the task graph index to be used on the NEXT time step.
  m_arches->setTaskGraphIndex(((timeStep+1) % d_rad_calc_frequency == 0) ?
                              1 :  // RMCRTCommon::TG_RMCRT :
                              0 ); // RMCRTCommon::TG_CARRY_FORWARD);
}

//--------------------------------------------------------------------------------------------------
void
ExplicitSolver::sched_initialize( const LevelP& level,
                                  SchedulerP & sched,
                                  const bool doing_restart )
{

  const MaterialSet* matls = d_lab->d_materialManager->allMaterials( "Arches" );

  setupBoundaryConditions( level, sched, false );

  d_tabulated_properties->set_bcHelper( m_bcHelper[level->getIndex()] );

  SourceTermFactory& srcFactory = SourceTermFactory::self();
  srcFactory.set_bcHelper( m_bcHelper[level->getIndex()]);

  if ( level->getIndex() == d_archesLevelIndex ){

    //formerly known as paramInit
    sched_initializeVariables( level, sched );

    // initialize hypre variables
    d_pressSolver->scheduleInitialize( level, sched, matls);

    //------------------ New Task Interface (start) ------------------------------------------------

    typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
    BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
    BFM::iterator i_trans_fac = _task_factory_map.find("transport_factory");
    BFM::iterator i_init_fac = _task_factory_map.find("initialize_factory");
    BFM::iterator i_partmod_fac = _task_factory_map.find("particle_model_factory");
    BFM::iterator i_lag_fac = _task_factory_map.find("lagrangian_factory");
    BFM::iterator i_property_models_fac = _task_factory_map.find("property_models_factory");

    i_trans_fac->second->set_bcHelper( m_bcHelper[level->getID()] );
    i_util_fac->second->set_bcHelper( m_bcHelper[level->getID()] );
    i_property_models_fac->second->set_bcHelper( m_bcHelper[level->getID()] );

    const bool dont_pack_tasks = false;
    TaskFactoryBase::TaskMap all_tasks;

    //utility factory
    // _task_factory_map["utility_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );
    _task_factory_map["utility_factory"]->schedule_task_group( "base_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

    //transport factory
    //  initialize
    _task_factory_map["transport_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

    //  apply BCs
    _task_factory_map["transport_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, dont_pack_tasks, level, sched, matls );

    // boundary condition factory
    _task_factory_map["boundary_condition_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

    i_lag_fac->second->schedule_task_group("all_tasks", TaskInterface::INITIALIZE,
      dont_pack_tasks, level, sched, matls );

    i_init_fac->second->schedule_task_group("all_tasks", TaskInterface::INITIALIZE,
      dont_pack_tasks, level, sched, matls );

    //------------------ New Task Interface (end) ------------------------------------------------

    sched_scalarInit( level, sched );

    //------------------ New Task Interface (start) ------------------------------------------------
    i_property_models_fac->second->schedule_initialization( level, sched, matls, false );
    //------------------ New Task Interface (end) ------------------------------------------------

    //(old) Property model initialization
    PropertyModelFactory& propFactory = PropertyModelFactory::self();
    PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++) {

      PropertyModelBase* prop_model = iprop->second;
      prop_model->sched_initialize( level, sched );
    }

    IntVector periodic_vector = level->getPeriodicBoundaries();
    bool d_3d_periodic = (periodic_vector == IntVector(1,1,1));
    d_turbModel->set3dPeriodic(d_3d_periodic);
    d_props->set3dPeriodic(d_3d_periodic);

    // Table Lookup
    bool initialize_it = true;
    bool modify_ref_den = true;
    int time_substep = 0; //no meaning here, but is required to be zero for
                          //variables to be properly allocated.

    d_tabulated_properties->sched_checkTableBCs( level, sched );
    d_tabulated_properties->sched_getState( level, sched, initialize_it, modify_ref_den, time_substep );

    d_init_timelabel = scinew TimeIntegratorLabel(d_lab, TimeIntegratorStepType::FE);

    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++) {
      PropertyModelBase* prop_model = iprop->second;
      if ( prop_model->initType()=="physical" )
        prop_model->sched_computeProp( level, sched, 1 );
    }

    //For debugging
    //d_boundaryCondition->printBCInfo();

    //Setup initial inlet velocities
    d_boundaryCondition->sched_setupBCInletVelocities( sched, level, matls, doing_restart,false );

    //Set the initial profiles
    d_boundaryCondition->sched_setInitProfile( sched, level, matls );

    //Setup the intrusions.
    d_boundaryCondition->sched_setupNewIntrusions( sched, level, matls );

    sched_setInitVelCond( level, sched, matls );

    sched_getCCVelocities(level, sched);

    if ( !d_using_TI_turb ){
      d_turbModel->sched_reComputeTurbSubmodel(sched, level, matls, d_init_timelabel);
    } else {
      _task_factory_map["turbulence_model_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );
    }

    //----------------------
    //DQMOM initialization
    if (d_doDQMOM) {

      if ( !d_kokkos_dqmom_Translate ){

        sched_weightInit(level, sched);
        sched_weightedAbsInit(level, sched);

        // check to make sure that all dqmom equations have BCs set.
        DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();
        DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns();
        for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++) {
          EqnBase* eqn = ieqn->second;
          eqn->sched_checkBCs( level, sched, false );
          //as needed for the coal propery models
          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(ieqn->second);
          dqmom_eqn->sched_getUnscaledValues( level, sched );
        }
      } else {

        // Models
        // initialize all of the computed variables for the coal models
        CoalModelFactory& modelFactory = CoalModelFactory::self();
        modelFactory.sched_init_all_models( level, sched );
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
      for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin(); ieqn != cqmom_eqns.end(); ieqn++) {
        EqnBase* eqn = ieqn->second;
        eqn->sched_checkBCs( level, sched,false );
      }
      //call the cqmom inversion so weights and abscissas are calculated at the start
      d_cqmomSolver->sched_solveCQMOMInversion( level, sched, 0 );
    }

    //------------------ New Task Interface (start) ------------------------------------------------
    i_partmod_fac->second->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE,
                                                dont_pack_tasks, level, sched, matls );

    //------------------ New Task Interface (end) ------------------------------------------------

    // check to make sure that all the scalar variables have BCs set and set intrusions:
    EqnFactory& eqnFactory = EqnFactory::self();
    EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
    for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++) {
      EqnBase* eqn = ieqn->second;
      eqn->sched_checkBCs( level, sched,false );

      // also, do table initialization here since all scalars should be initialized by now
      if (eqn->does_table_initialization()) {
        eqn->sched_tableInitialization( level, sched );
      }
    }

    d_boundaryCondition->sched_setIntrusionTemperature( sched, level, matls );

    d_boundaryCondition->sched_create_radiation_temperature( sched, level, matls, doing_restart, false );

    // Utility Factory : Balance Terms computation
    _task_factory_map["utility_factory"]->schedule_task_group( "mass_flow_rate", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );
  }
}

void
ExplicitSolver::sched_initializeVariables( const LevelP& level,
                                           SchedulerP& sched )
{

  Task* tsk = scinew Task( "ExplicitSolver::initializeVariables", this, &ExplicitSolver::initializeVariables);

  tsk->computesVar(d_lab->d_cellInfoLabel);
  tsk->computesVar(d_lab->d_uVelocitySPBCLabel);
  tsk->computesVar(d_lab->d_vVelocitySPBCLabel);
  tsk->computesVar(d_lab->d_wVelocitySPBCLabel);
  tsk->computesVar(d_lab->d_uVelocityLabel);
  tsk->computesVar(d_lab->d_vVelocityLabel);
  tsk->computesVar(d_lab->d_wVelocityLabel);
  tsk->computesVar(d_lab->d_uVelRhoHatLabel);
  tsk->computesVar(d_lab->d_vVelRhoHatLabel);
  tsk->computesVar(d_lab->d_wVelRhoHatLabel);
  tsk->computesVar(d_lab->d_CCVelocityLabel);
  tsk->computesVar(d_lab->d_CCUVelocityLabel);
  tsk->computesVar(d_lab->d_CCVVelocityLabel);
  tsk->computesVar(d_lab->d_CCWVelocityLabel);
  tsk->computesVar(d_lab->d_pressurePSLabel);
  tsk->computesVar(d_lab->d_densityGuessLabel);
  tsk->computesVar(d_lab->d_totalKineticEnergyLabel);
  tsk->computesVar(d_lab->d_kineticEnergyLabel);
  tsk->computesVar(d_lab->d_pressurePredLabel);
  tsk->computesVar(d_lab->d_pressureIntermLabel);
  tsk->computesVar(d_lab->d_densityCPLabel);
  if ( !d_using_TI_turb ){
    tsk->computesVar(d_lab->d_viscosityCTSLabel);
    tsk->computesVar(d_lab->d_turbViscosLabel);
  }
  tsk->computesVar(d_lab->d_oldDeltaTLabel);
  tsk->computesVar(d_lab->d_conv_scheme_x_Label);
  tsk->computesVar(d_lab->d_conv_scheme_y_Label);
  tsk->computesVar(d_lab->d_conv_scheme_z_Label);

  if ( VarLabel::find("true_wall_temperature"))
    tsk->computesVar(VarLabel::find("true_wall_temperature"));

  if ( VarLabel::find("deposit_thickness"))
    tsk->computesVar(VarLabel::find("deposit_thickness"));
  if ( VarLabel::find("deposit_thickness_sb_s"))
    tsk->computesVar(VarLabel::find("deposit_thickness_sb_s"));
  if ( VarLabel::find("deposit_thickness_sb_l"))
    tsk->computesVar(VarLabel::find("deposit_thickness_sb_l"));
  if ( VarLabel::find("emissivity"))
    tsk->computesVar(VarLabel::find("emissivity"));
  if ( VarLabel::find("thermal_cond_en"))
    tsk->computesVar(VarLabel::find("thermal_cond_en"));
  if ( VarLabel::find("thermal_cond_sb_s"))
    tsk->computesVar(VarLabel::find("thermal_cond_sb_s"));
  if ( VarLabel::find("thermal_cond_sb_l"))
    tsk->computesVar(VarLabel::find("thermal_cond_sb_l"));

sched->addTask(tsk, level->eachPatch(), d_lab->d_materialManager->allMaterials( "Arches" ));

}

void
ExplicitSolver::sched_restartInitialize( const LevelP& level, SchedulerP& sched )
{
  bool doingRestart = true;

  const MaterialSet* matls = d_lab->d_materialManager->allMaterials( "Arches" );

  typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;

  _task_factory_map["property_models_factory"]->schedule_task_group("all_tasks", TaskInterface::RESTART_INITIALIZE,
    false, level, sched, matls );

  setupBoundaryConditions( level, sched, doingRestart );

  d_tabulated_properties->set_bcHelper( m_bcHelper[level->getIndex()]);

  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
  i_util_fac->second->set_bcHelper( m_bcHelper[level->getIndex()]);

  //Arches only currently solves on the finest level
  if ( !level->hasFinerLevel() ){

    // initialize hypre variables
    d_pressSolver->scheduleRestartInitialize( level, sched, matls);

    d_boundaryCondition->sched_setupBCInletVelocities( sched, level, matls, doingRestart, false);

    d_boundaryCondition->sched_create_radiation_temperature( sched, level, matls, doingRestart, false );

    //__________________________________
    //  initialize src terms
    SourceTermFactory& srcFactory = SourceTermFactory::self();
    SourceTermFactory::SourceMap& sources = srcFactory.retrieve_all_sources();
    for (SourceTermFactory::SourceMap::iterator isrc=sources.begin(); isrc !=sources.end(); isrc++){
      SourceTermBase* src = isrc->second;
      src->sched_restartInitialize(level, sched);
    }
    srcFactory.set_bcHelper( m_bcHelper[level->getIndex()]); 

    //__________________________________
    //  initialize property models
    PropertyModelFactory& propFactory = PropertyModelFactory::self();
    PropertyModelFactory::PropMap& properties = propFactory.retrieve_all_property_models();
    for (PropertyModelFactory::PropMap::iterator iprop=properties.begin(); iprop !=properties.end(); iprop++){
      PropertyModelBase* prop = iprop->second;
      prop->sched_restartInitialize(level, sched);
    }

    EqnFactory& eqnFactory = EqnFactory::self();
    EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
    for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++) {
      EqnBase* eqn = ieqn->second;
      eqn->sched_checkBCs( level, sched,false );
    }

    // check to make sure that all dqmom equations have BCs set.
    DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns();
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++) {
      EqnBase* eqn = ieqn->second;
      eqn->sched_checkBCs( level, sched,false );
    }

    checkMomBCs( sched, level, matls );

    d_boundaryCondition->sched_setupNewIntrusionCellType( sched, level, matls, doingRestart );

    d_boundaryCondition->sched_setupNewIntrusions( sched, level, matls );

    //turbulence models
    _task_factory_map["turbulence_model_factory"]->schedule_task_group("all_tasks", TaskInterface::RESTART_INITIALIZE,
                                                                        false, level, sched, matls );

  }

  // When running with radiation compute the next task graph index
  // using a fixed time stepping. Not used when doing dynamic time
  // stepping.
  if( d_perform_radiation && !d_dynamicSolveFrequency ) {
    sched_computeTaskGraphIndex( level, sched );
  }
}

void
ExplicitSolver::sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched )
{

  bool doingRegrid  = true;
  const MaterialSet* matls = d_lab->d_materialManager->allMaterials( "Arches" );

  d_boundaryCondition->sched_setupBCInletVelocities( sched, level, matls, false, doingRegrid);

  EqnFactory& eqnFactory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++) {
    EqnBase* eqn = ieqn->second;
    eqn->sched_checkBCs( level, sched,doingRegrid );
  }

  // check to make sure that all dqmom equations have BCs set.
  DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self();
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmom_factory.retrieve_all_eqns();
  for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++) {
    EqnBase* eqn = ieqn->second;
    eqn->sched_checkBCs( level, sched ,doingRegrid);
  }

  checkMomBCs( sched, level, matls );

}


void
ExplicitSolver::initializeVariables(const ProcessorGroup* ,
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{

  double old_delta_t = 0.0;
  new_dw->put(delt_vartype(old_delta_t), d_lab->d_oldDeltaTLabel);

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    //const Level* level = patch->getLevel();
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    // Initialize cellInformation
    PerPatch<CellInformationP> cellInfoP;
    cellInfoP.setData(scinew CellInformation(patch));
    new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);

    //total KE:
    new_dw->put( sum_vartype(0.0), d_lab->d_totalKineticEnergyLabel );

    allocateAndInitializeToC( d_lab->d_densityGuessLabel  , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_uVelRhoHatLabel    , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_uVelocitySPBCLabel , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_vVelRhoHatLabel    , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_vVelocitySPBCLabel , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_wVelRhoHatLabel    , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_wVelocitySPBCLabel , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_uVelocityLabel     , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_vVelocityLabel     , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_wVelocityLabel     , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_CCUVelocityLabel   , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_CCVVelocityLabel   , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_CCWVelocityLabel   , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_CCVelocityLabel    , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_kineticEnergyLabel , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_pressurePSLabel    , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_pressurePredLabel  , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_pressureIntermLabel, new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_densityCPLabel     , new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_viscosityCTSLabel  , new_dw, indx, patch, 0.0 );
    if ( !d_using_TI_turb ){
      allocateAndInitializeToC( d_lab->d_turbViscosLabel    , new_dw, indx, patch, 0.0 );
      allocateAndInitializeToC( d_lab->d_conv_scheme_x_Label, new_dw, indx, patch, 0.0 );
    }
    allocateAndInitializeToC( d_lab->d_conv_scheme_y_Label, new_dw, indx, patch, 0.0 );
    allocateAndInitializeToC( d_lab->d_conv_scheme_z_Label, new_dw, indx, patch, 0.0 );

    if ( VarLabel::find("true_wall_temperature")){
      allocateAndInitializeToC( VarLabel::find("true_wall_temperature"), new_dw, indx, patch, 0.0 );
    }

    if ( VarLabel::find("deposit_thickness"))
      allocateAndInitializeToC( VarLabel::find("deposit_thickness"), new_dw, indx, patch, 0.0 );
    if ( VarLabel::find("deposit_thickness_sb_s"))
      allocateAndInitializeToC( VarLabel::find("deposit_thickness_sb_s"), new_dw, indx, patch, 0.0 );
    if ( VarLabel::find("deposit_thickness_sb_l"))
      allocateAndInitializeToC( VarLabel::find("deposit_thickness_sb_l"), new_dw, indx, patch, 0.0 );
    if ( VarLabel::find("emissivity"))
      allocateAndInitializeToC( VarLabel::find("emissivity"), new_dw, indx, patch, 1.0 );
    if ( VarLabel::find("thermal_cond_en"))
      allocateAndInitializeToC( VarLabel::find("thermal_cond_en"), new_dw, indx, patch, 1.0 );
    if ( VarLabel::find("thermal_cond_sb_s"))
      allocateAndInitializeToC( VarLabel::find("thermal_cond_sb_s"), new_dw, indx, patch, 1.0 );
    if ( VarLabel::find("thermal_cond_sb_l"))
      allocateAndInitializeToC( VarLabel::find("thermal_cond_sb_l"), new_dw, indx, patch, 1.0 );

    CCVariable<double> viscosity;
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, indx, patch);
    double visVal = d_physicalConsts->getMolecularViscosity();
    viscosity.initialize(visVal);

  } // patches
}

void
ExplicitSolver::allocateAndInitializeToC( const VarLabel* label,
                                          DataWarehouse* dw,
                                          const int index,
                                          const Patch* patch, const double C ){

  const Uintah::TypeDescription* type_desc = label->typeDescription();

  if ( type_desc == CCVariable<double>::getTypeDescription() ){
    CCVariable<double> var;
    dw->allocateAndPut( var, label, index, patch );
    var.initialize(C);
  } else if ( type_desc == CCVariable<int>::getTypeDescription() ){
    CCVariable<int> var;
    dw->allocateAndPut( var, label, index, patch );
    var.initialize(C);
  } else if ( type_desc == CCVariable<Vector>::getTypeDescription() ){
    CCVariable<Vector> var;
    dw->allocateAndPut( var, label, index, patch );
    var.initialize(Vector(0.,0.,0.));
  } else if ( type_desc == SFCXVariable<double>::getTypeDescription() ){
    SFCXVariable<double> var;
    dw->allocateAndPut( var, label, index, patch );
    var.initialize(0.0);
  } else if ( type_desc == SFCYVariable<double>::getTypeDescription() ){
    SFCYVariable<double> var;
    dw->allocateAndPut( var, label, index, patch );
    var.initialize(0.0);
  } else if ( type_desc == SFCZVariable<double>::getTypeDescription() ){
    SFCZVariable<double> var;
    dw->allocateAndPut( var, label, index, patch );
    var.initialize(0.0);
  } else {
    throw InvalidValue("Error: Type not supported.",__FILE__,__LINE__);
  }
}

void
ExplicitSolver::checkMomBCs( SchedulerP& sched,
                             const LevelP& level,
                             const MaterialSet* matls)
{

  d_boundaryCondition->sched_checkMomBCs( sched, level, matls );

}

// ****************************************************************************
// Schedule non linear solve and carry out some actual operations
// ****************************************************************************
int ExplicitSolver::sched_nonlinearSolve(const LevelP& level,
                                         SchedulerP& sched)
{

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_materialManager->allMaterials( "Arches" );
  IntVector periodic_vector = level->getPeriodicBoundaries();
  d_3d_periodic = (periodic_vector == IntVector(1,1,1));
  d_turbModel->set3dPeriodic(d_3d_periodic);
  d_props->set3dPeriodic(d_3d_periodic);

  sched_setInitialGuess(sched, patches, matls);

  d_boundaryCondition->sched_setAreaFraction(sched, level, matls, 0, false );
  d_turbModel->sched_carryForwardFilterVol(sched, patches, matls);

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  if (dqmomFactory.get_quad_nodes() > 0)
    d_doDQMOM = true;
  else
    d_doDQMOM = false; // probably need to sync this better with the bool being set in Arches

  CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
  if (cqmomFactory.get_number_moments() > 0)
    d_doCQMOM = true;
  else
    d_doCQMOM = false;


  EqnFactory& eqn_factory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqn_factory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

    EqnBase* eqn = iter->second;
    eqn->sched_initializeVariables(level, sched);

  }

  //-------- carry forward the geometry modifier variable --------
  d_boundaryCondition->sched_computeAlphaG( sched, level, matls, true );

  //copy the temperature into a radiation temperature variable:
  d_boundaryCondition->sched_create_radiation_temperature( sched, level, matls, false, true );

  if ( d_wall_ht_models != nullptr ){
    d_wall_ht_models->sched_doWallHT( level, sched, 0 );
  }

  //------------------ New Task Interface (start) --------------------------------------------------

  typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
  const bool dont_pack_tasks = false;

  BFM::iterator i_util = _task_factory_map.find("utility_factory");
  i_util->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE, dont_pack_tasks, level, sched,
    matls );

  BFM::iterator i_transport = _task_factory_map.find("transport_factory");
  i_transport->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE, dont_pack_tasks, level, sched,
    matls );

  BFM::iterator i_property_models = _task_factory_map.find("property_models_factory");
  i_property_models->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE, dont_pack_tasks, level, sched,
    matls );

  BFM::iterator i_particle_models = _task_factory_map.find("particle_model_factory");
  i_particle_models->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
                                                  dont_pack_tasks, level, sched,
                                                  matls );

  _task_factory_map["turbulence_model_factory"]->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE, dont_pack_tasks, level, sched,
    matls );

  BFM::iterator i_bc_fac = _task_factory_map.find("boundary_condition_factory");

  //------------------ New Task Interface (end) --------------------------------------------------

  // --------> START RK LOOP <---------
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
  {

    //------------------ New Task Interface (start) ------------------------------------------------

    using namespace ArchesCore;
    TaskController& tsk_controller = TaskController::self();
    const TaskController::Packing& packed_info = tsk_controller.get_packing_info();

    _task_factory_map["property_models_factory"]->schedule_task_group( "pre_update_property_models",
      TaskInterface::TIMESTEP_EVAL, dont_pack_tasks, level, sched, matls, curr_level );

    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::TIMESTEP_EVAL, dont_pack_tasks, level, sched, matls, curr_level );

    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::BC, dont_pack_tasks, level, sched, matls, curr_level );

    i_transport->second->schedule_task_group( "momentum_conv", TaskInterface::TIMESTEP_EVAL,
      dont_pack_tasks, level, sched, matls, curr_level );

    i_transport->second->schedule_task_group( "momentum_fe_update", TaskInterface::TIMESTEP_EVAL,
      dont_pack_tasks, level, sched, matls, curr_level );

    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::BC, packed_info.global, level, sched, matls, curr_level );
    // bc factory tasks
    i_bc_fac->second->schedule_task_group("all_tasks",
      TaskInterface::BC, packed_info.global, level, sched, matls, curr_level );

    //------------------ New Task Interface (end) ------------------------------------------------

    // Create this timestep labels for properties
    PropertyModelFactory& propFactory = PropertyModelFactory::self();
    PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
        iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;

      if ( curr_level == 0 )
        prop_model->sched_timeStepInit( level, sched );

    }

    if (d_doDQMOM) {

      // Compute the particle velocities at time t w^tu^t/w^t
      d_partVel->schedComputePartVel( level, sched, curr_level );
      CoalModelFactory& modelFactory = CoalModelFactory::self();

      // Evaluate DQMOM equations
      if ( !d_kokkos_dqmom_Translate ){

        DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
        DQMOMEqnFactory::EqnMap& weights_eqns = dqmomFactory.retrieve_weights_eqns();
        DQMOMEqnFactory::EqnMap& abscissas_eqns = dqmomFactory.retrieve_abscissas_eqns();

        for ( DQMOMEqnFactory::EqnMap::iterator iEqn = weights_eqns.begin();
              iEqn != weights_eqns.end(); iEqn++){

          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

          dqmom_eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::DOALL );//compute rhs
        }

        for ( DQMOMEqnFactory::EqnMap::iterator iEqn = abscissas_eqns.begin();
              iEqn != abscissas_eqns.end(); iEqn++){

          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

          dqmom_eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::DOALL );//compute rhs
        }


        i_particle_models->second->schedule_task_group( "dqmom_transport_variables",
                                                         TaskInterface::TIMESTEP_EVAL,
                                                         dont_pack_tasks, level, sched,
                                                         matls, curr_level );
        // schedule the models for evaluation
        modelFactory.sched_coalParticleCalculation( level, sched, curr_level );// compute drag, devol, char, etc models..

        i_particle_models->second->schedule_task_group( "particle_models",
                                                         TaskInterface::TIMESTEP_EVAL,
                                                         dont_pack_tasks, level, sched,
                                                         matls, curr_level );

        // schedule DQMOM linear solve
        if ( m_DQMOMSolverType != "No_Inversion"){

          d_dqmomSolver->sched_solveLinearSystem( level, sched, curr_level );

        } else {

          i_particle_models->second->schedule_task( "[DQMOMNoInversion]",
                                                    TaskInterface::TIMESTEP_EVAL,
                                                    level, sched,
                                                    matls, curr_level);

        }

        // Evaluate DQMOM equations
        for ( DQMOMEqnFactory::EqnMap::iterator iEqn = weights_eqns.begin();
              iEqn != weights_eqns.end(); iEqn++){

          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

          dqmom_eqn->sched_updateTransportEqn( level, sched, curr_level );// add sources and solve equation
        }

        for ( DQMOMEqnFactory::EqnMap::iterator iEqn = abscissas_eqns.begin();
              iEqn != abscissas_eqns.end(); iEqn++){

          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

          dqmom_eqn->sched_updateTransportEqn( level, sched, curr_level );// add sources and solve equation
        }

        for( DQMOMEqnFactory::EqnMap::iterator iEqn = dqmom_eqns.begin();
             iEqn!=dqmom_eqns.end(); ++iEqn ) {

          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

          //get the abscissa values
          dqmom_eqn->sched_getUnscaledValues( level, sched );
        }

        // calculate the moments
        bool saveMoments = d_dqmomSolver->getSaveMoments();
        if( saveMoments ) {
          // schedule DQMOM moment calculation
          d_dqmomSolver->sched_calculateMoments( level, sched, curr_level );
        }

      } else {

        const int time_substep = curr_level;

        // schedule the models for evaluation
        modelFactory.sched_coalParticleCalculation( level, sched, curr_level );// compute drag, devol, char, etc models..

        i_particle_models->second->schedule_task_group( "part_face_velocities",
                                                         TaskInterface::TIMESTEP_EVAL,
                                                         dont_pack_tasks, level, sched,
                                                         matls, time_substep);

        i_particle_models->second->schedule_task_group( "drag_model_task",
                                                         TaskInterface::TIMESTEP_EVAL,
                                                         dont_pack_tasks, level, sched,
                                                         matls, time_substep);

        i_particle_models->second->schedule_task_group( "pre_update_property_models",
                                                        TaskInterface::TIMESTEP_EVAL,
                                                        dont_pack_tasks, level, sched,
                                                        matls , time_substep);

        i_transport->second->schedule_task_group("dqmom_diffusion_flux_builders",
          TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

        i_transport->second->schedule_task_group("dqmom_eqns",
          TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

        i_transport->second->schedule_task_group("dqmom_eqns",
          TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

        i_transport->second->schedule_task_group("dqmom_fe_update",
          TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

        // computes ic from w*ic  :
        i_transport->second->schedule_task_group("dqmom_ic_from_wic",
          TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

        i_transport->second->schedule_task_group("dqmom_ic_from_wic",
          TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

        i_transport->second->schedule_task_group("dqmom_fe_update",
          TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );
      }

    }

    if ( d_doCQMOM ) {
      bool doOperatorSplit;
      bool doPartVel;
      bool doSourceTerms;

      doOperatorSplit = d_cqmomSolver->getOperatorSplitting();
      doPartVel = d_cqmomSolver->getPartVel();
      doSourceTerms = d_cqmomSource->getAddSources();
      CQMOMEqnFactory::EqnMap& moment_eqns = cqmomFactory.retrieve_all_eqns();

      if ( doSourceTerms ) {
        if (curr_level == 0)
          d_cqmomSource->sched_initializeVariables( level, sched );
        d_cqmomSource->sched_buildSourceTerm( level, sched, curr_level );
      }

      if (!doOperatorSplit) {
      //Evaluate CQMOM equations
        if (doPartVel) {
          if (curr_level == 0 )
            d_cqmomConvect->sched_initializeVariables( level, sched );
          d_cqmomConvect->sched_initializeWalls( level, sched, curr_level );
          d_cqmomConvect->sched_solveCQMOMConvection( level, sched, curr_level);
        }

        for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
             iEqn != moment_eqns.end(); iEqn++){

          CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
          cqmom_eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::DOALL );

        }
        //get new weights and absicissa
        d_cqmomSolver->sched_solveCQMOMInversion( level, sched, curr_level );
        d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
      } else {
        //if operator splitting is turned on use a different CQMOM permutation for each convection direction
        int uVelIndex = d_cqmomSolver->getUVelIndex();
        int vVelIndex = d_cqmomSolver->getVVelIndex();
        int wVelIndex = d_cqmomSolver->getWVelIndex();

        for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
             iEqn != moment_eqns.end(); iEqn++){
          CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
          if (curr_level == 0)
            cqmom_eqn->sched_initializeVariables( level, sched );
          cqmom_eqn->sched_computeSources( level, sched, curr_level );
        }
        //x-direction - do the CQMOM inversion of this permutation, then do the convection
        if ( uVelIndex > -1 ) {
          d_cqmomSolver->sched_solveCQMOMInversion321( level, sched, curr_level );
          d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
          for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
               iEqn != moment_eqns.end(); iEqn++){

            CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
            cqmom_eqn->sched_buildXConvection( level, sched, curr_level );
          }
        }
        //y-direction
        if ( vVelIndex > -1 ) {
          d_cqmomSolver->sched_solveCQMOMInversion312( level, sched, curr_level );
          d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
          for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
               iEqn != moment_eqns.end(); iEqn++){

            CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
            cqmom_eqn->sched_buildYConvection( level, sched, curr_level );
          }
        }
        //z-direction
        if ( wVelIndex > -1 ) {
          d_cqmomSolver->sched_solveCQMOMInversion213( level, sched, curr_level );
          d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
          for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
               iEqn != moment_eqns.end(); iEqn++){

            CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
            cqmom_eqn->sched_buildZConvection( level, sched, curr_level );
          }
        }

        //combine all 3 fluxes and actually solve eqn with other sources
        for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
             iEqn != moment_eqns.end(); iEqn++){

          CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
          cqmom_eqn->sched_buildSplitRHS( level, sched, curr_level );
          cqmom_eqn->sched_solveTransportEqn( level, sched, curr_level );
        }
      }
    }

    //Particle Properties dependent on the latest DW IC's
    //All particle transport should draw from the "latest" DW (old for rk = 0, new for rk > 0)
    i_particle_models->second->schedule_task_group("particle_properties", TaskInterface::TIMESTEP_EVAL,
      dont_pack_tasks, level, sched, matls, curr_level );

    // Deposition (called out specifically because it has a specified order)
    i_particle_models->second->schedule_task_group("deposition_models", TaskInterface::TIMESTEP_EVAL,
      dont_pack_tasks, level, sched, matls, curr_level );


    // STAGE 0
    //these equations use a density guess

    sched_saveTempCopies( sched, patches, matls,d_timeIntegratorLabels[curr_level]);

    sched_getDensityGuess( sched, patches, matls,
                           d_timeIntegratorLabels[curr_level] );

    //------------------ New Task Interface (start) ------------------------------------------------
    // i_property_models->second->schedule_task_group("density_predictor", TaskInterface::TIMESTEP_EVAL,
    //   dont_pack_tasks, level, sched, matls, curr_level );
    //------------------ New Task Interface (end) ------------------------------------------------


    sched_checkDensityGuess(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level]);

    for (auto iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;

      if ( eqn->get_stage() == 0 )
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::CONVDIFF );

    }

    SourceTermFactory& src_factory = SourceTermFactory::self();

    src_factory.sched_computeSources( level, sched, curr_level, 0 );

    for (auto iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() == 0 ){
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::ADDSOURCES );
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::UPDATE );
      }

    }


    // Property models needed before table lookup:
    PropertyModelBase* hl_model = 0;
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;

      if ( prop_model->getPropType() == "heat_loss" ){
        hl_model = prop_model;
      } else {
        if ( prop_model->beforeTableLookUp() )
          prop_model->sched_computeProp( level, sched, curr_level );
      }

    }

    if ( hl_model != nullptr )
      hl_model->sched_computeProp( level, sched, curr_level );

    //------------------ New Task Interface (start) ------------------------------------------------
    i_property_models->second->schedule_task_group("pre_table_post_iv_update", TaskInterface::TIMESTEP_EVAL,
      dont_pack_tasks, level, sched, matls, curr_level );
    //------------------ New Task Interface (end) ------------------------------------------------

    //1st TABLE LOOKUP
    bool initialize_it  = false;
    bool modify_ref_den = false;
    if ( curr_level == 0 ) initialize_it = true;
    //d_props->sched_computeProps( level, sched, initialize_it, modify_ref_den, curr_level );

    d_tabulated_properties->sched_getState( level, sched, initialize_it, modify_ref_den, curr_level );

    d_boundaryCondition->sched_setIntrusionTemperature( sched, level, matls );

    // STAGE 1

    // Property models needed after table lookup:
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;
      if ( !prop_model->beforeTableLookUp() )
        prop_model->sched_computeProp( level, sched, curr_level );

    }

    for (auto iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;

      if ( eqn->get_stage() == 1 )
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::CONVDIFF );

    }

    src_factory.sched_computeSources( level, sched, curr_level, 1 );

    for (auto iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() == 1 ){
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::ADDSOURCES );
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::UPDATE );
      }

    }

    sched_computeDensityLag(sched, patches, matls,
                           d_timeIntegratorLabels[curr_level],
                           false);

    if (d_maxDensityLag > 0.0)
      sched_checkDensityLag(sched, patches, matls,
                            d_timeIntegratorLabels[curr_level],
                            false);

    // linearizes and solves pressure eqn
    // first computes, hatted velocities and then computes
    // the pressure poisson equation

//    _task_factory_map["turbulence_model_factory"]->schedule_task_group( "momentum_closure",
//      TaskInterface::TIMESTEP_EVAL, dont_pack_tasks, level, sched, matls, curr_level );

    d_momSolver->solveVelHat(level, sched, d_timeIntegratorLabels[curr_level], curr_level );

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() < 2 )
        eqn->sched_timeAve( level, sched, curr_level );
    }

    // averaging for RKSSP
    if ((curr_level>0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {

      //currently averaging density guess as
      // rho_guess = alpha*old_den + beta*new_den
      // not sure that this is used anywhere now
      d_props->sched_averageRKProps(sched, patches, matls,
                                    d_timeIntegratorLabels[curr_level]);

      //copies current density into temp density
      d_props->sched_saveTempDensity(sched, patches, matls,
                                     d_timeIntegratorLabels[curr_level]);

      // Property models before table lookup
      PropertyModelBase* hl_model = 0;
      for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
            iprop != all_prop_models.end(); iprop++){

        PropertyModelBase* prop_model = iprop->second;

        if ( prop_model->getPropType() == "heat_loss" ){
          hl_model = prop_model;
        } else {
          if ( prop_model->beforeTableLookUp() )
            prop_model->sched_computeProp( level, sched, curr_level );
        }

      }
      if ( hl_model != nullptr ) {
        hl_model->sched_computeProp( level, sched, curr_level );
      }
      //TABLE LOOKUP #2
      bool initialize_it  = false;
      bool modify_ref_den = false;
      //d_props->sched_computeProps( level, sched, initialize_it, modify_ref_den, curr_level );
      d_tabulated_properties->sched_getState( level, sched, initialize_it, modify_ref_den, curr_level );

      // Property models after table lookup
      for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin(); iprop != all_prop_models.end(); iprop++){

        PropertyModelBase* prop_model = iprop->second;
        if ( !prop_model->beforeTableLookUp() )
          prop_model->sched_computeProp( level, sched, curr_level );

      }
    }

    //STAGE 2

    for (auto iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;

      if ( eqn->get_stage() == 2 )
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::CONVDIFF );

    }

    src_factory.sched_computeSources( level, sched, curr_level, 2 );

    for (auto iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() == 2 ){
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::ADDSOURCES );
        eqn->sched_evalTransportEqn( level, sched, curr_level, EqnBase::UPDATE );
      }

    }

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() == 2 )
        eqn->sched_timeAve( level, sched, curr_level );
    }

    if ( d_extra_table_lookup ){
      //TABLE LOOKUP #3
      initialize_it  = false;
      modify_ref_den = false;
      d_tabulated_properties->sched_getState( level, sched, initialize_it, modify_ref_den, curr_level );
    }

    if ((curr_level>0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {

      sched_computeDensityLag(sched, patches, matls,
                              d_timeIntegratorLabels[curr_level],
                              true);
      if (d_maxDensityLag > 0.0)
        sched_checkDensityLag(sched, patches, matls,
                              d_timeIntegratorLabels[curr_level],
                              true);

      d_momSolver->sched_averageRKHatVelocities(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level] );

    }

    d_boundaryCondition->sched_setIntrusionTemperature( sched, level, matls );

    if ( d_wall_ht_models != nullptr ){
      d_wall_ht_models->sched_copyWallTintoT( level, sched );
    }

    d_boundaryCondition->sched_setIntrusionDensity( sched, level, matls );

    d_props->sched_computeDrhodt(sched, patches, matls,
                                 d_timeIntegratorLabels[curr_level]);

    d_pressSolver->sched_solve(level, sched, d_timeIntegratorLabels[curr_level],
                               false, curr_level);

    // project velocities using the projection step
    d_momSolver->solve(sched, patches, matls,
                       d_timeIntegratorLabels[curr_level], false);


    // Schedule an interpolation of the face centered velocity data
    sched_interpolateFromFCToCC(sched, patches, matls, d_timeIntegratorLabels[curr_level], curr_level);

    if (d_mixedModel) {
      d_scaleSimilarityModel->sched_reComputeTurbSubmodel( sched, level, matls,
                                                           d_timeIntegratorLabels[curr_level]);
    }

    timeStep_vartype timeStep(0);
    if( sched->get_dw(0) && sched->get_dw(0)->exists( d_lab->d_timeStepLabel ) )
      sched->get_dw(0)->get( timeStep, d_lab->d_timeStepLabel );
    else if( sched->get_dw(1) && sched->get_dw(1)->exists( d_lab->d_timeStepLabel ) )
      sched->get_dw(1)->get( timeStep, d_lab->d_timeStepLabel );

    d_turbCounter = timeStep;

    if ((d_turbCounter%d_turbModelCalcFreq == 0)&&
        ((curr_level==0)||((!(curr_level==0))&&d_turbModelRKsteps)))
      d_turbModel->sched_reComputeTurbSubmodel(sched, level, matls,
                                               d_timeIntegratorLabels[curr_level]);

    _task_factory_map["turbulence_model_factory"]->schedule_task_group("all_tasks", TaskInterface::TIMESTEP_EVAL,
      dont_pack_tasks, level, sched, matls, curr_level );


  } // END OF RK LOOP

  //variable math:
  const std::vector<std::string> math_tasks = i_util->second->retrieve_tasks_by_type("variable_math");
  i_util->second->schedule_task_group("all_math_tasks", math_tasks, TaskInterface::TIMESTEP_EVAL,
    dont_pack_tasks,
    level, sched, matls, 0 );

  // massFlowRate
  const std::vector<std::string> flowRate_tasks = i_util->second->retrieve_tasks_by_type("mass_flow_rate");
  i_util->second->schedule_task_group("all_flowRate_tasks", flowRate_tasks, TaskInterface::TIMESTEP_EVAL,
    dont_pack_tasks,
    level, sched, matls, 0 );

  //Property Models before starting over
  i_property_models->second->schedule_task_group("final_property_models",
    TaskInterface::TIMESTEP_EVAL, dont_pack_tasks,
    level, sched, matls, 1 );

  //Wall HF
  std::vector<std::string> wall_hf_tasks =
    i_property_models->second->retrieve_tasks_by_type("wall_heatflux_variable");
  i_property_models->second->schedule_task_group("wall_hf_tasks", wall_hf_tasks, TaskInterface::TIMESTEP_EVAL,
    dont_pack_tasks,
    level, sched, matls );

  //Variable stats stuff
  i_property_models->second->schedule_task_group("variable_stat_models",
    TaskInterface::TIMESTEP_EVAL, dont_pack_tasks,
    level, sched, matls, 1 );

  if ( d_printTotalKE ){
   sched_computeKE( sched, patches, matls );
   sched_printTotalKE( sched, patches, matls );
  }

  if ( _doLagrangianParticles ) {

    //UPDATE SIZE
    _task_factory_map["lagrangian_factory"]->schedule_task( "update_particle_size", TaskInterface::TIMESTEP_EVAL, level, sched, matls, 0 );
    _task_factory_map["lagrangian_factory"]->schedule_task( "update_particle_position", TaskInterface::TIMESTEP_EVAL, level, sched, matls, 0 );
    _task_factory_map["lagrangian_factory"]->schedule_task( "update_particle_velocity", TaskInterface::TIMESTEP_EVAL, level, sched, matls, 0 );

    _particlesHelper->schedule_sync_particle_position(level,sched);
    _particlesHelper->schedule_transfer_particle_ids(level,sched);
    _particlesHelper->schedule_relocate_particles(level,sched);
    _particlesHelper->schedule_add_particles(level, sched);

  }

  return 0;

}

// ****************************************************************************
// Schedule initialize
// ****************************************************************************
void
ExplicitSolver::sched_setInitialGuess(       SchedulerP  & sched,
                                       const PatchSet    * patches,
                                       const MaterialSet * matls )
{
  //copies old db to new_db and then uses non-linear
  //solver to compute new values
  Task* tsk = scinew Task( "ExplicitSolver::setInitialGuess",this, &ExplicitSolver::setInitialGuess);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requiresVar(Task::OldDW, d_lab->d_cellTypeLabel,      gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_wVelocitySPBCLabel, gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_viscosityCTSLabel,  gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_turbViscosLabel,  gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_CCVelocityLabel, gn, 0);
  tsk->requiresVar(Task::OldDW, d_lab->d_densityGuessLabel,  gn, 0);

  tsk->computesVar(d_lab->d_cellInfoLabel);
  tsk->computesVar(d_lab->d_cellTypeLabel);
  tsk->computesVar(d_lab->d_uVelocitySPBCLabel);
  tsk->computesVar(d_lab->d_vVelocitySPBCLabel);
  tsk->computesVar(d_lab->d_wVelocitySPBCLabel);
  tsk->computesVar(d_lab->d_uVelocityLabel);
  tsk->computesVar(d_lab->d_vVelocityLabel);
  tsk->computesVar(d_lab->d_wVelocityLabel);
  tsk->computesVar(d_lab->d_uVelRhoHatLabel);
  tsk->computesVar(d_lab->d_vVelRhoHatLabel);
  tsk->computesVar(d_lab->d_wVelRhoHatLabel);
  tsk->computesVar(d_lab->d_densityCPLabel);
  tsk->computesVar(d_lab->d_viscosityCTSLabel);
  tsk->computesVar(d_lab->d_turbViscosLabel);
  tsk->computesVar(d_lab->d_conv_scheme_x_Label);
  tsk->computesVar(d_lab->d_conv_scheme_y_Label);
  tsk->computesVar(d_lab->d_conv_scheme_z_Label);

  //__________________________________
  tsk->computesVar(d_lab->d_densityTempLabel);
  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void
ExplicitSolver::sched_interpolateFromFCToCC(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls,
                                            const TimeIntegratorLabel* timelabels,
                                            const int curr_level)
{
  {
    string taskname =  "ExplicitSolver::interpFCToCC" +
                     timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
                         &ExplicitSolver::interpolateFromFCToCC, timelabels, curr_level);

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    tsk->requiresVar(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->requiresVar(Task::OldDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    // hat velocities are only interpolated for first substep, since they are
    // not really needed anyway
      tsk->requiresVar(Task::NewDW, d_lab->d_uVelRhoHatLabel,  gaf, 1);
      tsk->requiresVar(Task::NewDW, d_lab->d_vVelRhoHatLabel,  gaf, 1);
      tsk->requiresVar(Task::NewDW, d_lab->d_wVelRhoHatLabel,  gaf, 1);

    }

    tsk->requiresVar(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requiresVar(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requiresVar(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requiresVar(Task::NewDW, d_lab->d_filterdrhodtLabel,  gn,  0);
    tsk->requiresVar(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
    tsk->requiresVar(Task::OldDW, d_lab->d_delTLabel);

    if ( d_solvability ){
      std::stringstream strRKStage;
      strRKStage << curr_level;
      tsk->requiresVar(Task::NewDW, VarLabel::find("poisson_rhs_integral"+strRKStage.str()));
    }

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computesVar(d_lab->d_CCVelocityLabel);
      tsk->computesVar(d_lab->d_velocityDivergenceLabel);
      tsk->computesVar(d_lab->d_continuityResidualLabel);
      tsk->computesVar(d_lab->d_CCUVelocityLabel);
      tsk->computesVar(d_lab->d_CCVVelocityLabel);
      tsk->computesVar(d_lab->d_CCWVelocityLabel);
    }
    else {
      tsk->modifiesVar(d_lab->d_CCVelocityLabel);
      tsk->modifiesVar(d_lab->d_velocityDivergenceLabel);
      tsk->modifiesVar(d_lab->d_continuityResidualLabel);
      tsk->modifiesVar(d_lab->d_CCUVelocityLabel);
      tsk->modifiesVar(d_lab->d_CCVVelocityLabel);
      tsk->modifiesVar(d_lab->d_CCWVelocityLabel);
    }

    // add access to sources:
    for (vector<std::string>::iterator iter = d_mass_sources.begin();
        iter != d_mass_sources.end(); iter++){
      tsk->requiresVar( Task::NewDW, VarLabel::find( *iter ), gn, 0 );
    }

    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "ExplicitSolver::computeVorticity" +
                     timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
                         &ExplicitSolver::computeVorticity, timelabels);

    Ghost::GhostType  gac = Ghost::AroundCells;

    tsk->requiresVar(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);
    tsk->requiresVar(Task::NewDW, d_lab->d_CCVelocityLabel,  gac, 1);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computesVar(d_lab->d_vorticityXLabel);
      tsk->computesVar(d_lab->d_vorticityYLabel);
      tsk->computesVar(d_lab->d_vorticityZLabel);
      tsk->computesVar(d_lab->d_vorticityLabel);
    }
    else {
      tsk->modifiesVar(d_lab->d_vorticityXLabel);
      tsk->modifiesVar(d_lab->d_vorticityYLabel);
      tsk->modifiesVar(d_lab->d_vorticityZLabel);
      tsk->modifiesVar(d_lab->d_vorticityLabel);
    }

    sched->addTask(tsk, patches, matls);
  }
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void
ExplicitSolver::interpolateFromFCToCC(const ProcessorGroup* ,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      const TimeIntegratorLabel* timelabels,
                                      const int curr_level)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> divergence;
    CCVariable<double> residual;
    CCVariable<Vector> newCCVel;
    constCCVariable<double> density;
    constCCVariable<double> drhodt;
    constCCVariable<double> volFraction;

    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    CCVariable<Vector> CCVel;
    CCVariable<double> ccUVel;
    CCVariable<double> ccVVel;
    CCVariable<double> ccWVel;

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    delt_vartype delT;
    old_dw->get(delT, d_lab->d_delTLabel );
    double dt = delT;

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    double vol_integral=0.0;
    if ( d_solvability ){

      std::stringstream strRKStage;
      strRKStage << curr_level;
      sum_vartype the_sum;
      new_dw->get(the_sum, VarLabel::find("poisson_rhs_integral"+strRKStage.str()));
      vol_integral = the_sum;

    }

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    new_dw->get(newUVel        , d_lab->d_uVelocitySPBCLabel , indx , patch , gaf , 1);
    new_dw->get(newVVel        , d_lab->d_vVelocitySPBCLabel , indx , patch , gaf , 1);
    new_dw->get(newWVel        , d_lab->d_wVelocitySPBCLabel , indx , patch , gaf , 1);
    new_dw->get(drhodt         , d_lab->d_filterdrhodtLabel  , indx , patch , gn  , 0);
    new_dw->get(density        , d_lab->d_densityCPLabel     , indx , patch , gac , 1);
    new_dw->get(volFraction  ,   d_lab->d_volFractionLabel, indx,patch, gac, 1);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(newCCVel,      d_lab->d_CCVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(divergence,    d_lab->d_velocityDivergenceLabel,indx, patch);
      new_dw->allocateAndPut(residual,      d_lab->d_continuityResidualLabel,indx, patch);
      new_dw->allocateAndPut(ccUVel,        d_lab->d_CCUVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(ccVVel,        d_lab->d_CCVVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(ccWVel,        d_lab->d_CCWVelocityLabel,     indx, patch);
    }
    else {
      new_dw->getModifiable(newCCVel,       d_lab->d_CCVelocityLabel,      indx, patch);
      new_dw->getModifiable(divergence,     d_lab->d_velocityDivergenceLabel, indx, patch);
      new_dw->getModifiable(residual,       d_lab->d_continuityResidualLabel, indx, patch);
      new_dw->getModifiable(ccUVel,         d_lab->d_CCUVelocityLabel,     indx, patch);
      new_dw->getModifiable(ccVVel,         d_lab->d_CCVVelocityLabel,     indx, patch);
      new_dw->getModifiable(ccWVel,         d_lab->d_CCWVelocityLabel,     indx, patch);
    }
    newCCVel.initialize(Vector(0.0,0.0,0.0));
    divergence.initialize(0.0);
    residual.initialize(0.0);
    ccUVel.initialize(0.0);
    ccVVel.initialize(0.0);
    ccWVel.initialize(0.0);

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {

          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;

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

          double new_u = newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
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

          double new_u = newUVel[idx];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
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

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
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

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = newVVel[idx];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
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

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
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

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = newWVel[idx];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }

    std::vector<constCCVariable<double> > mass_srcs(d_mass_sources.size());
    // Add other source terms to the continuity
    int m=0;
    for (auto iter = d_mass_sources.begin(); iter != d_mass_sources.end(); iter++){

      new_dw->get( mass_srcs[m], VarLabel::find(*iter), indx, patch, gn, 0 );
      m++;

    }

    // For intrusion inlets:
    CCVariable<double> mass_src_intrusion;
    new_dw->allocateTemporary(mass_src_intrusion, patch);
    mass_src_intrusion.initialize(0.0);
    d_boundaryCondition->addIntrusionMassRHS(patch, mass_src_intrusion);

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {

          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          IntVector idxxminus(ii-1,jj,kk);
          IntVector idxyminus(ii,jj-1,kk);
          IntVector idxzminus(ii,jj,kk-1);
          double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];

          divergence[idx] = (newUVel[idxU]-newUVel[idx])/cellinfo->sew[ii]+
                            (newVVel[idxV]-newVVel[idx])/cellinfo->sns[jj]+
                            (newWVel[idxW]-newWVel[idx])/cellinfo->stb[kk];

          residual[idx] = ((0.5*(density[idxU]+density[idx])*newUVel[idxU]-
                           0.5*(density[idx]+density[idxxminus])*newUVel[idx])/cellinfo->sew[ii]+
                          (0.5*(density[idxV]+density[idx])*newVVel[idxV]-
                           0.5*(density[idx]+density[idxyminus])*newVVel[idx])/cellinfo->sns[jj]+
                          (0.5*(density[idxW]+density[idx])*newWVel[idxW]-
                           0.5*(density[idx]+density[idxzminus])*newWVel[idx])/cellinfo->stb[kk]+
                          drhodt[idx]/vol)*volFraction[idx];

          //adding the mass sources to the residual
          for (unsigned int iii=0; iii < d_mass_sources.size(); iii++){
            residual[idx] -= mass_srcs[iii][idx];
          }

          // Add intrusion mass sources to the residual
          if ( d_boundaryCondition->is_using_new_intrusion() ){
            residual[idx] -= (mass_src_intrusion[idx]/vol)*volFraction[idx];
          }

          if ( d_solvability ){
            residual[idx] += vol_integral*dt/vol;
          }

        }
      }
    }
  }
}

// ****************************************************************************
// Actual calculation of vorticity
// ****************************************************************************
void
ExplicitSolver::computeVorticity(const ProcessorGroup* ,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> vorticityX, vorticityY, vorticityZ, vorticity;

    constCCVariable<Vector> CCVel;

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(CCVel, d_lab->d_CCVelocityLabel, indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(vorticityX, d_lab->d_vorticityXLabel, indx, patch);
      new_dw->allocateAndPut(vorticityY, d_lab->d_vorticityYLabel, indx, patch);
      new_dw->allocateAndPut(vorticityZ, d_lab->d_vorticityZLabel, indx, patch);
      new_dw->allocateAndPut(vorticity,  d_lab->d_vorticityLabel,  indx, patch);
    }
    else {
      new_dw->getModifiable(vorticityX, d_lab->d_vorticityXLabel, indx, patch);
      new_dw->getModifiable(vorticityY, d_lab->d_vorticityYLabel, indx, patch);
      new_dw->getModifiable(vorticityZ, d_lab->d_vorticityZLabel, indx, patch);
      new_dw->getModifiable(vorticity,  d_lab->d_vorticityLabel,  indx, patch);
    }
    vorticityX.initialize(0.0);
    vorticityY.initialize(0.0);
    vorticityZ.initialize(0.0);
    vorticity.initialize(0.0);

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          IntVector idxxminus(ii-1,jj,kk);
          IntVector idxyminus(ii,jj-1,kk);
          IntVector idxzminus(ii,jj,kk-1);

          // ii,jj,kk velocity component cancels out when computing derivative,
          // so it has been ommited

          vorticityX[idx] = 0.5*(CCVel[idxV].z()-CCVel[idxyminus].z())/cellinfo->sns[jj]
                           -0.5*(CCVel[idxW].y()-CCVel[idxzminus].y())/cellinfo->stb[kk];
          vorticityY[idx] = 0.5*(CCVel[idxW].x()-CCVel[idxzminus].x())/cellinfo->stb[kk]
                           -0.5*(CCVel[idxU].z()-CCVel[idxxminus].z())/cellinfo->sew[ii];
          vorticityZ[idx] = 0.5*(CCVel[idxU].y()-CCVel[idxxminus].y())/cellinfo->sew[ii]
                           -0.5*(CCVel[idxV].x()-CCVel[idxyminus].x())/cellinfo->sns[jj];
          vorticity[idx] = sqrt(vorticityX[idx]*vorticityX[idx]+vorticityY[idx]*vorticityY[idx]
                          + vorticityZ[idx]*vorticityZ[idx]);
        }
      }
    }
  }
}

// ****************************************************************************
// Actual initialize
// ****************************************************************************
void
ExplicitSolver::setInitialGuess(const ProcessorGroup* ,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    constCCVariable<double> denMicro;
    CCVariable<double> denMicro_new;

    Ghost::GhostType  gn = Ghost::None;

    constCCVariable<int> cellType;
    old_dw->get(cellType, d_lab->d_cellTypeLabel,   indx, patch, gn, 0);
    constCCVariable<double> old_density_guess;
    old_dw->get( old_density_guess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);


    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> turb_viscosity;
    constCCVariable<Vector> ccVel;
    constCCVariable<double> old_volq;

    old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(density,   d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    old_dw->get(viscosity, d_lab->d_viscosityCTSLabel,  indx, patch, gn, 0);
    old_dw->get(turb_viscosity,    d_lab->d_turbViscosLabel,  indx, patch, gn, 0);
    old_dw->get(ccVel,     d_lab->d_CCVelocityLabel, indx, patch, gn, 0);

  // Create vars for new_dw ***warning changed new_dw to old_dw...check
    CCVariable<int> cellType_new;
    new_dw->allocateAndPut(cellType_new, d_lab->d_cellTypeLabel, indx, patch);
    cellType_new.copyData(cellType);

    PerPatch<CellInformationP> cellInfoP;
    cellInfoP.setData(scinew CellInformation(patch));
    new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);

    SFCXVariable<double> uVelocity_new;
    new_dw->allocateAndPut(uVelocity_new, d_lab->d_uVelocitySPBCLabel, indx, patch);
    uVelocity_new.copyData(uVelocity); // copy old into new
    SFCYVariable<double> vVelocity_new;
    new_dw->allocateAndPut(vVelocity_new, d_lab->d_vVelocitySPBCLabel, indx, patch);
    vVelocity_new.copyData(vVelocity); // copy old into new
    SFCZVariable<double> wVelocity_new;
    new_dw->allocateAndPut(wVelocity_new, d_lab->d_wVelocitySPBCLabel, indx, patch);
    wVelocity_new.copyData(wVelocity); // copy old into new

    SFCXVariable<double> uVelocity_kokkos;
    new_dw->allocateAndPut(uVelocity_kokkos, d_lab->d_uVelocityLabel, indx, patch);
    uVelocity_kokkos.copyData(uVelocity); // copy old into new
    SFCYVariable<double> vVelocity_kokkos;
    new_dw->allocateAndPut(vVelocity_kokkos, d_lab->d_vVelocityLabel, indx, patch);
    vVelocity_kokkos.copyData(vVelocity); // copy old into new
    SFCZVariable<double> wVelocity_kokkos;
    new_dw->allocateAndPut(wVelocity_kokkos, d_lab->d_wVelocityLabel, indx, patch);
    wVelocity_kokkos.copyData(wVelocity); // copy old into new

    SFCXVariable<double> uVelRhoHat_new;
    new_dw->allocateAndPut(uVelRhoHat_new, d_lab->d_uVelRhoHatLabel, indx, patch);
    uVelRhoHat_new.initialize(0.0);     // copy old into new
    SFCYVariable<double> vVelRhoHat_new;
    new_dw->allocateAndPut(vVelRhoHat_new, d_lab->d_vVelRhoHatLabel, indx, patch);
    vVelRhoHat_new.initialize(0.0); // copy old into new
    SFCZVariable<double> wVelRhoHat_new;
    new_dw->allocateAndPut(wVelRhoHat_new, d_lab->d_wVelRhoHatLabel, indx, patch);
    wVelRhoHat_new.initialize(0.0); // copy old into new

    CCVariable<double> density_new;
    new_dw->allocateAndPut(density_new, d_lab->d_densityCPLabel, indx, patch);
    density_new.copyData(density); // copy old into new

    CCVariable<double> density_temp;
    new_dw->allocateAndPut(density_temp, d_lab->d_densityTempLabel, indx, patch);
    density_temp.copyData(density); // copy old into new

    CCVariable<double> viscosity_new;
    new_dw->allocateAndPut(viscosity_new, d_lab->d_viscosityCTSLabel, indx, patch);
    viscosity_new.copyData(viscosity); // copy old into new

    CCVariable<double> turb_viscosity_new;
    new_dw->allocateAndPut(turb_viscosity_new, d_lab->d_turbViscosLabel, indx, patch);
    turb_viscosity_new.copyData(turb_viscosity); // copy old into new

    SFCXVariable<double> conv_scheme_x;
    new_dw->allocateAndPut(conv_scheme_x, d_lab->d_conv_scheme_x_Label, indx, patch);
    conv_scheme_x.initialize(0.0); // copy old into new
    SFCYVariable<double> conv_scheme_y;
    new_dw->allocateAndPut(conv_scheme_y, d_lab->d_conv_scheme_y_Label, indx, patch);
    conv_scheme_y.initialize(0.0); // copy old into new
    SFCZVariable<double> conv_scheme_z;
    new_dw->allocateAndPut(conv_scheme_z, d_lab->d_conv_scheme_z_Label, indx, patch);
    conv_scheme_z.initialize(0.0); // copy old into new

  }
}


//______________________________________________________________________
//
void
ExplicitSolver::sched_printTotalKE( SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls )

{
  string taskname =  "ExplicitSolver::printTotalKE";
  Task* tsk = scinew Task( taskname,
                           this, &ExplicitSolver::printTotalKE );

  tsk->requiresVar(Task::NewDW, d_lab->d_totalKineticEnergyLabel);
  sched->addTask(tsk, patches, matls);
}
//______________________________________________________________________
void
ExplicitSolver::printTotalKE( const ProcessorGroup* ,
                              const PatchSubset* ,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse* new_dw )
{

  sum_vartype tke;
  new_dw->get( tke, d_lab->d_totalKineticEnergyLabel );
  double total_kin_energy = tke;

  proc0cout << "Total kinetic energy: " << total_kin_energy << "\n";

}

//****************************************************************************
// Schedule saving of temp copies of variables
//****************************************************************************
void
ExplicitSolver::sched_saveTempCopies(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::saveTempCopies" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::saveTempCopies,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requiresVar(Task::NewDW, d_lab->d_densityCPLabel, gn, 0);
  tsk->modifiesVar(d_lab->d_densityTempLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp copies here
//****************************************************************************
void
ExplicitSolver::saveTempCopies(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse*,
                               DataWarehouse* new_dw,
                               const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->
                     getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> temp_density;
    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,indx, patch);
    new_dw->copyOut(temp_density,       d_lab->d_densityCPLabel,  indx, patch);

  }
}
//****************************************************************************
// Schedule computation of density guess from the continuity equation
//****************************************************************************
void
ExplicitSolver::sched_getDensityGuess(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::getDensityGuess" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::getDensityGuess,
                          timelabels);

  tsk->requiresVar( Task::OldDW, d_lab->d_timeStepLabel );

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  tsk->requiresVar(parent_old_dw, d_lab->d_delTLabel);

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
    tsk->requiresVar(old_values_dw, d_lab->d_densityCPLabel,gn, 0);
  }


  tsk->requiresVar(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_cellTypeLabel,      gn, 0);
  tsk->requiresVar(Task::NewDW, d_lab->d_volFractionLabel,   gac, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_cellInfoLabel, gn);


  //__________________________________
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
    tsk->computesVar(d_lab->d_densityGuessLabel);
  }else{
    tsk->modifiesVar(d_lab->d_densityGuessLabel);
  }
  tsk->computesVar(timelabels->negativeDensityGuess);

  // extra mass source terms
  std::vector<std::string> extra_sources;
  extra_sources = d_pressSolver->get_pressure_source_ref();

  for ( auto iter = extra_sources.begin(); iter != extra_sources.end(); iter++){

    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
      tsk->requiresVar( Task::OldDW, VarLabel::find( *iter ), gn, 0 );
    } else {
      tsk->requiresVar( Task::NewDW, VarLabel::find( *iter ), gn, 0 );
    }
  }

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute density guess from the continuity equation
//****************************************************************************
void
ExplicitSolver::getDensityGuess(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels)
{
  // int timeStep = d_lab->d_materialManager->getCurrentTopLevelTimeStep();
  timeStep_vartype timeStep;
  old_dw->get( timeStep, d_lab->d_timeStepLabel );

  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_delTLabel );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  double negativeDensityGuess = 0.0;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    Vector Dx = patch->dCell();
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->
                     getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> densityGuess;
    constCCVariable<double> density;
    constCCVariable<double> volFraction;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<int> cellType;

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{
      old_values_dw = new_dw;
    }

    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
      new_dw->allocateAndPut(densityGuess, d_lab->d_densityGuessLabel, indx, patch);
    }else{
      new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel,  indx, patch);
    }
    old_values_dw->copyOut(densityGuess, d_lab->d_densityCPLabel,indx, patch);

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    new_dw->get(density,   d_lab->d_densityCPLabel,     indx,patch, gac, 1);
    new_dw->get(volFraction  ,   d_lab->d_volFractionLabel, indx,patch, gac, 1);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(cellType,  d_lab->d_cellTypeLabel,      indx,patch, gn, 0);

    // For adding other source terms as specified in the pressure solver section
    std::vector<std::string> extra_sources;
    extra_sources = d_pressSolver->get_pressure_source_ref();
    std::vector<constCCVariable<double> > src_values;
    bool have_extra_srcs = false;

    for ( auto iter = extra_sources.begin(); iter != extra_sources.end(); iter++){

      constCCVariable<double> src_value;

      if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
       old_dw->get( src_value, VarLabel::find( *iter ), indx, patch, gn, 0 );
      } else {
       new_dw->get( src_value, VarLabel::find( *iter ), indx, patch, gn, 0 );
      }

      src_values.push_back( src_value );

      have_extra_srcs = true;

    }

// Need to skip first timestep since we start with unprojected velocities
//    if (timeStep > 1) {
//
      if ( have_extra_srcs ){

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

          IntVector c   = *iter;
          IntVector cxp  = *iter + IntVector(1,0,0);
          IntVector cxm = *iter + IntVector(-1,0,0);
          IntVector cyp  = *iter + IntVector(0,1,0);
          IntVector cym = *iter + IntVector(0,-1,0);
          IntVector czp  = *iter + IntVector(0,0,1);
          IntVector czm = *iter + IntVector(0,0,-1);

          double epsE = volFraction[c]*volFraction[cxp] < 1. ? 0 : 1;
          double epsW = volFraction[c]*volFraction[cxm] < 1. ? 0 : 1;
          double epsN = volFraction[c]*volFraction[cyp] < 1. ? 0 : 1;
          double epsS = volFraction[c]*volFraction[cym] < 1. ? 0 : 1;
          double epsT = volFraction[c]*volFraction[czp] < 1. ? 0 : 1;
          double epsB = volFraction[c]*volFraction[czm] < 1. ? 0 : 1;

          densityGuess[c] -= delta_t * 0.5* (
          ((density[c]+density[cxp])*uVelocity[cxp]*epsE -
           (density[c]+density[cxm])*uVelocity[c]*epsW) / cellinfo->sew[c.x()] +
          ((density[c]+density[cyp])*vVelocity[cyp]*epsN -
           (density[c]+density[cym])*vVelocity[c]*epsS) / cellinfo->sns[c.y()] +
          ((density[c]+density[czp])*wVelocity[czp]*epsT -
           (density[c]+density[czm])*wVelocity[c]*epsB) / cellinfo->stb[c.z()]);

          for ( std::vector<constCCVariable<double> >::iterator viter = src_values.begin();
            viter != src_values.end(); viter++ ){

            densityGuess[c] += delta_t*((*viter)[c]);

          }

          if (densityGuess[c] < 0.0 && d_noisyDensityGuess) {
            proc0cout << "Negative density guess occured at " << c
              << " with a value of " << densityGuess[c] << endl;
            negativeDensityGuess = 1.0;
          }
          else if (densityGuess[c] < 0.0 && !(d_noisyDensityGuess)) {
            negativeDensityGuess = 1.0;
          }
        }

      } else {

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

          IntVector c   = *iter;
          IntVector cxp  = *iter + IntVector(1,0,0);
          IntVector cxm = *iter + IntVector(-1,0,0);
          IntVector cyp  = *iter + IntVector(0,1,0);
          IntVector cym = *iter + IntVector(0,-1,0);
          IntVector czp  = *iter + IntVector(0,0,1);
          IntVector czm = *iter + IntVector(0,0,-1);

          double epsE = volFraction[c]*volFraction[cxp] < 1. ? 0 : 1;
          double epsW = volFraction[c]*volFraction[cxm] < 1. ? 0 : 1;
          double epsN = volFraction[c]*volFraction[cyp] < 1. ? 0 : 1;
          double epsS = volFraction[c]*volFraction[cym] < 1. ? 0 : 1;
          double epsT = volFraction[c]*volFraction[czp] < 1. ? 0 : 1;
          double epsB = volFraction[c]*volFraction[czm] < 1. ? 0 : 1;

          densityGuess[c] -= delta_t * 0.5* (
          ((density[c]+density[cxp])*uVelocity[cxp]*epsE -
           (density[c]+density[cxm])*uVelocity[c]*epsW) / cellinfo->sew[c.x()] +
          ((density[c]+density[cyp])*vVelocity[cyp]*epsN -
           (density[c]+density[cym])*vVelocity[c]*epsS) / cellinfo->sns[c.y()] +
          ((density[c]+density[czp])*wVelocity[czp]*epsT -
           (density[c]+density[czm])*wVelocity[c]*epsB) / cellinfo->stb[c.z()]);

          if (densityGuess[c] < 0.0 && d_noisyDensityGuess) {
            cout << "Negative density guess occured at " << c
              << " with a value of " << densityGuess[c] << endl;
            negativeDensityGuess = 1.0;
          } else if (densityGuess[c] < 0.0 && !(d_noisyDensityGuess) && cellType[c] == -1 ) {
            negativeDensityGuess = 1.0;
          }

        }
      }

      // For intrusion inlets:
      if ( d_boundaryCondition->is_using_new_intrusion() ){

        CCVariable<double> mass_src;
        new_dw->allocateTemporary(mass_src, patch);
        mass_src.initialize(0.0);
        d_boundaryCondition->addIntrusionMassRHS(patch, mass_src);
        double vol = Dx.x() * Dx.y() * Dx.z();

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

          IntVector c = *iter;
          densityGuess[c] += delta_t / vol * mass_src[c] * volFraction[c];

          if (densityGuess[c] < 0.0 && d_noisyDensityGuess) {
            cout << "Negative density guess occured at " << c
              << " with a value of " << densityGuess[c] << endl;
            negativeDensityGuess = 1.0;
          } else if (densityGuess[c] < 0.0 && !(d_noisyDensityGuess) && cellType[c] == -1 ) {
            negativeDensityGuess = 1.0;
          }

        }
      }
      //-------------------

      std::vector<BoundaryCondition::BC_TYPE> bc_types;
      bc_types.push_back( BoundaryCondition::OUTLET );
      bc_types.push_back( BoundaryCondition::PRESSURE );

      d_boundaryCondition->zeroGradientBC( patch, indx, densityGuess, bc_types );

      new_dw->put(sum_vartype(negativeDensityGuess),
                  timelabels->negativeDensityGuess);
  }
}
//****************************************************************************
// Schedule check for negative density guess
//****************************************************************************
void
ExplicitSolver::sched_checkDensityGuess(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::checkDensityGuess" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::checkDensityGuess,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else {
    old_values_dw = Task::NewDW;
  }

  tsk->requiresVar(old_values_dw, d_lab->d_densityCPLabel,Ghost::None, 0);

  tsk->requiresVar(Task::NewDW, timelabels->negativeDensityGuess);

  tsk->modifiesVar(d_lab->d_densityGuessLabel);

  tsk->computesVar( VarLabel::find(abortTimeStep_name) );
  tsk->computesVar( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually check for negative density guess
//****************************************************************************
void
ExplicitSolver::checkDensityGuess(const ProcessorGroup* pc,
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

  double negativeDensityGuess = 0.0;
  sum_vartype nDG;
  new_dw->get(nDG, timelabels->negativeDensityGuess);

  negativeDensityGuess = nDG;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->
                     getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> densityGuess;
    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values)
      old_values_dw = parent_old_dw;
    else
      old_values_dw = new_dw;

    new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel, indx, patch);
    if (negativeDensityGuess > 0.0) {
      if (d_recompute_on_negative_density_guess) {
        proc0cout << "NOTICE: Negative density guess(es) occurred. Timestep restart has been requested under this condition by the user. Recomputing timestep." << endl;
        new_dw->put( bool_or_vartype(true), VarLabel::find(abortTimeStep_name));
        new_dw->put( bool_or_vartype(true), VarLabel::find(recomputeTimeStep_name) );
      }
      else {
        proc0cout << __FUNCTION__ << " NOTICE: Negative density guess(es) occurred. Reverting to old density." << endl;
        old_values_dw->copyOut(densityGuess, d_lab->d_densityCPLabel, indx, patch);
      }
    }
  }
}
//****************************************************************************
// Schedule update of density guess
//****************************************************************************
void
ExplicitSolver::sched_updateDensityGuess(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::updateDensityGuess" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::updateDensityGuess,
                          timelabels);

  tsk->requiresVar(Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 1);
  tsk->modifiesVar(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute density guess from the continuity equation
//****************************************************************************
void
ExplicitSolver::updateDensityGuess(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw,
                                   const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->
                     getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> densityGuess;
    constCCVariable<double> density;

    new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel,indx, patch);
    new_dw->copyOut(densityGuess,       d_lab->d_densityCPLabel,   indx, patch);
  }
}
//****************************************************************************
// Schedule computing density lag
//****************************************************************************
void
ExplicitSolver::sched_computeDensityLag(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        bool after_average)
{
  string taskname =  "ExplicitSolver::computeDensityLag" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::computeDensityLag,
                          timelabels, after_average);
  Ghost::GhostType  gn = Ghost::None;
  tsk->requiresVar(Task::NewDW, d_lab->d_densityGuessLabel,gn, 0);
  tsk->requiresVar(Task::NewDW, d_lab->d_densityCPLabel,   gn, 0);

  if (after_average){
    if ((timelabels->integrator_step_name == "Corrector")||
        (timelabels->integrator_step_name == "CorrectorRK3")){
      tsk->computesVar(d_lab->d_densityLagAfterAverage_label);
    }else{
      tsk->computesVar(d_lab->d_densityLagAfterIntermAverage_label);
    }
  }else{
    tsk->computesVar(timelabels->densityLag);
  }
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute deensity lag
//****************************************************************************
void
ExplicitSolver::computeDensityLag(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse*,
                                  DataWarehouse* new_dw,
                                  const TimeIntegratorLabel* timelabels,
                                  bool after_average)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->
                     getMaterial( "Arches", archIndex)->getDWIndex();

    constCCVariable<double> densityGuess;
    constCCVariable<double> density;

    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(densityGuess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);
    new_dw->get(density, d_lab->d_densityCPLabel,         indx, patch, gn, 0);

    double densityLag = 0.0;
    IntVector idxLo = patch->getExtraCellLowIndex();
    IntVector idxHi = patch->getExtraCellHighIndex();
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          densityLag += Abs(density[currCell] - densityGuess[currCell]);
        }
      }
    }
    if (after_average){
      if ((timelabels->integrator_step_name == "Corrector")||
          (timelabels->integrator_step_name == "CorrectorRK3")){
        new_dw->put(sum_vartype(densityLag), d_lab->d_densityLagAfterAverage_label);
      }else{
        new_dw->put(sum_vartype(densityLag), d_lab->d_densityLagAfterIntermAverage_label);
      }
    }else{
      new_dw->put(sum_vartype(densityLag), timelabels->densityLag);
    }
  }
}
//****************************************************************************
// Schedule check for density lag
//****************************************************************************
void
ExplicitSolver::sched_checkDensityLag(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels,
                                      bool after_average)
{
  string taskname =  "ExplicitSolver::checkDensityLag" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::checkDensityLag,
                          timelabels, after_average);

  if (after_average){
    if ((timelabels->integrator_step_name == "Corrector")||
        (timelabels->integrator_step_name == "CorrectorRK3")){
      tsk->requiresVar(Task::NewDW, d_lab->d_densityLagAfterAverage_label);
    }else{
      tsk->requiresVar(Task::NewDW, d_lab->d_densityLagAfterIntermAverage_label);
    }
  }else{
    tsk->requiresVar(Task::NewDW, timelabels->densityLag);
  }

  tsk->computesVar( VarLabel::find(abortTimeStep_name) );
  tsk->computesVar( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually check for density lag
//****************************************************************************
void
ExplicitSolver::checkDensityLag(const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels,
                                bool after_average)
{
  double densityLag = 0.0;
  sum_vartype denLag;
  if (after_average){
    if ((timelabels->integrator_step_name == "Corrector")||
        (timelabels->integrator_step_name == "CorrectorRK3")){
      new_dw->get(denLag, d_lab->d_densityLagAfterAverage_label);
    }else{
      new_dw->get(denLag, d_lab->d_densityLagAfterIntermAverage_label);
    }
  }else{
    new_dw->get(denLag, timelabels->densityLag);
  }
  densityLag = denLag;

  for (int p = 0; p < patches->size(); p++) {

    if (densityLag > d_maxDensityLag) {
        if (pc->myRank() == 0)
          proc0cout << "WARNING: density lag " << densityLag
               << " exceeding maximium "<< d_maxDensityLag
               << " specified. Recomputing timestep." << endl;
        new_dw->put( bool_or_vartype(true), VarLabel::find(abortTimeStep_name));
        new_dw->put( bool_or_vartype(true), VarLabel::find(recomputeTimeStep_name) );
    }
  }
}

void
ExplicitSolver::sched_setInitVelCond( const LevelP& level,
                                      SchedulerP& sched,
                                      const MaterialSet* matls ){

  string taskname = "ExplicitSolver::setInitVelCond";
  Task* tsk = scinew Task( taskname, this, &ExplicitSolver::setInitVelCond );

  tsk->requiresVar( Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 1 );
  tsk->modifiesVar( d_lab->d_uVelocitySPBCLabel );
  tsk->modifiesVar( d_lab->d_vVelocitySPBCLabel );
  tsk->modifiesVar( d_lab->d_wVelocitySPBCLabel );

  sched->addTask( tsk, level->eachPatch(), matls );

}

void ExplicitSolver::setInitVelCond( const ProcessorGroup* pc,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw )
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    constCCVariable<double> rho;
    SFCXVariable<double> u;
    SFCYVariable<double> v;
    SFCZVariable<double> w;

    new_dw->get( rho, d_lab->d_densityCPLabel, indx, patch, Ghost::AroundCells, 1 );
    new_dw->getModifiable( u, d_lab->d_uVelocitySPBCLabel, indx, patch );
    new_dw->getModifiable( v, d_lab->d_vVelocitySPBCLabel, indx, patch );
    new_dw->getModifiable( w, d_lab->d_wVelocitySPBCLabel, indx, patch );

    d_momSolver->setInitVelCondition( patch, u, v, w, rho );

  }
}


void ExplicitSolver::sched_computeKE( SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls )
{
  string taskname = "ExplicitSolver::computeKE";
  Task* tsk = scinew Task( taskname, this, &ExplicitSolver::computeKE );

  tsk->computesVar(d_lab->d_totalKineticEnergyLabel);
  tsk->computesVar(d_lab->d_kineticEnergyLabel);

  tsk->requiresVar( Task::NewDW, d_lab->d_uVelocitySPBCLabel, Ghost::None, 0 );
  tsk->requiresVar( Task::NewDW, d_lab->d_vVelocitySPBCLabel, Ghost::None, 0 );
  tsk->requiresVar( Task::NewDW, d_lab->d_wVelocitySPBCLabel, Ghost::None, 0 );

  sched->addTask( tsk, patches, matls );

}

void ExplicitSolver::computeKE( const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    constSFCXVariable<double> u;
    constSFCYVariable<double> v;
    constSFCZVariable<double> w;
    CCVariable<double> ke;

    new_dw->allocateAndPut( ke, d_lab->d_kineticEnergyLabel, indx, patch );
    ke.initialize(0.0);
    new_dw->get( u, d_lab->d_uVelocitySPBCLabel, indx, patch, Ghost::None, 0 );
    new_dw->get( v, d_lab->d_vVelocitySPBCLabel, indx, patch, Ghost::None, 0 );
    new_dw->get( w, d_lab->d_wVelocitySPBCLabel, indx, patch, Ghost::None, 0 );

    double max_ke = 0.0;
    double sum_ke = 0.0;

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

      IntVector c = *iter;

      ke[c] = 0.5 * ( u[c]*u[c] + v[c]*v[c] + w[c]*w[c] );

      if ( ke[c] > max_ke ) {
        max_ke = ke[c];
      }

      sum_ke += ke[c];
    }

    new_dw->put(sum_vartype(sum_ke), d_lab->d_totalKineticEnergyLabel);

    if ( sum_ke != sum_ke ) {
      std::stringstream msg;
      msg << "Error: Your KE, " << sum_ke << " has diverged." << endl;
      throw InvalidValue(msg.str(), __FILE__, __LINE__);
    }

    if ( sum_ke > d_ke_limit ) {
      std::stringstream msg;
      msg << "Error: Your KE, " << sum_ke << " has exceeded the max threshold allowed of: " << d_ke_limit << endl;
      throw InvalidValue(msg.str(), __FILE__, __LINE__);
    }
  }
}

//___________________________________________________________________________
//
void
ExplicitSolver::sched_weightInit( const LevelP& level,
                          SchedulerP& sched )
{
  Task* tsk = scinew Task( "ExplicitSolver::weightInit",
                           this, &ExplicitSolver::weightInit);

  // DQMOM weight transport vars
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
  for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++) {
    EqnBase* temp_eqn = ieqn->second;
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(temp_eqn);

    if (eqn->weight()) {
      const VarLabel* tempVar = eqn->getTransportEqnLabel();
      const VarLabel* tempVar_icv = eqn->getUnscaledLabel();

      tsk->computesVar( tempVar );
      tsk->computesVar( tempVar_icv );
    }
  }

  tsk->requiresVar( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_lab->d_materialManager->allMaterials( "Arches" ));
}
//______________________________________________________________________
//
void
ExplicitSolver::weightInit( const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw )
{

  proc0cout << "Initializing all DQMOM weight equations..." << endl;
  for (int p = 0; p < patches->size(); p++) {
    //assume only one material for now.
    int archIndex = 0;
    int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    CCVariable<Vector> partVel;
    constCCVariable<double> eps_v;

    new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, Ghost::None, 0 );

    DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();

    // --- DQMOM EQNS
    // do only weights
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin();
         ieqn != dqmom_eqns.end(); ieqn++) {

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(ieqn->second);
      string eqn_name = ieqn->first;

      if (eqn->weight()) {
        // This is a weight equation
        const VarLabel* phiLabel     = eqn->getTransportEqnLabel();
        const VarLabel* phiLabel_icv = eqn->getUnscaledLabel();

        //CCVariable<double> source;
        CCVariable<double> phi;
        CCVariable<double> phi_icv;

        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch );
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch );

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
ExplicitSolver::sched_weightedAbsInit( const LevelP& level,
                               SchedulerP& sched )
{
  Task* tsk = scinew Task( "ExplicitSolver::weightedAbsInit",
                           this, &ExplicitSolver::weightedAbsInit);
  // DQMOM transport vars
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
  for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++) {

    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(ieqn->second);

    if (!eqn->weight()) {
      const VarLabel* tempVar = eqn->getTransportEqnLabel();
      const VarLabel* tempVar_icv = eqn->getUnscaledLabel();
      tsk->computesVar( tempVar );
      tsk->computesVar( tempVar_icv );
    } else {
      const VarLabel* tempVar = eqn->getTransportEqnLabel();
      tsk->requiresVar( Task::NewDW, tempVar, Ghost::None, 0 );
    }
  }

  // Particle Velocities

  // Models
  // initialize all of the computed variables for the coal models
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  modelFactory.sched_init_all_models( level, sched );

  tsk->requiresVar( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_lab->d_materialManager->allMaterials( "Arches" ));
}
//______________________________________________________________________
//
void
ExplicitSolver::weightedAbsInit( const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset*,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw )
{

  string msg = "Initializing all DQMOM weighted abscissa equations...";
  proc0cout << msg << std::endl;

  for (int p = 0; p < patches->size(); p++) {
    //assume only one material for now.
    int archIndex = 0;
    int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    CCVariable<Vector> partVel;
    constCCVariable<double> eps_v;

    Ghost::GhostType gn = Ghost::None;

    new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, gn, 0 );

    DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();

    // --- DQMOM EQNS
    // do weights first because we need them later for the weighted abscissas
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin();
         ieqn != dqmom_eqns.end(); ieqn++) {

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(ieqn->second);
      string eqn_name = ieqn->first;
      int qn = eqn->getQuadNode();

      if (!eqn->weight()) {
        // This is a weighted abscissa
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

        CCVariable<double> phi;
        CCVariable<double> phi_icv;
        constCCVariable<double> weight;

        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch );
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch );
        new_dw->get( weight, weightLabel, matlIndex, patch, gn, 0 );

        phi.initialize(0.0);
        phi_icv.initialize(0.0);

        // initialize phi
        if( d_which_dqmom == "unweightedAbs" ) {
          eqn->initializationFunction( patch, phi, eps_v);
        } else {
          eqn->initializationFunction( patch, phi, weight, eps_v );
        }

        // do boundary conditions
        eqn->computeBCs( patch, eqn_name, phi );

      }
    }
  }
  proc0cout << endl;
}
//___________________________________________________________________________
//
void
ExplicitSolver::sched_momentInit( const LevelP& level,
                          SchedulerP& sched )
{
  Task* tsk = scinew Task( "ExplicitSolver::momentInit",
                           this, &ExplicitSolver::momentInit);

  // CQMOM moment transport vars
  CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
  CQMOMEqnFactory::EqnMap& cqmom_eqns = cqmomFactory.retrieve_all_eqns();
  for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin(); ieqn != cqmom_eqns.end(); ieqn++) {
    EqnBase* temp_eqn = ieqn->second;
    CQMOMEqn* eqn = dynamic_cast<CQMOMEqn*>(temp_eqn);

    const VarLabel* tempVar = eqn->getTransportEqnLabel();
    const VarLabel* tempSource = eqn->getSourceLabel();

    tsk->computesVar( tempVar );
    tsk->computesVar( tempSource );
  }

  tsk->requiresVar( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_lab->d_materialManager->allMaterials( "Arches" ));
}
//______________________________________________________________________
//
void
ExplicitSolver::momentInit( const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw )
{

  proc0cout << "Initializing all CQMOM moment equations..." << endl;
  for (int p = 0; p < patches->size(); p++) {
    //assume only one material for now.
    int archIndex = 0;
    int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    constCCVariable<double> eps_v;

    new_dw->get( eps_v, d_lab->d_volFractionLabel, matlIndex, patch, Ghost::None, 0 );

    CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
    CQMOMEqnFactory::EqnMap& cqmom_eqns = cqmomFactory.retrieve_all_eqns();

    // --- CQMOM EQNS
    for (CQMOMEqnFactory::EqnMap::iterator ieqn=cqmom_eqns.begin();
         ieqn != cqmom_eqns.end(); ieqn++) {

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

//___________________________________________________________________________
//
void
ExplicitSolver::sched_scalarInit( const LevelP     & level,
                                        SchedulerP & sched )
{
  Task* tsk = scinew Task( "ExplicitSolver::scalarInit", this, &ExplicitSolver::scalarInit );

  EqnFactory& eqnFactory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();

  for( EqnFactory::EqnMap::iterator ieqn = scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++ ) {

    EqnBase* eqn = ieqn->second;

    eqn->sched_checkBCs( level, sched, false );

    const VarLabel* tempVar = eqn->getTransportEqnLabel();
    tsk->computesVar( tempVar );
  }

  tsk->requiresVar( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None );

  sched->addTask(tsk, level->eachPatch(), d_lab->d_materialManager->allMaterials( "Arches" ));

  //__________________________________
  //  initialize src terms
  SourceTermFactory& srcFactory = SourceTermFactory::self();
  SourceTermFactory::SourceMap& sources = srcFactory.retrieve_all_sources();
  for( SourceTermFactory::SourceMap::iterator isrc=sources.begin(); isrc !=sources.end(); isrc++ ) {
    SourceTermBase* src = isrc->second;
    src->sched_initialize( level, sched );
  }
}

//______________________________________________________________________
//
void
ExplicitSolver::scalarInit( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{

  proc0cout << "Initializing all scalar equations and sources...\n";
  for (int p = 0; p < patches->size(); p++) {
    //assume only one material for now
    int archIndex = 0;
    int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    const Patch* patch=patches->get(p);

    EqnFactory& eqnFactory = EqnFactory::self();
    EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns();
    for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++) {

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
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC
// ****************************************************************************
void
ExplicitSolver::sched_getCCVelocities(const LevelP& level, SchedulerP& sched)
{
  Task* tsk = scinew Task("ExplicitSolver::getCCVelocities", this,
                          &ExplicitSolver::getCCVelocities);

  Ghost::GhostType gaf = Ghost::AroundFaces;
  tsk->requiresVar(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requiresVar(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  tsk->modifiesVar(d_lab->d_CCVelocityLabel);
  tsk->modifiesVar(d_lab->d_CCUVelocityLabel);
  tsk->modifiesVar(d_lab->d_CCVVelocityLabel);
  tsk->modifiesVar(d_lab->d_CCWVelocityLabel);

  sched->addTask(tsk, level->eachPatch(), d_lab->d_materialManager->allMaterials( "Arches" ));
}

// ****************************************************************************
// interpolation from FC to CC Variable
// ****************************************************************************
void
ExplicitSolver::getCCVelocities(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

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

    Ghost::GhostType gaf = Ghost::AroundFaces;
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
    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++) {
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
    for( itr = b_face.begin(); itr != b_face.end(); ++itr ) {
      Patch::FaceType face = *itr;

      IntVector f_dir = patch->getFaceDirection(face);

      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
      CellIterator iter=patch->getFaceIterator(face, MEC);

      Vector one_or_zero = Vector(1,1,1) - Abs(f_dir.asVector());
      // one_or_zero: faces x-+   (0,1,1)
      //                    y-+   (1,0,1)
      //                    z-+   (1,1,0)

      for(; !iter.done(); iter++) {
        IntVector c = *iter;

        int i = c.x();
        int j = c.y();
        int k = c.z();

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

//---------------------------------------------------------------------------
// Method: Register Models
//---------------------------------------------------------------------------
void ExplicitSolver::registerModels(ProblemSpecP& db)
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
    for (ProblemSpecP model_db = models_db->findBlock("model"); model_db != nullptr; model_db = model_db->findNextBlock("model")) {
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
        for( ProblemSpecP var = icvar_db->findBlock("variable"); var != nullptr; var = var->findNextBlock("variable") ) {

          std::string label_name;
          var->getAttribute("label", label_name);

          proc0cout << "label = " << label_name << endl;
          // This map hold the labels that are required to compute this model term.
          requiredICVarLabels.push_back(label_name);
        }
      }
      else {
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
        proc0cout << "Requires the following scalar variables:\n";
        for (ProblemSpecP var = scalarvar_db->findBlock("variable"); var != nullptr; var = var->findNextBlock("variable") ) {
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
      for (int iqn = 0; iqn < numQuadNodes; iqn++) {
        std::string temp_model_name = model_name;
        std::string node;
        std::stringstream out;
        out << iqn;
        node = out.str();
        temp_model_name += "_qn";
        temp_model_name += node;

        if ( model_type == "ConstantModel" ) {
          // Model term G = constant (G = 1)
          ModelBuilder* modelBuilder = scinew ConstantModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "KobayashiSarofimDevol" ) {
          // Kobayashi Sarofim devolatilization model
          ModelBuilder* modelBuilder = scinew KobayashiSarofimDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "RichardsFletcherDevol" ) {
          // Richards Fletcher devolatilization model
          ModelBuilder* modelBuilder = scinew RichardsFletcherDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "FOWYDevol" ) {
          // Biagini Tognotti devolatilization model
          ModelBuilder* modelBuilder = scinew FOWYDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "YamamotoDevol" ) {
          //ModelBuilder* modelBuilder = scinew YamamotoDevolBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          //model_factory.register_model( temp_model_name, modelBuilder );
          throw InvalidValue("Error: YamamotoDevol is not currently supported. The model needs to be updated/fixed. See FOWYDevol as an example.", __FILE__,__LINE__);
        } else if ( model_type == "CharOxidationShaddix" ) {
          ModelBuilder* modelBuilder = scinew CharOxidationShaddixBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "CharOxidationSmithConstLv0" ) {
          ModelBuilder* modelBuilder = scinew CharOxidationSmithConstLv0Builder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "CharOxidationSmith" ) {
          ModelBuilder* modelBuilder = scinew CharOxidationSmithBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "EnthalpyShaddix" ) {
          ModelBuilder* modelBuilder = scinew EnthalpyShaddixBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, d_props, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "MaximumTemperature" ) {
          ModelBuilder* modelBuilder = scinew MaximumTemperatureBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "Thermophoresis" ) {
          ModelBuilder* modelBuilder = scinew ThermophoresisBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "LinearSwelling" ) {
          ModelBuilder* modelBuilder = scinew LinearSwellingBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "ShrinkageRate" ) {
          ModelBuilder* modelBuilder = scinew ShrinkageRateBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "Drag" ) {
          ModelBuilder* modelBuilder = scinew DragModelBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "BirthDeath" ) {
          ModelBuilder* modelBuilder = scinew BirthDeathBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "Deposition" ) {
          ModelBuilder* modelBuilder = scinew DepositionBuilder(temp_model_name, requiredICVarLabels, requiredScalarVarLabels, d_lab, d_lab->d_materialManager, iqn);
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
void ExplicitSolver::registerTransportEqns(ProblemSpecP& db)
{
  ProblemSpecP eqns_db = db;

  // Get reference to the source factory
  EqnFactory& eqnFactory = EqnFactory::self();

  if (eqns_db) {

    proc0cout << "\n";
    proc0cout << "******* Equation Registration ********\n";

    for (ProblemSpecP eqn_db = eqns_db->findBlock("Eqn"); eqn_db != nullptr; eqn_db = eqn_db->findNextBlock("Eqn")) {
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
void ExplicitSolver::registerPropertyModels(ProblemSpecP& db)
{
  ProblemSpecP propmodels_db = db;
  PropertyModelFactory& prop_factory = PropertyModelFactory::self();

  if ( propmodels_db ) {

    proc0cout << "\n";
    proc0cout << "******* Property Model Registration *******\n";

    for ( ProblemSpecP prop_db = propmodels_db->findBlock("model"); prop_db != nullptr; prop_db = prop_db->findNextBlock("model") ) {

      std::string prop_name;
      prop_db->getAttribute("label", prop_name);
      std::string prop_type;
      prop_db->getAttribute("type", prop_type);

      proc0cout << "Found a property model: " << prop_name << endl;

      if ( prop_type == "cc_constant" ) {

        // An example of a constant CC variable property
        PropertyModelBase::Builder* the_builder = new ConstProperty<CCVariable<double>, constCCVariable<double> >::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      }  else if ( prop_type == "extent_rxn" ) {

        // Scalar dissipation rate calculation
        PropertyModelBase::Builder* the_builder = new ExtentRxn::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "tab_strip_factor" ) {

        // Scalar dissipation rate calculation
        PropertyModelBase::Builder* the_builder = new TabStripFactor::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "fx_constant" ) {

        // An example of a constant FCX variable property
        PropertyModelBase::Builder* the_builder = new ConstProperty<SFCXVariable<double>, constCCVariable<double> >::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "empirical_soot" ) {

        // emperical soot model (computes soot volume fraction and abskp)
        PropertyModelBase::Builder* the_builder = new EmpSoot::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "algebraic_scalar_diss" ) {

        // Algebraic scalar dissipation rate
        PropertyModelBase::Builder* the_builder = new AlgebraicScalarDiss::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "heat_loss" ) {

        // Heat loss
        PropertyModelBase::Builder* the_builder = new HeatLoss::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "scalsim_variance" ) {

        //Scalar variance using a scale similarity concept
        PropertyModelBase::Builder* the_builder = new ScalarVarianceScaleSim::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "norm_scalar_var") {

        //Normalized scalar variance based on second mixfrac moment
        PropertyModelBase::Builder* the_builder = new NormScalarVariance::Builder( prop_name, d_lab->d_materialManager );
        prop_factory.register_property_model( prop_name, the_builder );

      } else if ( prop_type == "scalar_diss") {

        //Scalar dissipation based on the transported squared gradient of mixture fraction for 2-eqn scalar var model
        PropertyModelBase::Builder* the_builder = new ScalarDissipation::Builder( prop_name, d_lab->d_materialManager );
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
void ExplicitSolver::registerCQMOMEqns(ProblemSpecP& db)
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
    proc0cout << "******* CQMOM Equation Registration ********\n";
    // Make the moment transport equations
    vector<int> temp_moment_index;
    for ( ProblemSpecP db_moments = cqmom_db->findBlock("Moment"); db_moments != nullptr; db_moments = db_moments->findNextBlock("Moment") ) {
      temp_moment_index.resize(0);
      db_moments->get("m", temp_moment_index);

      proc0cout << "creating a moment equation for: " << temp_moment_index << " as ";
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
      for (int i = 0; i<M; i++) {
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

void
ExplicitSolver::setupBoundaryConditions( const LevelP& level,
                                         SchedulerP& sched,
                                         const bool doingRestart ){

  //calls setupBCs which is setting up d_bcinformation in BoundaryCondition.cc
  // as parsed from the input file
  d_boundaryCondition->set_bc_information(level);

  //set a reference to BChelper map
  d_boundaryCondition->setBCHelper( &m_bcHelper );

  const MaterialSet* matls = d_lab->d_materialManager->allMaterials( "Arches" );

  //this is creating a WBCHelper for this specific Level (need one/level)
  //the check ensures that, in the case of multiple patches/core, we only
  //create one of them.
  auto iter_check = m_bcHelper.find(level->getID());
  if ( iter_check == m_bcHelper.end() ){
    m_bcHelper.insert(std::make_pair(level->getID(), scinew WBCHelper( level, sched, matls, m_arches_spec )));
  }

  //computes the area for each inlet using reduction variables
  m_bcHelper[level->getID()]->sched_computeBCAreaHelper( sched, level, matls );

  //copies the reduction area variable information on area to a double in the BndCond spec
  m_bcHelper[level->getID()]->sched_bindBCAreaHelper( sched, level, matls );

  //delete non-patch-local information on the old BC object
  d_boundaryCondition->prune_per_patch_bcinfo( sched, level, m_bcHelper[level->getID()] );

  if ( level->getIndex() == d_archesLevelIndex ){

     //check the sanity of the momentum BCs
    d_boundaryCondition->sched_checkMomBCs( sched, level, matls );

    //for RMCRT (why not on all levels?) compute cellType
    // using the BCInfoMap
    d_boundaryCondition->sched_cellTypeInit( sched, level, matls );

    // compute the cell area fraction
    d_boundaryCondition->sched_setAreaFraction( sched, level, matls, 0, true );

    // setup intrusion cell type
    // computes area and volume fractions for those cells that are intrusions
    // also sets a cell type in the same cell
    // also is computing an iterator(s) that is stored in the intrusion
    d_boundaryCondition->sched_setupNewIntrusionCellType( sched, level, matls, false );

    //AF must be called again to account for intrusions (can this be the ONLY call?)
    d_boundaryCondition->sched_setAreaFraction( sched, level, matls, 1, true );

    d_turbModel->sched_computeFilterVol( sched, level, matls );
  }
}

// An optional call for the application to check their reduction vars.
void
ExplicitSolver::checkReductionVars( const ProcessorGroup * pg,
                                    const PatchSubset    * patches,
                                    const MaterialSubset * matls,
                                          DataWarehouse  * old_dw,
                                          DataWarehouse  * new_dw )
{
    //SourceTermFactory& srcFactory = SourceTermFactory::self();
    //SourceTermFactory::SourceMap& sources = srcFactory.retrieve_all_sources();
    //for (SourceTermFactory::SourceMap::iterator isrc=sources.begin(); isrc !=sources.end(); isrc++){
      //SourceTermBase* src = isrc->second;
      //src->checkReductionVars( pg, patches, matls, old_dw, new_dw );
    //}
}
