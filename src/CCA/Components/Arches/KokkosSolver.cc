/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Arches/KokkosSolver.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/UPSHelper.h>
#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <CCA/Components/Arches/Task/TaskController.h>
//factories
#include <CCA/Components/Arches/Task/TaskFactoryHelper.h>
#include <CCA/Components/Arches/TurbulenceModels/TurbulenceModelFactory.h>
#include <CCA/Components/Arches/Utility/UtilityFactory.h>
#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/ChemMixV2/ChemMixFactory.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/Task/TaskFactoryHelper.h>
#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/LagrangianParticles/LagrangianParticleFactory.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/SourceTermsV2/SourceTermFactoryV2.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryConditionFactory.h>
//#include <CCA/Components/Arches/Task/SampleFactory.h>
#include <CCA/Components/Arches/ArchesExamples/ExampleFactory.h>

#include <sci_defs/kokkos_defs.h>

using namespace Uintah;

typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
typedef std::vector<std::string> SVec;

//--------------------------------------------------------------------------------------------------
KokkosSolver::KokkosSolver( MaterialManagerP& materialManager,
                            const ProcessorGroup* myworld,
                            SolverInterface* solver,
                            ApplicationCommon* arches )
  : NonlinearSolver ( myworld, arches )
  , m_materialManager   ( materialManager )
  , m_hypreSolver   ( solver )
{
  // delta t
  VarLabel* nonconstDelT =
    VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
  nonconstDelT->allowMultipleComputes();
  m_delTLabel = nonconstDelT;
  // Simulation time
  //VarLabel* m_simtime_label = VarLabel::create(simTime_name, simTime_vartype::getTypeDescription());
}

//--------------------------------------------------------------------------------------------------
KokkosSolver::~KokkosSolver()
{

  for (auto i = m_bcHelper.begin(); i != m_bcHelper.end(); i++){
    delete i->second;
  }
  m_bcHelper.clear();

  delete m_table_lookup;

  VarLabel::destroy(m_delTLabel);
 // VarLabel::destroy(m_simtime_label);

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::problemSetup( const ProblemSpecP     & input_db
                          ,       MaterialManagerP & materialManager
                          ,       GridP            & grid
                          )
{

  ProblemSpecP db = input_db;
  ProblemSpecP db_ks = db->findBlock("KokkosSolver");
  ProblemSpecP db_root = db->getRootNode();
  m_archesLevelIndex = grid->numLevels() - 1;

  ArchesCore::TaskController& tsk_controller = ArchesCore::TaskController::self();
  tsk_controller.parse_task_controller(db);

  db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->getAttribute("order", m_rk_order);
  proc0cout << " Time integrator: RK of order " << m_rk_order << "\n \n";

  commonProblemSetup( db_ks );

  if( db->findBlock("ParticleProperties") ) {
    std::string particle_type;
    db->findBlock("ParticleProperties")->getAttribute("type", particle_type);
    if ( particle_type == "coal" ) {
      CoalHelper& coal_helper = CoalHelper::self();
      coal_helper.parse_for_coal_info( db );
    } else {
      throw InvalidValue("Error: Particle type not recognized. Current types supported: coal",__FILE__,__LINE__);
    }
  }

  std::shared_ptr<UtilityFactory> UtilF(scinew UtilityFactory(m_arches));
  std::shared_ptr<TransportFactory> TransF(scinew TransportFactory(m_arches));
  std::shared_ptr<InitializeFactory> InitF(scinew InitializeFactory(m_arches));
  std::shared_ptr<ParticleModelFactory> PartModF(scinew ParticleModelFactory(m_arches));
  std::shared_ptr<LagrangianParticleFactory> LagF(scinew LagrangianParticleFactory(m_arches));
  std::shared_ptr<PropertyModelFactoryV2> PropModels(scinew PropertyModelFactoryV2(m_arches));
  std::shared_ptr<BoundaryConditionFactory> BC(scinew BoundaryConditionFactory(m_arches));
  std::shared_ptr<ChemMixFactory> TableModels(scinew ChemMixFactory(m_arches));
  std::shared_ptr<TurbulenceModelFactory> TurbModelF(scinew TurbulenceModelFactory(m_arches));
  std::shared_ptr<SourceTermFactoryV2> SourceTermV2(scinew SourceTermFactoryV2(m_arches));
  std::shared_ptr<ArchesExamples::ExampleFactory> ExampleFactoryF(scinew ArchesExamples::ExampleFactory(m_arches));

  m_task_factory_map.clear();
  m_task_factory_map.insert(std::make_pair("utility_factory",UtilF));
  m_task_factory_map.insert(std::make_pair("transport_factory",TransF));
  m_task_factory_map.insert(std::make_pair("initialize_factory",InitF));
  m_task_factory_map.insert(std::make_pair("particle_model_factory",PartModF));
  m_task_factory_map.insert(std::make_pair("lagrangian_factory",LagF));
  m_task_factory_map.insert(std::make_pair("property_models_factory", PropModels));
  m_task_factory_map.insert(std::make_pair("boundary_condition_factory", BC));
  m_task_factory_map.insert(std::make_pair("table_factory", TableModels));
  m_task_factory_map.insert(std::make_pair("turbulence_model_factory", TurbModelF));
  m_task_factory_map.insert(std::make_pair("source_term_factory",SourceTermV2));
  if(db->findBlock("ArchesExample"))
	  m_task_factory_map.insert(std::make_pair("ExampleFactory",ExampleFactoryF));

  typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
  proc0cout << "\n Registering and Building Tasks For: " << std::endl;
  for ( BFM::iterator i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->set_materialManager(m_materialManager);
    i->second->register_all_tasks(db);

  }

  //Set the hypre solver in the pressure eqn:
  if ( m_task_factory_map["transport_factory"]->has_task("[PressureEqn]")){
    PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(m_task_factory_map["transport_factory"]->retrieve_task("[PressureEqn]"));
    press_tsk->set_solver( m_hypreSolver );
    press_tsk->setup_solver( db );
  } else if ( m_task_factory_map["utility_factory"]->has_task("[PressureEqn]")){
    PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(m_task_factory_map["utility_factory"]->retrieve_task("[PressureEqn]"));
    press_tsk->set_solver( m_hypreSolver );
    press_tsk->setup_solver( db );
  }

  // Adds any additional lookup species as specified by the models.
  m_table_lookup = scinew TableLookup( m_materialManager );
  m_table_lookup->problemSetup( db );
  m_table_lookup->addLookupSpecies();

  std::string integrator;
  db_ks->getWithDefault("integrator", integrator, "ssprk");
  setSolver( integrator );

  proc0cout << std::endl;

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::sched_restartInitialize( const LevelP     & level
                                     ,       SchedulerP & sched
                                     )
{

  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  // Setup BCs
  setupBCs( level, sched, matls );

  //transport factory
  BFM::iterator i_trans_fac = m_task_factory_map.find("transport_factory");
  i_trans_fac->second->set_bcHelper( m_bcHelper[level->getID()]);

  // CFD only on the finest level
  if ( !level->hasFinerLevel() ){
    // initialize hypre objects
    if ( m_task_factory_map["transport_factory"]->has_task("[PressureEqn]")){
      PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(m_task_factory_map["transport_factory"]->retrieve_task("[PressureEqn]"));
      press_tsk->sched_restartInitialize( level, sched );
    }
  }

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::sched_restartInitializeTimeAdvance( const LevelP     & level
                                                ,       SchedulerP & sched
                                                )
{}


//--------------------------------------------------------------------------------------------------
void
KokkosSolver::computeTimestep( const LevelP     & level
                             ,       SchedulerP & sched
                             )
{

  using namespace ArchesCore;

  std::vector<std::string> var_names;
  std::string uname = parse_ups_for_role( UVELOCITY_ROLE, m_arches_spec, ArchesCore::default_uVel_name );
  var_names.push_back(uname);
  std::string vname = parse_ups_for_role( VVELOCITY_ROLE, m_arches_spec, ArchesCore::default_vVel_name );
  var_names.push_back(vname);
  std::string wname = parse_ups_for_role( WVELOCITY_ROLE, m_arches_spec, ArchesCore::default_wVel_name );
  var_names.push_back(wname);
  std::string muname = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, m_arches_spec, "NotFound" );
  var_names.push_back(muname);
  std::string rhoname = parse_ups_for_role( DENSITY_ROLE, m_arches_spec, "NotFound" );
  var_names.push_back(rhoname);

  bool found_all_vars = true;
  for ( auto i = var_names.begin(); i != var_names.end(); i++ ){
    if ( *i == "NotFound" ){
      found_all_vars = false;
      break;
    }
  }

  if ( found_all_vars ){

    auto taskDependencies = [&](Task* tsk) {

      // Actually compute the dt based on CFD variables.
      tsk->computes( m_delTLabel, level.get_rep() );

      m_uLabel = VarLabel::find( uname );
      m_vLabel = VarLabel::find( vname );
      m_wLabel = VarLabel::find( wname );
      m_rhoLabel = VarLabel::find( rhoname );
      m_tot_muLabel = VarLabel::find( muname );

      if ( level->getIndex() == m_archesLevelIndex ){
        tsk->requires( Task::NewDW, m_uLabel, Ghost::None, 0 );
        tsk->requires( Task::NewDW, m_vLabel, Ghost::None, 0 );
        tsk->requires( Task::NewDW, m_wLabel, Ghost::None, 0 );
        tsk->requires( Task::NewDW, m_rhoLabel, Ghost::None, 0 );
        tsk->requires( Task::NewDW, m_tot_muLabel, Ghost::None, 0 );
      }

      m_arches_spec->getRootNode()->findBlock("Time")->getWithDefault( "delt_init", m_dt_init, 1. );
    };

    //some race condition in kokkos::parallel_reduce. So combine all patches together in a single reduction task to avoid the multiple cpu threads calling parallel_reduce
    //temp work around until the permanent solution
    if ( Uintah::Parallel::usingDevice() ) {
	  LoadBalancer * lb = sched->getLoadBalancer();
	  //printf("warning: Creating per processor task for KokkosSolver::computeStableTimeStep due to race condition in kokkos cuda parallel_reduce %s %d\n", __FILE__, __LINE__);
	  create_portable_tasks(taskDependencies, this,
						  "KokkosSolver::computeStableTimeStep",
						  &KokkosSolver::computeStableTimeStep<UINTAH_CPU_TAG>,
						  &KokkosSolver::computeStableTimeStep<KOKKOS_OPENMP_TAG>,
						  &KokkosSolver::computeStableTimeStep<KOKKOS_CUDA_TAG>,
						  sched, lb->getPerProcessorPatchSet(level), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
    }
    else{
      create_portable_tasks(taskDependencies, this,
                          "KokkosSolver::computeStableTimeStep",
                          &KokkosSolver::computeStableTimeStep<UINTAH_CPU_TAG>,
                          &KokkosSolver::computeStableTimeStep<KOKKOS_OPENMP_TAG>,
                          &KokkosSolver::computeStableTimeStep<KOKKOS_CUDA_TAG>,
                          sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
    }

  } else {

    // Just set the dt to the init_dt because the CFD variables weren't found
    proc0cout << "\n ****************** WARNING ***************** " << std::endl;
    proc0cout << "  The CFD variable mapping was not complete   " << std::endl;
    proc0cout << "  because I could not find the appropriate    " << std::endl;
    proc0cout << "  variable mapping from <VarID>. As a result  " << std::endl;
    proc0cout << "  I am going to set dt to the delt_init as    " << std::endl;
    proc0cout << "  specified in the input file.                " << std::endl;
    proc0cout << " **************** END WARNING ***************\n " << std::endl;

    if ( !m_arches_spec->getRootNode()->findBlock("Time")->findBlock( "delt_init") ){
      throw ProblemSetupException("\n Error: Oops... please specify a delt_init in your input file.\n", __FILE__, __LINE__ );
    }

    auto taskDependencies = [&](Task* tsk) {
      m_arches_spec->getRootNode()->findBlock("Time")->require("delt_init", m_dt_init );
      proc0cout << " Note: Setting constant dt = " << m_dt_init << "\n" << std::endl;
      tsk->computes( m_delTLabel, level.get_rep() );
    };

    create_portable_tasks(taskDependencies, this,
                          "KokkosSolver::setTimeStep",
                          &KokkosSolver::setTimeStep<UINTAH_CPU_TAG>,
                          &KokkosSolver::setTimeStep<KOKKOS_OPENMP_TAG>,
                          //&KokkosSolver::setTimeStep<KOKKOS_CUDA_TAG>,
                          sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);

  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void
KokkosSolver::computeStableTimeStep( const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       OnDemandDataWarehouse                * old_dw
                                   ,       OnDemandDataWarehouse                * new_dw
                                   ,       UintahParams                         & uintahParams
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   )
{

  const Level* level = getLevel(patches);

  double dt = m_dt_init;

  if ( level->getIndex() == m_archesLevelIndex ){
    for (int p = 0; p < patches->size(); p++) {

      const Patch* patch = patches->get(p);
      int archIndex = 0; // only one arches material
      int indx = m_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

      Vector Dx = patch->dCell();

      auto u   = new_dw->getConstGridVariable<constSFCXVariable<double>, double, MemSpace>( m_uLabel     , indx, patch, Ghost::None, 0 );
      auto v   = new_dw->getConstGridVariable<constSFCYVariable<double>, double, MemSpace>( m_vLabel     , indx, patch, Ghost::None, 0 );
      auto w   = new_dw->getConstGridVariable<constSFCZVariable<double>, double, MemSpace>( m_wLabel     , indx, patch, Ghost::None, 0 );
      auto rho = new_dw->getConstGridVariable<constCCVariable<double>,   double, MemSpace>( m_rhoLabel   , indx, patch, Ghost::None, 0 );
      auto mu  = new_dw->getConstGridVariable<constCCVariable<double>,   double, MemSpace>( m_tot_muLabel, indx, patch, Ghost::None, 0 );

      Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

      const double small_num = 1.e-10;

      const double dx = Dx.x();
      const double dy = Dx.y();
      const double dz = Dx.z();

      Uintah::parallel_reduce_min( execObj, range, KOKKOS_LAMBDA ( const int i, const int j, const int k, double & m_dt ) {

        m_dt = 1. / ( std::fabs( u(i,j,k) ) / dx +
                      std::fabs( v(i,j,k) ) / dy +
                      std::fabs( w(i,j,k) ) / dz +
                      mu(i,j,k) / rho(i,j,k) *
                      ( 1. / ( dx * dx ) + 1. / ( dy * dy ) + 1. / ( dz * dz ) ) +
                      small_num);

      }, dt );

      new_dw->put(delt_vartype(dt), m_delTLabel, level);
    }
  } else {
    new_dw->put(delt_vartype(9e99), m_delTLabel, level);
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void
KokkosSolver::setTimeStep( const PatchSubset                          * patches
                         , const MaterialSubset                       * matls
                         ,       OnDemandDataWarehouse                * old_dw
                         ,       OnDemandDataWarehouse                * new_dw
                         ,       UintahParams                         & uintahParams
                         ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                         )
{
  const Level* level = getLevel(patches);
  if ( level->getIndex() == m_archesLevelIndex ){
    for (int p = 0; p < patches->size(); p++) {
      new_dw->put(delt_vartype(m_dt_init), m_delTLabel, level);
    }
  } else {
    new_dw->put(delt_vartype(9e99), m_delTLabel, level);
  }
}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::sched_initialize( const LevelP& level,
                                SchedulerP& sched,
                                const bool doing_restart )
{
  using namespace Uintah::ArchesCore;

  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  if ( level->getIndex() == m_archesLevelIndex ){

    // Setup BCs
    setupBCs( level, sched, matls );

    if ( m_nonlinear_solver == SANDBOX ){

      SandBox_initialize( level, sched );

    } else {

      SSPRKSolve_initialize( level, sched );

    }

    for (auto i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ){
      std::map<std::string, TaskFactoryBase::GhostHelper>& the_ghost_info = i->second->get_max_ghost_info();
      insert_max_ghost( the_ghost_info );
      //SCI_DEBUG for printing information per task.
      i->second->print_variable_max_ghost();
    }

    // SCI_DEBUG for printing across ALL tasks.
    print_variable_max_ghost();

  }
}

//--------------------------------------------------------------------------------------------------
int
KokkosSolver::sched_nonlinearSolve( const LevelP & level,
                                    SchedulerP & sched )
{
  using namespace Uintah::ArchesCore;

  //clear the factory ghost lists from information inserted from scheduleInitialize
  for ( auto i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ){
    (*i->second).clear_max_ghost_list();
  }

  //also clear the master ghost list
  clear_max_ghost_list();


  if ( m_nonlinear_solver == SANDBOX ){

    SandBox( level, sched );

  } else {

    SSPRKSolve( level, sched );

  }

  for (auto i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ){
    std::map<std::string, TaskFactoryBase::GhostHelper>& the_ghost_info = i->second->get_max_ghost_info();
    insert_max_ghost( the_ghost_info );
    //SCI_DEBUG for printing across tasks per factory
    i->second->print_variable_max_ghost();
  }

  //SCI_DEBUG for printing across all tasks
  print_variable_max_ghost();

  return 0;

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::setupBCs( const LevelP      & level
                      ,       SchedulerP  & sched
                      , const MaterialSet * matls
                      )
{
  //boundary condition helper
  m_bcHelper.insert(std::make_pair(level->getID(), scinew WBCHelper( level, sched, matls, m_arches_spec )));

  //computes the area for each inlet through the use of a reduction variables
  m_bcHelper[level->getID()]->sched_computeBCAreaHelper( sched, level, matls );

  //copies the reduction area variable information on area to a double in the BndCond spec
  m_bcHelper[level->getID()]->sched_bindBCAreaHelper( sched, level, matls );

  proc0cout << "\n Setting BCHelper for all Factories. \n" << std::endl;
  for ( BFM::iterator i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ) {
    i->second->set_bcHelper( m_bcHelper[level->getID()]);
  }
}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::SSPRKSolve_initialize( const LevelP & level,
                                     SchedulerP & sched )
{
  using namespace Uintah::ArchesCore;

  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  TaskController& tsk_controller = TaskController::self();
  const TaskController::Packing& packed_info = tsk_controller.get_packing_info();

  // initialize hypre objects
  if ( m_task_factory_map["transport_factory"]->has_task("[PressureEqn]")){
    PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(m_task_factory_map["transport_factory"]->retrieve_task("[PressureEqn]"));
    press_tsk->sched_Initialize( level, sched );
  }

  // grid spacing etc...
  m_task_factory_map["utility_factory"]->schedule_task( "grid_info", TaskInterface::INITIALIZE, level, sched, matls );

  // set the volume fractions
  m_task_factory_map["utility_factory"]->schedule_task( "vol_fraction_calc", TaskInterface::INITIALIZE, level, sched, matls );

  // transport factory
  m_task_factory_map["transport_factory"]->schedule_task_group( "scalar_rhs_builders", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );
  m_task_factory_map["transport_factory"]->schedule_task_group( "diffusion_flux_builders", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );
  m_task_factory_map["transport_factory"]->schedule_task_group( "dqmom_rhs_builders", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );
  m_task_factory_map["transport_factory"]->schedule_task_group( "dqmom_diffusion_flux_builders", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );
  m_task_factory_map["transport_factory"]->schedule_task_group( "pressure_eqn", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );
  m_task_factory_map["transport_factory"]->schedule_task_group( "mom_rhs_builders", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // property factory
  m_task_factory_map["property_models_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // generic field initializer
  m_task_factory_map["initialize_factory"]->schedule_task_group( "phi_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // boundary condition factory
  m_task_factory_map["boundary_condition_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  m_task_factory_map["boundary_condition_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, packed_info.global, level, sched, matls );

  // Need to apply BC's after everything is initialized
  m_task_factory_map["transport_factory"]->schedule_task_group( "scalar_rhs_builders", TaskInterface::BC, packed_info.global, level, sched, matls );

  // set tabulated initial conditions
  m_task_factory_map["table_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // Need to updated table BC
  m_task_factory_map["table_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, packed_info.global, level, sched, matls );

  // variable that computed with density such as rho*u
  m_task_factory_map["initialize_factory"]->schedule_task_group( "rho_phi_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // actually computes rho*phi here
  m_task_factory_map["transport_factory"]->schedule_task_group("phi_from_rho_phi", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  m_task_factory_map["transport_factory"]->schedule_task_group("phi_from_rho_phi", TaskInterface::BC, packed_info.global, level, sched, matls );

  m_task_factory_map["transport_factory"]->schedule_task_group( "dqmom_rhs_builders", TaskInterface::BC, packed_info.global, level, sched, matls );

  m_task_factory_map["transport_factory"]->schedule_task_group( "dqmom_ic_from_wic", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // Need to apply BC's after everything is initialized
  m_task_factory_map["transport_factory"]->schedule_task_group( "momentum_conv", TaskInterface::BC, packed_info.global, level, sched, matls );

  m_task_factory_map["transport_factory"]->schedule_task_group("u_from_rhou", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  m_task_factory_map["transport_factory"]->schedule_task_group("u_from_rhou", TaskInterface::BC, packed_info.global, level, sched, matls );

  m_task_factory_map["property_models_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, packed_info.global, level, sched, matls );

  // recomputing the CCVelocity calc here:
  if ( m_task_factory_map["property_models_factory"]->has_task("compute_cc_velocities")){
    m_task_factory_map["property_models_factory"]->schedule_task( "compute_cc_velocities", TaskInterface::TIMESTEP_EVAL, level, sched,
      matls, 1 );
  }

  // particle models
  m_task_factory_map["particle_model_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // turbulence model
  m_task_factory_map["turbulence_model_factory"]->schedule_task_group( "momentum_closure", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  // source_term_kokkos_factory
  m_task_factory_map["source_term_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );

  m_task_factory_map["utility_factory"]->schedule_task( "forced_turbulence", TaskInterface::INITIALIZE, level, sched, matls, 0, false, true );

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::SSPRKSolve( const LevelP & level,
                          SchedulerP & sched )
{

  using namespace Uintah::ArchesCore;

  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  BFM::iterator i_util_fac = m_task_factory_map.find("utility_factory");
  BFM::iterator i_transport = m_task_factory_map.find("transport_factory");
  BFM::iterator i_prop_fac = m_task_factory_map.find("property_models_factory");
  BFM::iterator i_bc_fac = m_task_factory_map.find("boundary_condition_factory");
  BFM::iterator i_table_fac = m_task_factory_map.find("table_factory");
  BFM::iterator i_source_fac = m_task_factory_map.find("source_term_factory");
  BFM::iterator i_turb_model_fac = m_task_factory_map.find("turbulence_model_factory");
  BFM::iterator i_particle_model_fac = m_task_factory_map.find("particle_model_factory");

  TaskFactoryBase::TaskMap all_bc_tasks = i_bc_fac->second->retrieve_all_tasks();
  TaskFactoryBase::TaskMap all_table_tasks = i_table_fac->second->retrieve_all_tasks();

  TaskController& tsk_controller = TaskController::self();
  const TaskController::Packing& packed_info = tsk_controller.get_packing_info();

  // ----------------- Timestep Initialize ---------------------------------------------------------

  i_util_fac->second->schedule_task( "grid_info", TaskInterface::TIMESTEP_INITIALIZE, level, sched,
    matls );

  i_util_fac->second->schedule_task( "vol_fraction_calc", TaskInterface::TIMESTEP_INITIALIZE, level,
    sched, matls );

  i_transport->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    packed_info.global, level, sched, matls );

  i_particle_model_fac->second->schedule_task_group("all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    packed_info.global, level, sched, matls );

  i_prop_fac->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    packed_info.global, level, sched, matls );

  i_source_fac->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    packed_info.global, level, sched, matls );

  i_turb_model_fac->second->schedule_task_group( "momentum_closure", TaskInterface::TIMESTEP_INITIALIZE,
    packed_info.global, level, sched, matls );

  i_bc_fac->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    packed_info.global, level, sched, matls );

  m_task_factory_map["table_factory"]->schedule_task_group( "all_tasks",
    TaskInterface::TIMESTEP_INITIALIZE, packed_info.global, level, sched, matls );

  // ----------------- SSP RK LOOP -----------------------------------------------------------------

  for ( int time_substep = 0; time_substep < m_rk_order; time_substep++ ){

    // utility Factory
    i_util_fac->second->schedule_task_group( "mass_flow_rate",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // pre-update properties/source tasks)
    i_prop_fac->second->schedule_task_group( "pre_update_property_models",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // compute momentum closure
    i_turb_model_fac->second->schedule_task_group("momentum_closure",
      TaskInterface::TIMESTEP_EVAL, packed_info.turbulence, level, sched, matls, time_substep );

    // pre-update properties/source tasks)
    i_prop_fac->second->schedule_task_group( "diffusion_property_models",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

   // (pre-update source terms)
    i_source_fac->second->schedule_task_group( "pre_update_source_tasks",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls , time_substep );

    // ** DQMOM **
    // compute particle face velocities
    i_transport->second->schedule_task_group("dqmom_diffusion_flux_builders",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group("dqmom_rhs_builders",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_particle_model_fac->second->schedule_task_group("dqmom_transport_variables",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_particle_model_fac->second->schedule_task_group("particle_models",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    if ( i_particle_model_fac->second->has_task("[DQMOMNoInversion]")){
      i_particle_model_fac->second->schedule_task("[DQMOMNoInversion]",
        TaskInterface::TIMESTEP_EVAL, level, sched, matls, time_substep );
    }

    i_transport->second->schedule_task_group("dqmom_fe_update",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group("dqmom_rhs_builders",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // computes ic from w*ic  :
    i_transport->second->schedule_task_group("dqmom_ic_from_wic",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group("dqmom_ic_from_wic",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group("dqmom_fe_update",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    i_particle_model_fac->second->schedule_task_group("particle_properties",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_particle_model_fac->second->schedule_task_group("deposition_models",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );


    // ** SCALARS **
    // PRE-PROJECTION
    i_transport->second->schedule_task_group("diffusion_flux_builders",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // now construct the RHS:
    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // update phi
    i_transport->second->schedule_task_group("scalar_update",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // Compute density star
    i_prop_fac->second->schedule_task( "density_star", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );

    // get phi from rho*phi :
    i_transport->second->schedule_task_group("phi_from_rho_phi",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // bc factory tasks
    i_bc_fac->second->schedule_task_group("all_tasks",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // Scalar BCs
    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // Set BC for rho*phi
    i_transport->second->schedule_task_group("phi_from_rho_phi",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // ** TABLE LOOKUP **
    i_table_fac->second->schedule_task_group("all_tasks",
     TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_table_fac->second->schedule_task_group("all_tasks",
     TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    if (time_substep > 0) {
      // time average using rk method
      i_transport->second->schedule_task_group("rk_time_ave",
        TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );
     }
     // Compute density rk
     i_prop_fac->second->schedule_task( "density_rk", TaskInterface::TIMESTEP_EVAL,
       level, sched, matls, time_substep, false, true );

     if (time_substep > 0) {
      // get phi from phi*rho
      i_transport->second->schedule_task_group("phi_from_rho_phi",
        TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

      // Scalar BCs
      i_transport->second->schedule_task_group("scalar_rhs_builders",
        TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

      // Set BC for rho*phi
      i_transport->second->schedule_task_group("phi_from_rho_phi",
        TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

      // ** TABLE LOOKUP **
      i_table_fac->second->schedule_task_group("all_tasks",
       TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

      i_table_fac->second->schedule_task_group("all_tasks",
       TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

     }

    // bc factory tasks
    i_bc_fac->second->schedule_task_group("all_tasks",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // ** MOMENTUM **

    i_transport->second->schedule_task_group( "momentum_stress_tensor", TaskInterface::TIMESTEP_EVAL,
      packed_info.global, level, sched, matls, time_substep );

    // compute wall momentum closure
    //i_turb_model_fac->second->schedule_task_group("wall_momentum_closure",
    //  TaskInterface::TIMESTEP_EVAL, packed_info.turbulence, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group( "momentum_conv", TaskInterface::TIMESTEP_EVAL,
      packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group( "momentum_fe_update", TaskInterface::TIMESTEP_EVAL,
      packed_info.global, level, sched, matls, time_substep );

    // apply boundary conditions
     i_transport->second->schedule_task_group( "momentum_conv", TaskInterface::BC, false,
                                              level, sched, matls, time_substep );

    //Compute drhodt
    i_prop_fac->second->schedule_task( "drhodt", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );

    i_util_fac->second->schedule_task( "forced_turbulence", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );

    // ** PRESSURE PROJECTION **
    if ( i_transport->second->has_task("[PressureEqn]")){

      //APPLY BC for OULET AND PRESSURE PER STAS'S BCs
      i_transport->second->schedule_task("[VelRhoHatBC]", TaskInterface::ATOMIC,
                                          level, sched, matls, time_substep );

      PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(
        i_transport->second->retrieve_task("[PressureEqn]"));

      // Compute the coeffificients
      i_transport->second->schedule_task("[PressureEqn]", TaskInterface::TIMESTEP_EVAL,
                                          level, sched, matls, time_substep );

      // Compute the boundary conditions on the linear system
      i_transport->second->schedule_task("[PressureEqn]", TaskInterface::BC,
                                          level, sched, matls, time_substep );

      // Solve it - calling out to hypre external lib
      press_tsk->solve(level, sched, time_substep);

      // Apply boundary conditions on the pressure field. The BCs are initially applied on the
      // linear system, however, the resulting pressure field also needs BCs so that the correction
      // to the velocities is done correctly.
      i_transport->second->schedule_task("[PressureBC]", TaskInterface::ATOMIC,
                                          level, sched, matls, time_substep );

      // Correct velocities
      i_transport->second->schedule_task("[AddPressGradient]", TaskInterface::ATOMIC,
                                          level, sched, matls, time_substep );

      // apply boundary conditions
      i_transport->second->schedule_task_group( "momentum_conv", TaskInterface::BC, false,
                                               level, sched, matls, time_substep );

    }

    // Get velocities from momemtum: u = x-mom/rho
    i_transport->second->schedule_task_group("u_from_rhou",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // apply boundary conditions to rho*ui
    i_transport->second->schedule_task_group("u_from_rhou",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    i_prop_fac->second->schedule_task( "compute_cc_velocities", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );

    //Continuity check
    i_prop_fac->second->schedule_task( "continuity_check", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );

    // compute kinetic energy
    i_prop_fac->second->schedule_task_group( "post_update_property_models",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

  } // RK Integrator

  //Variable stats stuff
  i_prop_fac->second->schedule_task_group( "variable_stat_models",
    TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, 1 );

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::SandBox_initialize( const LevelP & level,
                                     SchedulerP & sched )
{
  using namespace ArchesCore;

  TaskController& tsk_controller = TaskController::self();
  const TaskController::Packing& packed_info = tsk_controller.get_packing_info();
  const int time_substep = 0;
  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  BFM::iterator i_prop_fac = m_task_factory_map.find("property_models_factory");
  BFM::iterator i_util_fac = m_task_factory_map.find("utility_factory");
  BFM::iterator i_exam_fac = m_task_factory_map.find("ExampleFactory");

  if(i_exam_fac==m_task_factory_map.end()){	//schedule default sandbox if there are no examples.
    // initialize hypre objects
    if ( m_task_factory_map["utility_factory"]->has_task("[PressureEqn]")){
      PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(m_task_factory_map["utility_factory"]->retrieve_task("[PressureEqn]"));
      press_tsk->sched_Initialize( level, sched );
    }

    i_prop_fac->second->schedule_task_group( "all_tasks",
      TaskInterface::INITIALIZE, packed_info.global, level, sched, matls, time_substep );

    i_util_fac->second->schedule_task_group( "all_tasks",
      TaskInterface::INITIALIZE, packed_info.global, level, sched, matls, time_substep );

    m_task_factory_map["initialize_factory"]->schedule_task_group( "all_tasks",
      TaskInterface::INITIALIZE, packed_info.global, level, sched, matls );
  }
  else{//schedule examples if present
    i_exam_fac->second->schedule_task_group( "all_tasks",
      TaskInterface::INITIALIZE, packed_info.global, level, sched, matls, time_substep );
  }

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::SandBox( const LevelP     & level
                           , SchedulerP & sched )
{

  using namespace ArchesCore;

  TaskController& tsk_controller = TaskController::self();
  const TaskController::Packing& packed_info = tsk_controller.get_packing_info();
  const int time_substep = 0;
  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  BFM::iterator i_prop_fac = m_task_factory_map.find("property_models_factory");
  BFM::iterator i_util_fac = m_task_factory_map.find("utility_factory");
  BFM::iterator i_exam_fac = m_task_factory_map.find("ExampleFactory");

  if(i_exam_fac==m_task_factory_map.end()){	//schedule default sandbox if there are no examples.
    PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(
                             i_util_fac->second->retrieve_task("[PressureEqn]"));

    // ----------------- Time Integration ------------------------------------------------------------
    i_prop_fac->second->schedule_task_group("all_tasks",
      TaskInterface::TIMESTEP_INITIALIZE, packed_info.global, level, sched, matls, time_substep );

    i_util_fac->second->schedule_task_group("all_tasks",
      TaskInterface::TIMESTEP_INITIALIZE, packed_info.global, level, sched, matls, time_substep );

    //m_task_factory_map["utility_factory"]->schedule_task( "grid_info", TaskInterface::INITIALIZE, level, sched, matls );
    i_prop_fac->second->schedule_task_group( "all_tasks",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_util_fac->second->schedule_task_group( "all_tasks",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_util_fac->second->schedule_task_group( "all_tasks",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // Solve it - calling out to hypre external lib
    press_tsk->solve(level, sched, time_substep);
  }
  else{	//schedule examples if present
    i_exam_fac->second->schedule_task_group("all_tasks",
        TaskInterface::TIMESTEP_INITIALIZE, packed_info.global, level, sched, matls, time_substep );

    i_exam_fac->second->schedule_task_group( "all_tasks",
        TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_exam_fac->second->schedule_task_group( "all_tasks",
        TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );
  }

}
