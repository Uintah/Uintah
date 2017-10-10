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

#include <CCA/Components/Arches/KokkosSolver.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/UPSHelper.h>
#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
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
//END NEW TASK INTERFACE STUFF

using namespace Uintah;

typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
typedef std::vector<std::string> SVec;

//--------------------------------------------------------------------------------------------------
KokkosSolver::KokkosSolver( SimulationStateP& shared_state,
                            const ProcessorGroup* myworld,
                            SolverInterface* solver )
  : NonlinearSolver( myworld ), m_sharedState(shared_state), m_hypreSolver(solver)
{}

//--------------------------------------------------------------------------------------------------
KokkosSolver::~KokkosSolver(){
  for (auto i = m_bcHelper.begin(); i != m_bcHelper.end(); i++){
    delete i->second;
  }
  m_bcHelper.clear();

  delete m_table_lookup;

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::sched_restartInitialize( const LevelP& level, SchedulerP& sched )
{

  const MaterialSet* matls = m_sharedState->allArchesMaterials();

  // Setup BCs
  setupBCs( level, sched, matls );

  //transport factory
  BFM::iterator i_trans_fac = m_task_factory_map.find("transport_factory");
  i_trans_fac->second->set_bcHelper( m_bcHelper[level->getID()]);

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched )
{}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::problemSetup( const ProblemSpecP& input_db,
                            SimulationStateP& state,
                            GridP& grid )
{

  ProblemSpecP db = input_db;
  ProblemSpecP db_ks = db->findBlock("KokkosSolver");
  ProblemSpecP db_root = db->getRootNode();

  ArchesCore::TaskController& tsk_controller = ArchesCore::TaskController::self();
  tsk_controller.parse_task_controller(db);

  db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->getAttribute("order", m_rk_order);
  proc0cout << " Time integrator: RK of order " << m_rk_order << "\n \n";

  commonProblemSetup( db_ks );

  std::shared_ptr<UtilityFactory> UtilF(scinew UtilityFactory());
  std::shared_ptr<TransportFactory> TransF(scinew TransportFactory());
  std::shared_ptr<InitializeFactory> InitF(scinew InitializeFactory());
  std::shared_ptr<ParticleModelFactory> PartModF(scinew ParticleModelFactory());
  std::shared_ptr<LagrangianParticleFactory> LagF(scinew LagrangianParticleFactory());
  std::shared_ptr<PropertyModelFactoryV2> PropModels(scinew PropertyModelFactoryV2());
  std::shared_ptr<BoundaryConditionFactory> BC(scinew BoundaryConditionFactory());
  std::shared_ptr<ChemMixFactory> TableModels(scinew ChemMixFactory());
  std::shared_ptr<TurbulenceModelFactory> TurbModelF(scinew TurbulenceModelFactory());
  std::shared_ptr<SourceTermFactoryV2> SourceTermV2(scinew SourceTermFactoryV2());

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

  typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
  proc0cout << "\n Registering Tasks For: " << std::endl;
  for ( BFM::iterator i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->set_shared_state(m_sharedState);
    i->second->register_all_tasks(db);

  }

  proc0cout << "\n Building Tasks For: " << std::endl;

  for ( BFM::iterator i = m_task_factory_map.begin(); i != m_task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->build_all_tasks(db);

  }

  //Set the hypre solver in the pressure eqn:
  if ( m_task_factory_map["transport_factory"]->has_task("build_pressure_system")){
    PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(m_task_factory_map["transport_factory"]->retrieve_task("build_pressure_system"));
    press_tsk->set_solver( m_hypreSolver );
    press_tsk->setup_solver( db );
  }

  // Adds any additional lookup species as specified by the models.
  m_table_lookup = scinew TableLookup( m_sharedState );
  m_table_lookup->problemSetup( db );
  m_table_lookup->addLookupSpecies();

  std::string integrator;
  db_ks->getWithDefault("integrator", integrator, "ssprk");
  setSolver( integrator );

  proc0cout << std::endl;

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::computeTimestep(const LevelP& level, SchedulerP& sched)
{

  using namespace ArchesCore;

  std::vector<std::string> var_names;
  std::string uname = parse_ups_for_role( UVELOCITY, m_arches_spec, "NotFound" );
  var_names.push_back(uname);
  std::string vname = parse_ups_for_role( VVELOCITY, m_arches_spec, "NotFound" );
  var_names.push_back(vname);
  std::string wname = parse_ups_for_role( WVELOCITY, m_arches_spec, "NotFound" );
  var_names.push_back(wname);
  std::string muname = parse_ups_for_role( TOTAL_VISCOSITY, m_arches_spec, "NotFound" );
  var_names.push_back(muname);
  std::string rhoname = parse_ups_for_role( DENSITY, m_arches_spec, "NotFound" );
  var_names.push_back(rhoname);

  bool found_all_vars = true;
  for ( auto i = var_names.begin(); i != var_names.end(); i++ ){
    if ( *i == "NotFound" ){
      found_all_vars = false;
      break;
    }
  }

  if ( found_all_vars ){

    Task* tsk = scinew Task( "KokkosSolver::computeStableTimeStep", this,
                             &KokkosSolver::computeStableTimeStep);

    // Actually compute the dt based on CFD variables.

    tsk->computes( m_sharedState->get_delt_label(), level.get_rep() );

    m_uLabel = VarLabel::find( uname );
    m_vLabel = VarLabel::find( vname );
    m_wLabel = VarLabel::find( wname );
    m_rhoLabel = VarLabel::find( rhoname );
    m_tot_muLabel = VarLabel::find( muname );

    tsk->requires( Task::NewDW, m_uLabel, Ghost::None, 0 );
    tsk->requires( Task::NewDW, m_vLabel, Ghost::None, 0 );
    tsk->requires( Task::NewDW, m_wLabel, Ghost::None, 0 );
    tsk->requires( Task::NewDW, m_rhoLabel, Ghost::None, 0 );
    tsk->requires( Task::NewDW, m_tot_muLabel, Ghost::None, 0 );

    m_arches_spec->getRootNode()->findBlock("Time")->getWithDefault( "delt_init", m_dt_init, 1. );

    sched->addTask( tsk, level->eachPatch(), m_sharedState->allArchesMaterials() );

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

    Task* tsk = scinew Task( "KokkosSolver::setTimeStep", this,
                             &KokkosSolver::setTimeStep );

    tsk->computes( m_sharedState->get_delt_label(), level.get_rep() );

    sched->addTask( tsk, level->eachPatch(), m_sharedState->allArchesMaterials() );

  }

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::computeStableTimeStep( const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw )
{

  const Level* level = getLevel(patches);
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = m_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    Vector Dx = patch->dCell();

    constSFCXVariable<double> u;
    constSFCYVariable<double> v;
    constSFCZVariable<double> w;
    constCCVariable<double> rho;
    constCCVariable<double> mu;

    new_dw->get( u, m_uLabel, indx, patch, Ghost::None, 0 );
    new_dw->get( v, m_vLabel, indx, patch, Ghost::None, 0 );
    new_dw->get( w, m_wLabel, indx, patch, Ghost::None, 0 );
    new_dw->get( rho, m_rhoLabel, indx, patch, Ghost::None, 0 );
    new_dw->get( mu, m_tot_muLabel, indx, patch, Ghost::None, 0 );

    double dt = m_dt_init;

    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
    // Shouldn't this be a reduce???
    // When I tried parallel_reduce I didn't get the behavior I expected.
    Uintah::parallel_for( range, [&]( int i, int j, int k ) {

      const double small_num = 1.e-10;
      const double dx = Dx.x();
      const double dy = Dx.y();
      const double dz = Dx.z();

      double denom_dt = std::abs( u(i,j,k) ) / dx +
                        std::abs( v(i,j,k) ) / dy +
                        std::abs( w(i,j,k) ) / dz +
                        mu(i,j,k) / rho(i,j,k) * (
                          1./(dx*dx) + 1./(dy*dy) + 1./(dz*dz)
                        ) + small_num;

      dt = std::min( 1.0 / denom_dt, dt );

    });

    new_dw->put(delt_vartype(dt), m_sharedState->get_delt_label(), level);

  }
}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::setTimeStep( const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw ){
  const Level* level = getLevel(patches);
  for (int p = 0; p < patches->size(); p++) {
    new_dw->put(delt_vartype(m_dt_init), m_sharedState->get_delt_label(), level);
  }
}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::initialize( const LevelP& level, SchedulerP& sched, const bool doing_restart )
{
  const MaterialSet* matls = m_sharedState->allArchesMaterials();
  //bool is_restart = false;
  const bool pack_tasks = true;
  const bool dont_pack_tasks = false;

  // Setup BCs
  setupBCs( level, sched, matls );

  // grid spacing etc...
  m_task_factory_map["utility_factory"]->schedule_task( "grid_info", TaskInterface::INITIALIZE, level, sched, matls );

  // set the volume fractions
  m_task_factory_map["utility_factory"]->schedule_task( "vol_fraction_calc", TaskInterface::INITIALIZE, level, sched, matls );

  //transport factory
  m_task_factory_map["transport_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

  //property factory
  m_task_factory_map["property_models_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

  // turbulence model
  m_task_factory_map["turbulence_model_factory"]->schedule_task_group( "momentum_closure", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

  //source_term_kokkos_factory
  m_task_factory_map["source_term_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, dont_pack_tasks, level, sched, matls );

  // generic field initializer
  m_task_factory_map["initialize_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, pack_tasks, level, sched, matls );

  // boundary condition factory
  m_task_factory_map["boundary_condition_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, pack_tasks, level, sched, matls );

  // tabulated factory
  m_task_factory_map["table_factory"]->schedule_task_group( "all_tasks", TaskInterface::INITIALIZE, pack_tasks, level, sched, matls );

  //Need to apply BC's after everything is initialized
  m_task_factory_map["table_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, pack_tasks, level, sched, matls );
  m_task_factory_map["transport_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, pack_tasks, level, sched, matls );
  m_task_factory_map["boundary_condition_factory"]->schedule_task_group( "all_tasks", TaskInterface::BC, pack_tasks, level, sched, matls );

  // Compute density
  m_task_factory_map["property_models_factory"]->schedule_task( "density_guess",
    TaskInterface::INITIALIZE, level, sched, matls, 0, true, true );

  //Recompute velocities from momentum:
  m_task_factory_map["property_models_factory"]->schedule_task( "u_from_rho_u",
    TaskInterface::INITIALIZE, level, sched, matls, 0, true, true );
  m_task_factory_map["property_models_factory"]->schedule_task( "compute_cc_velocities",
    TaskInterface::INITIALIZE, level, sched, matls, 0, true, true );

}

//--------------------------------------------------------------------------------------------------
int
KokkosSolver::nonlinearSolve( const LevelP& level,
                              SchedulerP& sched )
{
  const bool pack_tasks = true;
  //const bool dont_pack_tasks = false;

  const MaterialSet* matls = m_sharedState->allArchesMaterials();

  BFM::iterator i_util_fac = m_task_factory_map.find("utility_factory");
  BFM::iterator i_transport = m_task_factory_map.find("transport_factory");
  BFM::iterator i_prop_fac = m_task_factory_map.find("property_models_factory");
  BFM::iterator i_source_fac = m_task_factory_map.find("source_term_factory");
  BFM::iterator i_bc_fac = m_task_factory_map.find("boundary_condition_factory");
  BFM::iterator i_table_fac = m_task_factory_map.find("table_factory");
  BFM::iterator i_turb_model_fac = m_task_factory_map.find("turbulence_model_factory");

  TaskFactoryBase::TaskMap all_bc_tasks = i_bc_fac->second->retrieve_all_tasks();

  // ----------------- Timestep Initialize ---------------------------------------------------------

  i_util_fac->second->schedule_task( "grid_info", TaskInterface::TIMESTEP_INITIALIZE, level, sched,
    matls );

  i_util_fac->second->schedule_task( "vol_fraction_calc", TaskInterface::TIMESTEP_INITIALIZE, level,
    sched, matls );

  i_transport->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    pack_tasks, level, sched, matls );

  i_prop_fac->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    pack_tasks, level, sched, matls );

  i_source_fac->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    pack_tasks, level, sched, matls );

  i_turb_model_fac->second->schedule_task_group( "momentum_closure", TaskInterface::TIMESTEP_INITIALIZE,
    pack_tasks, level, sched, matls );

  i_bc_fac->second->schedule_task_group( "all_tasks", TaskInterface::TIMESTEP_INITIALIZE,
    pack_tasks, level, sched, matls );

  m_task_factory_map["table_factory"]->schedule_task_group( "all_tasks",
    TaskInterface::TIMESTEP_INITIALIZE, pack_tasks, level, sched, matls );

  // --------------- Actual Solve ------------------------------------------------------------------

  if ( m_nonlinear_solver == SANDBOX ){

    SandBox( level, sched );

  } else {

    SSPRKSolve( level, sched );

  }

  return 0;

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::setupBCs( const LevelP& level, SchedulerP& sched, const MaterialSet* matls ){
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
KokkosSolver::SSPRKSolve( const LevelP& level, SchedulerP& sched ){

  using namespace Uintah::ArchesCore;

  const MaterialSet* matls = m_sharedState->allArchesMaterials();

  BFM::iterator i_util_fac = m_task_factory_map.find("utility_factory");
  BFM::iterator i_transport = m_task_factory_map.find("transport_factory");
  BFM::iterator i_prop_fac = m_task_factory_map.find("property_models_factory");
  BFM::iterator i_bc_fac = m_task_factory_map.find("boundary_condition_factory");
  BFM::iterator i_table_fac = m_task_factory_map.find("table_factory");
  BFM::iterator i_source_fac = m_task_factory_map.find("source_term_factory");
  BFM::iterator i_turb_model_fac = m_task_factory_map.find("turbulence_model_factory");

  TaskFactoryBase::TaskMap all_bc_tasks = i_bc_fac->second->retrieve_all_tasks();
  TaskFactoryBase::TaskMap all_table_tasks = i_table_fac->second->retrieve_all_tasks();

  TaskController& tsk_controller = TaskController::self();
  const TaskController::Packing& packed_info = tsk_controller.get_packing_info();

  // ----------------- SSP RK LOOP -----------------------------------------------------------------

  for ( int time_substep = 0; time_substep < m_rk_order; time_substep++ ){


    // pre-update properties/source tasks)
    i_prop_fac->second->schedule_task_group( "pre_update_property_models",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // compute momentum closure
    i_turb_model_fac->second->schedule_task_group("momentum_closure",
      TaskInterface::TIMESTEP_EVAL, packed_info.turbulence, level, sched, matls, time_substep );

   // (pre-update source terms)
    i_source_fac->second->schedule_task_group( "pre_update_source_tasks",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls , time_substep );

    // ** SCALARS **
    // PRE-PROJECTION
    // first compute the psi functions for the limiters:
    i_transport->second->schedule_task_group("scalar_psi_builders",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // now construct the RHS:
    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // now update them:
    i_transport->second->schedule_task_group("scalar_fe_update",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    // ** TABLE LOOKUP **
    i_table_fac->second->schedule_task_group("all_tasks",
      TaskInterface::TIMESTEP_EVAL, packed_info.global, level, sched, matls, time_substep );

    i_table_fac->second->schedule_task_group("all_tasks",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );

    // ** MOMENTUM **
    i_transport->second->schedule_task_group( "momentum_construction", TaskInterface::TIMESTEP_EVAL,
      packed_info.global, level, sched, matls, time_substep );

    i_transport->second->schedule_task_group( "momentum_fe_update", TaskInterface::TIMESTEP_EVAL,
      packed_info.global, level, sched, matls, time_substep );

    // Scalar BCs
    i_transport->second->schedule_task_group("scalar_rhs_builders",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );
    // bc factory tasks
    i_bc_fac->second->schedule_task_group("all_tasks",
      TaskInterface::BC, packed_info.global, level, sched, matls, time_substep );
    // ** PRESSURE PROJECTION **
    if ( i_transport->second->has_task("build_pressure_system")){

      //APPLY BC for OULET AND PRESSURE PER STAS'S BCs
      AtomicTaskInterface* rhohat_tsk = i_transport->second->retrieve_atomic_task("vel_rho_hat_bc");
      rhohat_tsk->schedule_task(level, sched, matls, AtomicTaskInterface::ATOMIC_STANDARD_TASK, time_substep);

      PressureEqn* press_tsk = dynamic_cast<PressureEqn*>(i_transport->second->retrieve_task("build_pressure_system"));
      // Compute the coeffificients
      press_tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep );
      // Compute the boundary conditions on the linear system
      press_tsk->schedule_task(level, sched, matls, TaskInterface::BC_TASK, time_substep );
      // Solve it - calling out to hypre external lib
      press_tsk->solve(level, sched, time_substep);
      // Apply boundary conditions on the pressure field. The BCs are initially applied on the
      // linear system, however, the resulting pressure field also needs BCs so that the correction
      // to the velocities is done correctly.
      AtomicTaskInterface* press_bc_tsk = i_transport->second->retrieve_atomic_task("pressure_bcs");
      press_bc_tsk->schedule_task(level, sched, matls, AtomicTaskInterface::ATOMIC_STANDARD_TASK, time_substep);
      // Correct velocities
      AtomicTaskInterface* gradP_tsk = i_transport->second->retrieve_atomic_task("pressure_correction");
      gradP_tsk->schedule_task(level, sched, matls, AtomicTaskInterface::ATOMIC_STANDARD_TASK, time_substep);
 
    }
    // apply boundary conditions
     i_transport->second->schedule_task_group( "momentum_construction", TaskInterface::BC, false, level, sched, matls, time_substep );

    //Compute U from rhoU
    i_prop_fac->second->schedule_task( "u_from_rho_u", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );
      
    i_prop_fac->second->schedule_task( "compute_cc_velocities", TaskInterface::TIMESTEP_EVAL,
      level, sched, matls, time_substep, false, true );




  } // RK Integrator

}

//--------------------------------------------------------------------------------------------------
void
KokkosSolver::SandBox( const LevelP& level, SchedulerP& sched ){

  //const bool pack_tasks = true;
  //const bool dont_pack_tasks = false;

  const int time_substep = 0;

  const MaterialSet* matls = m_sharedState->allArchesMaterials();

  BFM::iterator i_util_fac = m_task_factory_map.find("utility_factory");
  BFM::iterator i_transport = m_task_factory_map.find("transport_factory");
  BFM::iterator i_prop_fac = m_task_factory_map.find("property_models_factory");
  BFM::iterator i_bc_fac = m_task_factory_map.find("boundary_condition_factory");
  BFM::iterator i_table_fac = m_task_factory_map.find("table_factory");
  BFM::iterator i_source_fac = m_task_factory_map.find("source_term_factory");
  TaskFactoryBase::TaskMap all_bc_tasks = i_bc_fac->second->retrieve_all_tasks();
  TaskFactoryBase::TaskMap all_table_tasks = i_table_fac->second->retrieve_all_tasks();

  // ----------------- Time Integration ------------------------------------------------------------
  // (pre-update properties tasks)
  SVec prop_preupdate_tasks = i_prop_fac->second->retrieve_task_subset("pre_update_property_models");
  for (auto i = prop_preupdate_tasks.begin(); i != prop_preupdate_tasks.end(); i++){
    TaskInterface* tsk = i_prop_fac->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
  }

   // (pre-update source terms)
  SVec pre_update_source = i_source_fac ->second->retrieve_task_subset("pre_update_source_task");
  for (auto i = pre_update_source.begin(); i != pre_update_source.end(); i++){
    TaskInterface* tsk = i_source_fac->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
  }

  // first compute the psi functions for the limiters:
  SVec scalar_psi_builders = i_transport->second->retrieve_task_subset("scalar_psi_builders");
  for ( SVec::iterator i = scalar_psi_builders.begin(); i != scalar_psi_builders.end(); i++){
    TaskInterface* tsk = i_transport->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
  }

  // now construct the RHS:
  SVec scalar_rhs_builders = i_transport->second->retrieve_task_subset("scalar_rhs_builders");
  for ( SVec::iterator i = scalar_rhs_builders.begin(); i != scalar_rhs_builders.end(); i++){
    TaskInterface* tsk = i_transport->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
    tsk->schedule_task(level, sched, matls, TaskInterface::BC_TASK, time_substep);
  }

  // now update them:
  SVec scalar_fe_up = i_transport->second->retrieve_task_subset("scalar_fe_update");
  for ( SVec::iterator i = scalar_fe_up.begin(); i != scalar_fe_up.end(); i++){
    TaskInterface* tsk = i_transport->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
  }

}
