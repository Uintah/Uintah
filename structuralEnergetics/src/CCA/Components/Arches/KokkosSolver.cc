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
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/UPSHelper.h>
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

using namespace Uintah;

typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
typedef std::vector<std::string> SVec;

KokkosSolver::KokkosSolver( SimulationStateP& shared_state,
                            const ProcessorGroup* myworld )
  : NonlinearSolver( myworld ), m_sharedState(shared_state)
{}

KokkosSolver::~KokkosSolver(){
  for (auto i = m_bcHelper.begin(); i != m_bcHelper.end(); i++){
    delete i->second;
  }
  m_bcHelper.clear();
}

void
KokkosSolver::sched_restartInitialize( const LevelP& level, SchedulerP& sched )
{}

void
KokkosSolver::sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched )
{}

void
KokkosSolver::problemSetup( const ProblemSpecP& input_db,
                            SimulationStateP& state,
                            GridP& grid )
{

  ProblemSpecP db = input_db;
  ProblemSpecP db_ks = db->findBlock("KokkosSolver");
  ProblemSpecP db_root = db->getRootNode();

  db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->getAttribute("order", _rk_order);
  proc0cout << " Time integrator: RK of order " << _rk_order << "\n \n";

  commonProblemSetup( db_ks );

  //------------------------------------------------------------------------------------------------
  //NEW TASK STUFF
  //build the factories
  std::shared_ptr<UtilityFactory> UtilF(scinew UtilityFactory());
  std::shared_ptr<TransportFactory> TransF(scinew TransportFactory());
  std::shared_ptr<InitializeFactory> InitF(scinew InitializeFactory());
  std::shared_ptr<ParticleModelFactory> PartModF(scinew ParticleModelFactory());
  std::shared_ptr<LagrangianParticleFactory> LagF(scinew LagrangianParticleFactory());
  std::shared_ptr<PropertyModelFactoryV2> PropModels(scinew PropertyModelFactoryV2());

  _task_factory_map.clear();
  _task_factory_map.insert(std::make_pair("utility_factory",UtilF));
  _task_factory_map.insert(std::make_pair("transport_factory",TransF));
  _task_factory_map.insert(std::make_pair("initialize_factory",InitF));
  _task_factory_map.insert(std::make_pair("particle_model_factory",PartModF));
  _task_factory_map.insert(std::make_pair("lagrangian_factory",LagF));
  _task_factory_map.insert(std::make_pair("property_models_factory", PropModels));

  typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
  proc0cout << "\n Registering Tasks For: " << std::endl;
  for ( BFM::iterator i = _task_factory_map.begin(); i != _task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->set_shared_state(m_sharedState);
    i->second->register_all_tasks(db);

  }

  proc0cout << "\n Building Tasks For: " << std::endl;

  for ( BFM::iterator i = _task_factory_map.begin(); i != _task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->build_all_tasks(db);

  }

  proc0cout << std::endl;

}

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

    Task* tsk = scinew Task( "KokkosSolver::computeStableTimeStep",this,
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
    std::cout << "\n ****************** WARNING ***************** " << std::endl;
    std::cout << "  The CFD variable mapping was not complete   " << std::endl;
    std::cout << "  because I could not find the appropriate    " << std::endl;
    std::cout << "  variable mapping from <VarID>. As a result  " << std::endl;
    std::cout << "  I am going to set dt to the delt_init as    " << std::endl;
    std::cout << "  specified in the input file.                " << std::endl;
    std::cout << " **************** END WARNING ***************\n " << std::endl;

    if ( !m_arches_spec->getRootNode()->findBlock("Time")->findBlock( "delt_init") ){
      throw ProblemSetupException("\n Error: Oops... please specify a delt_init in your input file.\n", __FILE__, __LINE__ );
    }

    Task* tsk = scinew Task( "KokkosSolver::setTimeStep", this,
                             &KokkosSolver::setTimeStep );

    tsk->computes( m_sharedState->get_delt_label(), level.get_rep() );

    sched->addTask( tsk, level->eachPatch(), m_sharedState->allArchesMaterials() );

  }

}

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

void
KokkosSolver::initialize( const LevelP& level, SchedulerP& sched, const bool doing_restart )
{
  const MaterialSet* matls = m_sharedState->allArchesMaterials();
  bool is_restart = false;

  //boundary condition helper
  m_bcHelper.insert(std::make_pair(level->getID(), scinew WBCHelper( level, sched, matls, m_arches_spec )));

  //computes the area for each inlet through the use of a reduction variables
  m_bcHelper[level->getID()]->sched_computeBCAreaHelper( sched, level, matls );

  //copies the reduction area variable information on area to a double in the BndCond spec
  m_bcHelper[level->getID()]->sched_bindBCAreaHelper( sched, level, matls );

  //utility factory
  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
  // build the x,y,z location scalars
  TaskInterface* tsk = i_util_fac->second->retrieve_task("grid_info");
  tsk->schedule_init( level, sched, matls, is_restart );
  // set the volume fractions
  tsk = i_util_fac->second->retrieve_task("vol_fraction_calc");
  tsk->schedule_init( level, sched, matls, is_restart );

  //property factory
  BFM::iterator i_prop_fac = _task_factory_map.find("property_models_factory");
  TaskFactoryBase::TaskMap all_prop_tasks = i_prop_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_prop_tasks.begin(); i != all_prop_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }

  //transport factory
  BFM::iterator i_trans_fac = _task_factory_map.find("transport_factory");
  i_trans_fac->second->set_bcHelper( m_bcHelper[level->getID()]);
  TaskFactoryBase::TaskMap all_trans_tasks = i_trans_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_trans_tasks.begin(); i != all_trans_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }

  // generic field initializer
  BFM::iterator i_init_fac = _task_factory_map.find("initialize_factory");
  TaskFactoryBase::TaskMap all_init_tasks = i_init_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_init_tasks.begin(); i != all_init_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }

  //Need to apply BC's after everything is initialized
  for ( TaskFactoryBase::TaskMap::iterator i = all_trans_tasks.begin(); i != all_trans_tasks.end(); i++) {
    i->second->schedule_task(level, sched, matls, TaskInterface::BC_TASK, 0);
  }

}

int
KokkosSolver::nonlinearSolve( const LevelP& level,
                                     SchedulerP& sched )
{

  const MaterialSet* matls = m_sharedState->allArchesMaterials();
  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");

  // carry forward the grid x,y,z
  TaskInterface* tsk = i_util_fac->second->retrieve_task("grid_info");
  tsk->schedule_timestep_init( level, sched, matls );

  // carry forward volume fraction
  tsk = i_util_fac->second->retrieve_task("vol_fraction_calc");
  tsk->schedule_timestep_init( level, sched, matls );

  BFM::iterator i_transport = _task_factory_map.find("transport_factory");
  TaskFactoryBase::TaskMap all_trans_tasks = i_transport->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_trans_tasks.begin(); i != all_trans_tasks.end(); i++){
    i->second->schedule_timestep_init(level, sched, matls);
  }

  BFM::iterator i_prop_fac = _task_factory_map.find("property_models_factory");
  TaskFactoryBase::TaskMap all_prop_tasks = i_prop_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_prop_tasks.begin(); i != all_prop_tasks.end(); i++) {
    i->second->schedule_timestep_init(level, sched, matls);
  }

  //RK loop
  for ( int time_substep = 0; time_substep < _rk_order; time_substep++ ){

    //(pre-update properties tasks)
    SVec prop_preupdate_tasks = i_prop_fac->second->retrieve_task_subset("pre_update_property_models");
    for (auto i = prop_preupdate_tasks.begin(); i != prop_preupdate_tasks.end(); i++){
      TaskInterface* tsk = i_prop_fac->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
    }

    //(scalars)
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

    //(momentum)
    // first compute the psi functions for the limiters:
    SVec momentum_psi_builders = i_transport->second->retrieve_task_subset("momentum_psi_builders");
    for ( SVec::iterator i = momentum_psi_builders.begin(); i != momentum_psi_builders.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
    }

    // now construct RHS:
    SVec mom_rhs_builders = i_transport->second->retrieve_task_subset("mom_rhs_builders");
    for ( SVec::iterator i = mom_rhs_builders.begin(); i != mom_rhs_builders.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
    }

    // now update them:
    SVec mom_fe_up = i_transport->second->retrieve_task_subset("mom_fe_update");
    for ( SVec::iterator i = mom_fe_up.begin(); i != mom_fe_up.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
    }

    // now apply boundary conditions for all scalar for the next timestep
    for ( SVec::iterator i = scalar_rhs_builders.begin(); i != scalar_rhs_builders.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::BC_TASK, time_substep);
    }
    for ( SVec::iterator i = mom_rhs_builders.begin(); i != mom_rhs_builders.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::BC_TASK, time_substep);
    }

  }

  return 0;

}
