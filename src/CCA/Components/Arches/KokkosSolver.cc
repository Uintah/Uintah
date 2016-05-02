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

using namespace Uintah;
typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
typedef std::vector<std::string> SVec;

KokkosSolver::KokkosSolver( SimulationStateP& shared_state,
  const ProcessorGroup* myworld,
  std::map<std::string,
  boost::shared_ptr<TaskFactoryBase> >& task_factory_map)
  : m_sharedState(shared_state), _task_factory_map(task_factory_map), NonlinearSolver( myworld )
{}

KokkosSolver::~KokkosSolver(){
  delete m_bcHelper;
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
  db->getWithDefault("temporal_order", _rk_order,1);
}

void
KokkosSolver::computeTimestep(const LevelP& level, SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "KokkosSolver::computeStableTimeStep",this,
                           &KokkosSolver::computeStableTimeStep);

  tsk->computes( m_sharedState->get_delt_label(), level.get_rep() );
  sched->addTask( tsk, level->eachPatch(), m_sharedState->allArchesMaterials() );
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

    new_dw->put(delt_vartype(0.1), m_sharedState->get_delt_label(), level);

  }

}

void
KokkosSolver::sched_setInitialGuess( SchedulerP&,
                                     const PatchSet* patches,
                                     const MaterialSet* matls)
{}

void
KokkosSolver::initialize( const LevelP& level, SchedulerP& sched, const bool doing_restart )
{
  const MaterialSet* matls = m_sharedState->allArchesMaterials();
  bool is_restart = false;

  //boundary condition
  m_bcHelper = scinew WBCHelper( level, sched, matls );
  m_bcHelper->parse_boundary_conditions();
  sched_checkBCs( sched, level );

  //utility factory
  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
  TaskInterface* tsk = i_util_fac->second->retrieve_task("grid_info");
  tsk->schedule_init( level, sched, matls, is_restart );

  //transport factory
  BFM::iterator i_trans_fac = _task_factory_map.find("transport_factory");
  TaskFactoryBase::TaskMap all_trans_tasks = i_trans_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_trans_tasks.begin(); i != all_trans_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }

  //property factory
  BFM::iterator i_prop_fac = _task_factory_map.find("property_models_factory");
  TaskFactoryBase::TaskMap all_prop_tasks = i_prop_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_prop_tasks.begin(); i != all_prop_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }

  BFM::iterator i_init_fac = _task_factory_map.find("initialize_factory");
  TaskFactoryBase::TaskMap all_init_tasks = i_init_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_init_tasks.begin(); i != all_init_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }


}

int
KokkosSolver::nonlinearSolve( const LevelP& level,
                                     SchedulerP& sched )
{

  const MaterialSet* matls = m_sharedState->allArchesMaterials();
  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
  TaskInterface* tsk = i_util_fac->second->retrieve_task("grid_info");
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

    //(vel)
    SVec mom_rhs_builders = i_transport->second->retrieve_task_subset("mom_rhs_builders");
    for ( SVec::iterator i = mom_rhs_builders.begin(); i != mom_rhs_builders.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
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
    }

    //now update them
    SVec scalar_fe_up = i_transport->second->retrieve_task_subset("scalar_fe_update");
    for ( SVec::iterator i = scalar_fe_up.begin(); i != scalar_fe_up.end(); i++){
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, time_substep);
    }
  }

  return 0;

}

void
KokkosSolver::sched_checkBCs( SchedulerP& sched, const LevelP& level ){
  Task* tsk = scinew Task("KokkosSolver::checkBCs", this,
                          &KokkosSolver::checkBCs);
  sched->addTask(tsk, level->eachPatch(), m_sharedState->allArchesMaterials());
}
void
KokkosSolver::checkBCs(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = m_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    using std::cout;

    const BndMapT& bc_info = m_bcHelper->get_boundary_information();

    m_bcHelper->print();

//    BndMapT::const_iterator iter = bc_info.begin();
//    for (; iter != bc_info.end(); iter++){
//      cout << iter->first << std::endl;
//      Uintah::Iterator& ui = m_bcHelper->get_uintah_extra_bnd_mask( iter->second, patch->getID() );
//      cout << "iterator = " << ui << std::endl;
//      cout << " in loop" << std::endl;
//    }

  }
}
