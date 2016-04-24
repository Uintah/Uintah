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

using namespace Uintah;
typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
typedef std::vector<std::string> SVec;

KokkosSolver::KokkosSolver( SimulationStateP& shared_state,
  const ProcessorGroup* myworld,
  std::map<std::string,
  boost::shared_ptr<TaskFactoryBase> >& task_factory_map)
  : _shared_state(shared_state), _task_factory_map(task_factory_map), NonlinearSolver( myworld )
{}

KokkosSolver::~KokkosSolver(){}


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
}

void
KokkosSolver::computeTimestep(const LevelP& level, SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "KokkosSolver::computeStableTimeStep",this,
                           &KokkosSolver::computeStableTimeStep);

  tsk->computes( _shared_state->get_delt_label(), level.get_rep() );
  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() );
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
    int indx = _shared_state->getArchesMaterial(archIndex)->getDWIndex();

    new_dw->put(delt_vartype(0.001), _shared_state->get_delt_label(), level);

  }

}

int
KokkosSolver::nonlinearSolve( const LevelP& level,
                                     SchedulerP& sched )
{

  const MaterialSet* matls = _shared_state->allArchesMaterials();
  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
  TaskInterface* tsk = i_util_fac->second->retrieve_task("grid_info");
  tsk->schedule_timestep_init( level, sched, matls );


  BFM::iterator i_transport = _task_factory_map.find("transport_factory");
  TaskFactoryBase::TaskMap all_tasks = i_transport->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++){
    i->second->schedule_timestep_init(level, sched, matls);
  }

  //(vel)
  SVec mom_rhs_builders = i_transport->second->retrieve_task_subset("mom_rhs_builders");
  for ( SVec::iterator i = mom_rhs_builders.begin(); i != mom_rhs_builders.end(); i++){
    TaskInterface* tsk = i_transport->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, 0);
  }

  //(scalars)
  SVec scalar_rhs_builders = i_transport->second->retrieve_task_subset("scalar_rhs_builders");
  for ( SVec::iterator i = scalar_rhs_builders.begin(); i != scalar_rhs_builders.end(); i++){
    TaskInterface* tsk = i_transport->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, 0);
  }

  //now update them
  SVec scalar_fe_up = i_transport->second->retrieve_task_subset("scalar_fe_update");
  for ( SVec::iterator i = scalar_fe_up.begin(); i != scalar_fe_up.end(); i++){
    TaskInterface* tsk = i_transport->second->retrieve_task(*i);
    tsk->schedule_task(level, sched, matls, TaskInterface::STANDARD_TASK, 0);
  }

  return 0;
}

void
KokkosSolver::sched_setInitialGuess( SchedulerP&,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{}

void
KokkosSolver::initialize( const LevelP& level, SchedulerP& sched, const bool doing_restart )
{
  const MaterialSet* matls = _shared_state->allArchesMaterials();
  bool is_restart = false;

  //utility factory
  BFM::iterator i_util_fac = _task_factory_map.find("utility_factory");
  TaskInterface* tsk = i_util_fac->second->retrieve_task("grid_info");
  tsk->schedule_init( level, sched, matls, is_restart );

  //transport factory
  BFM::iterator i_trans_fac = _task_factory_map.find("transport_factory");
  TaskFactoryBase::TaskMap all_tasks = i_trans_fac->second->retrieve_all_tasks();
  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, doing_restart);
  }

}
