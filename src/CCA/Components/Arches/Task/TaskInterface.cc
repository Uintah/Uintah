#include <CCA/Components/Arches/Task/TaskInterface.h>

//Uintah Includes:

using namespace Uintah;

typedef ArchesFieldContainer::WHICH_DW WHICH_DW;
typedef ArchesFieldContainer::VAR_DEPEND VAR_DEPEND;
typedef ArchesFieldContainer::VariableRegistry VariableRegistry;

TaskInterface::TaskInterface( std::string task_name, int matl_index ) :
  _task_name(task_name),
  _matl_index(matl_index)
{
}

TaskInterface::~TaskInterface()
{
  //destroy local labels
  for ( auto ilab = _local_labels.begin(); ilab != _local_labels.end(); ilab++ ){
    VarLabel::destroy(*ilab);
  }
}

//====================================================================================
//
//====================================================================================
void TaskInterface::schedule_task( const LevelP& level,
                                   SchedulerP& sched,
                                   const MaterialSet* matls,
                                   TASK_TYPE task_type,
                                   int time_substep ){

  VariableRegistry variable_registry;

  Task* tsk;

  const bool packed_tasks = false;

  if ( task_type == STANDARD_TASK ){
    register_timestep_eval( variable_registry, time_substep , packed_tasks);
    tsk = scinew Task( _task_name, this, &TaskInterface::do_task, variable_registry, time_substep );
  } else if ( task_type == BC_TASK ) {
    register_compute_bcs( variable_registry, time_substep , packed_tasks);
    tsk = scinew Task( _task_name+"_bc_task", this, &TaskInterface::do_bcs, variable_registry, time_substep );
  } else
    throw InvalidValue("Error: Task type not recognized.",__FILE__,__LINE__);

  int counter = 0;
  for ( auto pivar = variable_registry.begin(); pivar != variable_registry.end(); pivar++ ){

    counter++;

    ArchesFieldContainer::VariableInformation& ivar = *pivar;

    switch(ivar.depend) {
    case ArchesFieldContainer::COMPUTES:
      if ( time_substep == 0 ) {
        tsk->computes( ivar.label );   //only compute on the zero time substep
      } else {
        tsk->modifies( ivar.label );
      }
      break;
    case ArchesFieldContainer::MODIFIES:
      tsk->modifies( ivar.label );
      break;
    case ArchesFieldContainer::REQUIRES:
      tsk->requires( ivar.uintah_task_dw, ivar.label, ivar.ghost_type, ivar.nGhost );
      break;
    default:
      throw InvalidValue("Arches Task Error: Cannot schedule task becuase of incomplete variable dependency: "+_task_name, __FILE__, __LINE__);
      break;

    }
  }

  //other variables:
  tsk->requires(Task::OldDW, VarLabel::find("delT"));

  if ( counter > 0 ) {
    sched->addTask( tsk, level->eachPatch(), matls );
  }
  else {
    delete tsk;
  }

}

//====================================================================================
//
//====================================================================================
void TaskInterface::schedule_init( const LevelP& level,
                                   SchedulerP& sched,
                                   const MaterialSet* matls,
                                   const bool is_restart,
                                   const bool reinitialize ){

  VariableRegistry variable_registry;

  const bool packed_tasks = false;

  if ( is_restart ) {
    register_restart_initialize( variable_registry , packed_tasks);
  } else {
    register_initialize( variable_registry, packed_tasks );
  }

  Task* tsk;
  if ( is_restart ) {
    tsk = scinew Task( _task_name+"_restart_initialize", this, &TaskInterface::do_restart_init, variable_registry );
  } else {
    tsk = scinew Task( _task_name+"_initialize", this, &TaskInterface::do_init, variable_registry );
  }

  int counter = 0;

  for ( auto pivar = variable_registry.begin(); pivar != variable_registry.end(); pivar++ ){

    counter++;

    ArchesFieldContainer::VariableInformation& ivar = *pivar;

    switch(ivar.depend) {

    case ArchesFieldContainer::COMPUTES:
    {
      if ( reinitialize ){
        tsk->modifies( ivar.label );
      } else {
        tsk->computes( ivar.label );
      }
      break;
    }
    case ArchesFieldContainer::MODIFIES:
      tsk->modifies( ivar.label );
      break;
    case ArchesFieldContainer::REQUIRES:
      tsk->requires( ivar.uintah_task_dw, ivar.label, ivar.ghost_type, ivar.nGhost );
      break;
    default:
      throw InvalidValue("Arches Task Error: Cannot schedule task because of incomplete variable dependency: "+_task_name, __FILE__, __LINE__);
      break;

    }
  }

  if ( counter > 0 )
    sched->addTask( tsk, level->eachPatch(), matls );
  else
    delete tsk;

}

//====================================================================================
//
//====================================================================================
void TaskInterface::schedule_timestep_init( const LevelP& level,
                                            SchedulerP& sched,
                                            const MaterialSet* matls ){

  VariableRegistry variable_registry;

  const bool packed_tasks = false;

  register_timestep_init( variable_registry, packed_tasks );

  Task* tsk = scinew Task( _task_name+"_timestep_initialize", this, &TaskInterface::do_timestep_init, variable_registry );

  int counter = 0;

  for ( auto pivar = variable_registry.begin(); pivar != variable_registry.end(); pivar++ ){

    counter++;

    ArchesFieldContainer::VariableInformation& ivar = *pivar;

    switch(ivar.depend) {

    case ArchesFieldContainer::COMPUTES:
      tsk->computes( ivar.label );   //only compute on the zero time substep
      break;
    case ArchesFieldContainer::MODIFIES:
      tsk->modifies( ivar.label );
      break;
    case ArchesFieldContainer::REQUIRES:
      tsk->requires( ivar.uintah_task_dw, ivar.label, ivar.ghost_type, ivar.nGhost );
      break;
    default:
      throw InvalidValue("Arches Task Error: Cannot schedule task becuase of incomplete variable dependency: "+_task_name, __FILE__, __LINE__);
      break;

    }
  }

  //other variables:
  tsk->requires(Task::OldDW, VarLabel::find("delT"));

  if ( counter > 0 )
    sched->addTask( tsk, level->eachPatch(), matls );
  else
    delete tsk;

}

//====================================================================================
// (do tasks)
//====================================================================================
void TaskInterface::do_task( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             VariableRegistry variable_registry,
                             int time_substep ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, _matl_index, variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = time_substep;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    eval( patch, tsk_info_mngr );

    //clean up
    delete tsk_info_mngr;
    delete field_container;
  }
}

void TaskInterface::do_bcs( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            VariableRegistry variable_registry,
                            int time_substep ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, _matl_index, variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    //These lines don't work because we are applying the BC in scheduleInitialize.
    // During that phase, there is no valid DT. Need to work on this?
    /// @TODO: Work on getting DT to the BC task.
    //get the current dt
    // delt_vartype DT;
    // old_dw->get(DT, VarLabel::find("delT"));
    // info.dt = DT;
    info.time_substep = time_substep;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    compute_bcs( patch, tsk_info_mngr );

    //clean up
    delete tsk_info_mngr;
    delete field_container;
  }
}

void TaskInterface::do_init( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             VariableRegistry variable_registry ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, _matl_index, variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    initialize( patch, tsk_info_mngr );

    //clean up
    delete tsk_info_mngr;
    delete field_container;
  }
}

void TaskInterface::do_restart_init( const ProcessorGroup* pc,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     VariableRegistry variable_registry ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, _matl_index, variable_registry, old_dw, new_dw );

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    restart_initialize( patch, tsk_info_mngr );

    //clean up
    delete tsk_info_mngr;
    delete field_container;
  }
}

void TaskInterface::do_timestep_init( const ProcessorGroup* pc,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      VariableRegistry variable_registry ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, _matl_index, variable_registry, old_dw, new_dw );

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    timestep_init( patch, tsk_info_mngr );

    //clean up
    delete tsk_info_mngr;
    delete field_container;

  }
}
