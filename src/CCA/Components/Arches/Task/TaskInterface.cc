#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>

//Uintah Includes:

//3P Includes:
//#include <boost/foreach.hpp>

using namespace Uintah;
namespace so = SpatialOps;

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
  BOOST_FOREACH( const VarLabel* &ilab, _local_labels ){
    VarLabel::destroy(ilab);
  }
}

//====================================================================================
//
//====================================================================================
void
TaskInterface::register_variable( std::string name,
                                  VAR_DEPEND dep,
                                  int nGhost,
                                  WHICH_DW dw,
                                  VariableRegistry& variable_registry,
                                  const int time_substep ){

  register_variable_work( name, dep, nGhost, dw, variable_registry, time_substep );

}

void
TaskInterface::register_variable( std::string name,
                                  VAR_DEPEND dep,
                                  int nGhost,
                                  WHICH_DW dw,
                                  VariableRegistry& variable_registry ){

  register_variable_work( name, dep, nGhost, dw, variable_registry, 0 );

}

void
TaskInterface::register_variable( std::string name,
                                  VAR_DEPEND dep,
                                  VariableRegistry& variable_registry ){

  WHICH_DW dw = ArchesFieldContainer::NEWDW;
  int nGhost = 0;

  register_variable_work( name, dep, nGhost, dw, variable_registry, 0 );

}

void
TaskInterface::register_variable( std::string name,
                                  VAR_DEPEND dep,
                                  VariableRegistry& variable_registry,
                                  const int timesubstep ){

  WHICH_DW dw = ArchesFieldContainer::NEWDW;
  int nGhost = 0;

  register_variable_work( name, dep, nGhost, dw, variable_registry, timesubstep );

}

//====================================================================================
//
//====================================================================================
void
TaskInterface::register_variable_work( std::string name,
                                       VAR_DEPEND dep,
                                       int nGhost,
                                       WHICH_DW dw,
                                       VariableRegistry& variable_registry,
                                       const int time_substep ){

  ArchesFieldContainer::VariableInformation info;

  info.name   = name;
  info.depend = dep;
  info.dw     = dw;
  info.nGhost = nGhost;
  info.local = false;

  info.is_constant = false;
  if ( dep == ArchesFieldContainer::REQUIRES ){
    info.is_constant = true;
  }

  switch (dw) {

  case ArchesFieldContainer::OLDDW:

    info.uintah_task_dw = Task::OldDW;
    break;

  case ArchesFieldContainer::NEWDW:

    info.uintah_task_dw = Task::NewDW;
    break;

  case ArchesFieldContainer::LATEST:

    if ( time_substep == 0 ){
      info.dw = ArchesFieldContainer::OLDDW;
      info.uintah_task_dw = Task::OldDW;
    } else {
      info.dw = ArchesFieldContainer::NEWDW;
      info.uintah_task_dw = Task::NewDW;
    }
    break;

  default:

    throw InvalidValue("Arches Task Error: Cannot determine the DW needed for variable: "+name, __FILE__, __LINE__);
    break;

  }

  //check for conflicts:
  if (dep == ArchesFieldContainer::COMPUTES && dw == ArchesFieldContainer::OLDDW) {
    throw InvalidValue("Arches Task Error: Cannot COMPUTE (ArchesFieldContainer::COMPUTES) a variable from OldDW for variable: "+name, __FILE__, __LINE__);
  }

  if ( (dep == ArchesFieldContainer::MODIFIES && dw == ArchesFieldContainer::OLDDW) ) {
    throw InvalidValue("Arches Task Error: Cannot MODIFY a variable from OldDW for variable: "+name, __FILE__, __LINE__);
  }

  if ( dep == ArchesFieldContainer::COMPUTES || dep == ArchesFieldContainer::MODIFIES ){

    if ( nGhost > 0 ) {

      std::cout << "Arches Task Warning: A variable COMPUTE/MODIFIES found that is requesting ghosts for: "+name+" Nghosts set to zero!" << std::endl;
      info.nGhost = 0;

    }
  }

  const VarLabel* the_label = nullptr;
  the_label = VarLabel::find( name );

  if ( the_label == nullptr ){
    throw InvalidValue("Error: The variable named: "+name+" does not exist for task:"+_task_name,__FILE__,__LINE__);
  } else {
    info.label = the_label;
  }

  const Uintah::TypeDescription* type_desc = the_label->typeDescription();

  if ( dep == ArchesFieldContainer::REQUIRES ) {

    if ( nGhost == 0 ){
      info.ghost_type = Ghost::None;
    } else {
      if ( type_desc == CCVariable<int>::getTypeDescription() ) {
          info.ghost_type = Ghost::AroundCells;
      } else if ( type_desc == CCVariable<double>::getTypeDescription() ) {
          info.ghost_type = Ghost::AroundCells;
      } else if ( type_desc == CCVariable<Vector>::getTypeDescription() ) {
          info.ghost_type = Ghost::AroundCells;
      } else if ( type_desc == SFCXVariable<double>::getTypeDescription() ) {
          info.ghost_type = Ghost::AroundFaces;
      } else if ( type_desc == SFCYVariable<double>::getTypeDescription() ) {
          info.ghost_type = Ghost::AroundFaces;
      } else if ( type_desc == SFCZVariable<double>::getTypeDescription() ) {
          info.ghost_type = Ghost::AroundFaces;
      } else {
        throw InvalidValue("Error: No coverage yet for this type of variable.", __FILE__,__LINE__);
      }
    }
  }

  variable_registry.push_back( info );

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

  if ( task_type == STANDARD_TASK ){
    register_timestep_eval( variable_registry, time_substep );
    tsk = scinew Task( _task_name, this, &TaskInterface::do_task, variable_registry, time_substep );
  } else if ( task_type == BC_TASK ) {
    register_compute_bcs( variable_registry, time_substep );
    tsk = scinew Task( _task_name+"_bc_task", this, &TaskInterface::do_bcs, variable_registry, time_substep );
  } else
    throw InvalidValue("Error: Task type not recognized.",__FILE__,__LINE__);

  int counter = 0;
  BOOST_FOREACH( ArchesFieldContainer::VariableInformation &ivar, variable_registry ){

    counter++;

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

  if ( counter > 0 )
    sched->addTask( tsk, level->eachPatch(), matls );
  else
    delete tsk;

}

//====================================================================================
//
//====================================================================================
void TaskInterface::schedule_init( const LevelP& level,
                                   SchedulerP& sched,
                                   const MaterialSet* matls,
                                   const bool is_restart ){

  VariableRegistry variable_registry;

  if ( is_restart ) {
    register_restart_initialize( variable_registry );
  } else {
    register_initialize( variable_registry );
  }

  Task* tsk;
  if ( is_restart ) {
    tsk = scinew Task( _task_name+"_restart_initialize", this, &TaskInterface::do_restart_init, variable_registry );
  } else {
    tsk = scinew Task( _task_name+"_initialize", this, &TaskInterface::do_init, variable_registry );
  }

  int counter = 0;

  BOOST_FOREACH( ArchesFieldContainer::VariableInformation &ivar, variable_registry ){

    counter++;

    switch(ivar.depend) {

    case ArchesFieldContainer::COMPUTES:
      tsk->computes( ivar.label );
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

  register_timestep_init( variable_registry );

  Task* tsk = scinew Task( _task_name+"_timestep_initialize", this, &TaskInterface::do_timestep_init, variable_registry );

  int counter = 0;

  BOOST_FOREACH( ArchesFieldContainer::VariableInformation &ivar, variable_registry ){

    counter++;

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

    const WasatchCore::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch, _matl_index, variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = time_substep;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    //get the operator DB for this patch
    Operators& opr = Operators::self();
    Operators::PatchInfoMap::iterator i_opr = opr.patch_info_map.find(patch->getID());

    eval( patch, tsk_info_mngr, i_opr->second._sodb );

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

    const WasatchCore::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch, _matl_index, variable_registry, old_dw, new_dw);

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

    //get the operator DB for this patch
    Operators& opr = Operators::self();
    Operators::PatchInfoMap::iterator i_opr = opr.patch_info_map.find(patch->getID());

    compute_bcs( patch, tsk_info_mngr, i_opr->second._sodb );

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

    const WasatchCore::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch, _matl_index, variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    //get the operator DB for this patch
    Operators& opr = Operators::self();
    Operators::PatchInfoMap::iterator i_opr = opr.patch_info_map.find(patch->getID());

    initialize( patch, tsk_info_mngr, i_opr->second._sodb );

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

    const WasatchCore::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch, _matl_index, variable_registry, old_dw, new_dw );

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    //get the operator DB for this patch
    Operators& opr = Operators::self();
    Operators::PatchInfoMap::iterator i_opr = opr.patch_info_map.find(patch->getID());

    restart_initialize( patch, tsk_info_mngr, i_opr->second._sodb );

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

    const WasatchCore::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch, _matl_index, variable_registry, old_dw, new_dw );

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //this makes the "getting" of the grid variables easier from the user side (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    //get the operator DB for this patch
    Operators& opr = Operators::self();
    Operators::PatchInfoMap::iterator i_opr = opr.patch_info_map.find(patch->getID());

    timestep_init( patch, tsk_info_mngr, i_opr->second._sodb );

    //clean up
    delete tsk_info_mngr;
    delete field_container;

  }
}
