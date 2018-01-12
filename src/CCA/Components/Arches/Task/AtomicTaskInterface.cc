#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>

using namespace Uintah;

typedef ArchesFieldContainer::WHICH_DW WHICH_DW;
typedef ArchesFieldContainer::VAR_DEPEND VAR_DEPEND;
typedef ArchesFieldContainer::VariableRegistry VariableRegistry;

AtomicTaskInterface::AtomicTaskInterface( std::string task_name, int matl_index ) :
  m_task_name(task_name),
  m_matl_index(matl_index)
{
}

AtomicTaskInterface::~AtomicTaskInterface()
{
  //destroy local labels
  for ( auto ilab = m_local_labels.begin(); ilab != m_local_labels.end(); ilab++ ){
    VarLabel::destroy( *ilab );
  }
}

//--------------------------------------------------------------------------------------------------

void AtomicTaskInterface::schedule_task( const LevelP& level,
                                   SchedulerP& sched,
                                   const MaterialSet* matls,
                                   ATOMIC_TASK_TYPE task_type,
                                   int time_substep ){

  VariableRegistry variable_registry;

  Task* tsk;

  if ( task_type == ATOMIC_STANDARD_TASK ){
    register_eval( variable_registry, time_substep );
    tsk = scinew Task( m_task_name, this, &AtomicTaskInterface::do_task, variable_registry,
                       time_substep );
  } else {
    throw InvalidValue("Error: Task type not recognized.",__FILE__,__LINE__);
  }

  int counter = 0;
  for ( auto pivar = variable_registry.begin(); pivar != variable_registry.end(); pivar++ ){

    ArchesFieldContainer::VariableInformation& ivar = *pivar;
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
      throw InvalidValue("Arches Task Error: Cannot schedule task becuase of incomplete variable dependency: "+m_task_name, __FILE__, __LINE__);
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

//--------------------------------------------------------------------------------------------------

void AtomicTaskInterface::do_task( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             VariableRegistry variable_registry,
                             int time_substep ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, m_matl_index,
      variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = time_substep;

    Uintah::ArchesTaskInfoManager* tsk_info_mngr = scinew Uintah::ArchesTaskInfoManager(variable_registry,
      patch, info);

    //this makes the "getting" of the grid variables easier from the user side
    // (ie, only need a string name )
    tsk_info_mngr->set_field_container( field_container );

    eval( patch, tsk_info_mngr );

    //clean up
    delete tsk_info_mngr;
    delete field_container;
  }
}
