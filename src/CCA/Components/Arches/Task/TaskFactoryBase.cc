#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>
#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <Core/Parallel/Portability.h>

using namespace Uintah;

namespace {

  Uintah::Dout dbg_arches_task{"Arches_Task_DBG", "Arches::TaskFactoryBase",
    "Scheduling and execution information of Arches tasks.", false };

  Uintah::Dout dbg_fac_vartask_dep{"Arches_Fac_Var_Task_Dep", "Arches::TaskFactoryBase",
    "Prints variable ghost req. for tasks per factory.", false };


  std::string get_task_exec_str( TaskInterface::TASK_TYPE type ){

    if ( type == TaskInterface::INITIALIZE ){
      return "INITIALIZE";
    } else if ( type == TaskInterface::TIMESTEP_INITIALIZE ){
      return "TIMESTEP INITIALIZE";
    } else if ( type == TaskInterface::TIMESTEP_EVAL ){
      return "TIMESTEP EVAL";
    } else if ( type == TaskInterface::BC ){
      return "BC";
    } else if ( type == TaskInterface::RESTART_INITIALIZE ){
      return "RESTART INITIALIZE";
    } else if ( type == TaskInterface::ATOMIC ){
      return "ATOMIC";
    } else {
      throw InvalidValue("Error: TASK_TYPE not recognized.", __FILE__, __LINE__);
    }

  }

}

//--------------------------------------------------------------------------------------------------
TaskFactoryBase::TaskFactoryBase( const ApplicationCommon* arches ) : m_arches(arches)
{
  m_matl_index = 0; //Arches material
  _tasks.clear();
}

//--------------------------------------------------------------------------------------------------
TaskFactoryBase::~TaskFactoryBase()
{
  for (BuildMap::iterator i = _builders.begin(); i != _builders.end(); i++ ){
    delete i->second;
  }

  for (TaskMap::iterator i = _tasks.begin(); i != _tasks.end(); i++ ){
    delete i->second;
  }

  for (ABuildMap::iterator i = _atomic_builders.begin(); i != _atomic_builders.end(); i++ ){
    delete i->second;
  }

  for (ATaskMap::iterator i = _atomic_tasks.begin(); i != _atomic_tasks.end(); i++ ){
     delete i->second;
   }
 }

//--------------------------------------------------------------------------------------------------
void
TaskFactoryBase::register_task( std::string task_name,
                                TaskInterface::TaskBuilder* builder,
                                ProblemSpecP db ){

  m_task_init_order.push_back(task_name);
  ASSERT(builder != nullptr);

  BuildMap::iterator i = _builders.find(task_name);
  if ( i == _builders.end() ){

    _builders.insert(std::make_pair(task_name, builder ));

    //If we are creating a builder, we assume the task is "active"
    _active_tasks.push_back(task_name);

  } else {

    throw InvalidValue( "Error: Attemping to load a duplicate builder: "+task_name,
                        __FILE__, __LINE__ );

  }

  // Now build the task:
  TaskInterface* tsk = retrieve_task(task_name);
  print_task_setup_info(tsk->get_task_name(), tsk->get_task_function());
  tsk->problemSetup( db );
  tsk->create_local_labels();

}

//--------------------------------------------------------------------------------------------------
void
TaskFactoryBase::register_atomic_task(std::string task_name,
                                      AtomicTaskInterface::AtomicTaskBuilder* builder,
                                      ProblemSpecP db ){

  ASSERT(builder != nullptr);

  ABuildMap::iterator i = _atomic_builders.find(task_name);
  if ( i == _atomic_builders.end() ){

    _atomic_builders.insert(std::make_pair(task_name, builder ));
    //If we are creating a builder, we assume the task is "active"
    _active_atomic_tasks.push_back(task_name);

  } else {

    throw InvalidValue( "Error: Attemping to load a duplicate atomic builder: "+task_name,
                        __FILE__, __LINE__ );

  }

  // Now build the task:
  TaskInterface* tsk = retrieve_atomic_task(task_name);
  print_task_setup_info(tsk->get_task_name(), tsk->get_task_function());
  tsk->problemSetup( db );
  tsk->create_local_labels();

}

//--------------------------------------------------------------------------------------------------
TaskInterface*
TaskFactoryBase::retrieve_task( const std::string task_name, const bool ignore_missing_task ){

  TaskMap::iterator itsk = _tasks.find(task_name);

  if ( itsk != _tasks.end() ){

    return itsk->second;

  } else {

    BuildMap::iterator ibuild = _builders.find(task_name);

    if ( ibuild != _builders.end() ){

      TaskInterface::TaskBuilder* b = ibuild->second;
      TaskInterface* t = b->build();

      _tasks[task_name]=t;

      TaskMap::iterator itsk_new = _tasks.find(task_name);

      //don't return t. must return the element in the map to
      //have the correct object intitiated
      return itsk_new->second;

    } else {

      if ( ignore_missing_task ){
        return NULL;
      } else {
        throw InvalidValue("Error: Cannot find task named: "+task_name+" for factory: "+_factory_name,__FILE__,__LINE__);
      }

    }
  }
}

//--------------------------------------------------------------------------------------------------
AtomicTaskInterface*
TaskFactoryBase::retrieve_atomic_task( const std::string task_name ){

  ATaskMap::iterator itsk = _atomic_tasks.find(task_name);

  if ( itsk != _atomic_tasks.end() ){

    return itsk->second;

  } else {

    ABuildMap::iterator ibuild = _atomic_builders.find(task_name);

    if ( ibuild != _atomic_builders.end() ){

      AtomicTaskInterface::AtomicTaskBuilder* b = ibuild->second;
      AtomicTaskInterface* t = b->build();

      _atomic_tasks.insert(std::make_pair(task_name,t));

      ATaskMap::iterator itsk_new = _atomic_tasks.find(task_name);

      //don't return t. must return the element in the map to
      //have the correct object intitiated
      return itsk_new->second;

    } else {

      throw InvalidValue("Error: Cannot find task named: "+task_name,__FILE__,__LINE__);

    }
  }
}

//--------------------------------------------------------------------------------------------------
void TaskFactoryBase::schedule_task( const std::string task_name,
                                     TaskInterface::TASK_TYPE type,
                                     const LevelP& level,
                                     SchedulerP& sched,
                                     const MaterialSet* matls,
                                     const int time_substep,
                                     const bool reinitialize,
                                     const bool ignore_missing_task ){

  // Only putting one task in here but still need to pass it as vector due to the
  // task_group scheduling feature
  std::vector<TaskInterface*> task_list_dummy(1);

  TaskInterface* tsk;
  if ( type != TaskInterface::ATOMIC ){
    tsk = retrieve_task( task_name, ignore_missing_task );
  } else {
    tsk = retrieve_atomic_task( task_name );
  }

  if ( tsk != NULL ){

    task_list_dummy[0] = tsk;

    factory_schedule_task( level, sched, matls, type, task_list_dummy,
                           tsk->get_task_name(), time_substep, reinitialize, false );

  } else {

    proc0cout << "\n Warning: Attempted to schedule task: " << task_name << " but could not find it." << std::endl;
    proc0cout << "          As a result, I will skip scheduling this task. This *may* or *may not* be by design." << std::endl;
    proc0cout << "          Please consult a developer if you are concerned.\n" << std::endl;

  }

}

//--------------------------------------------------------------------------------------------------
void TaskFactoryBase::schedule_task_group( const std::string task_group_name,
                                           std::vector<std::string> task_names,
                                           TaskInterface::TASK_TYPE type,
                                           const bool pack_tasks,
                                           const LevelP& level,
                                           SchedulerP& sched,
                                           const MaterialSet* matls,
                                           const int time_substep,
                                           const bool reinitialize ){

  if ( pack_tasks ){

    std::vector<TaskInterface*> task_list_dummy( task_names.size() );

    for (unsigned int i = 0; i < task_names.size(); i++ ){
      task_list_dummy[i] = retrieve_task( task_names[i] );
    }

    factory_schedule_task( level, sched, matls, type, task_list_dummy, task_group_name,
                           time_substep, reinitialize, pack_tasks );

  } else {

    std::vector<TaskInterface*> task_list_dummy(1);
    for (unsigned int i = 0; i < task_names.size(); i++ ){

      task_list_dummy[0] = retrieve_task( task_names[i] );
      factory_schedule_task( level, sched, matls, type, task_list_dummy, task_group_name,
                             time_substep, reinitialize, pack_tasks );

    }
  }
}

//--------------------------------------------------------------------------------------------------
void TaskFactoryBase::schedule_task_group( const std::string task_group_name,
                                           TaskInterface::TASK_TYPE type,
                                           const bool pack_tasks,
                                           const LevelP& level,
                                           SchedulerP& sched,
                                           const MaterialSet* matls,
                                           const int time_substep,
                                           const bool reinitialize ){

  std::vector<std::string> task_names = retrieve_task_subset(task_group_name);

  if ( pack_tasks ){

    std::vector<TaskInterface*> task_list_dummy( task_names.size() );

    for (unsigned int i = 0; i < task_names.size(); i++ ){
      task_list_dummy[i] = retrieve_task( task_names[i] );
    }

    factory_schedule_task( level, sched, matls, type, task_list_dummy, task_group_name,
                           time_substep, reinitialize, pack_tasks );

  } else {

    std::vector<TaskInterface*> task_list_dummy(1);
    for (unsigned int i = 0; i < task_names.size(); i++ ){

      task_list_dummy[0] = retrieve_task( task_names[i] );
      factory_schedule_task( level, sched, matls, type, task_list_dummy, task_names[i],
                             time_substep, reinitialize, pack_tasks );

    }
  }
}

//--------------------------------------------------------------------------------------------------
void TaskFactoryBase::factory_schedule_task( const LevelP& level,
                                             SchedulerP& sched,
                                             const MaterialSet* matls,
                                             TaskInterface::TASK_TYPE type,
                                             std::vector<TaskInterface*> arches_tasks,
                                             const std::string task_group_name,
                                             int time_substep,
                                             const bool reinitialize,
                                             const bool pack_tasks ){

  ArchesFieldContainer::VariableRegistry variable_registry;

  std::string pack_string = "OFF";
  if ( pack_tasks == true ){
    pack_string = "ON";
  }

  const std::string type_string = TaskInterface::get_task_type_string(type);
  DOUT( dbg_arches_task, "\n[TaskFactoryBase]  Scheduling with mode " << type_string << " for factory " << _factory_name );
  DOUT( dbg_arches_task, "                   Task packing is " << pack_string );

  bool archesTasksMixMemSpaces = false;
  TaskAssignedExecutionSpace assignedExecutionSpace{};

  std::vector<std::string> taskNames(arches_tasks.size());
  std::vector<int> taskExecSpace(arches_tasks.size());

  int icount=0;

  for ( auto i_task = arches_tasks.begin(); i_task != arches_tasks.end(); i_task++ ){

    taskNames[icount]=(*i_task)->get_task_name();

    TaskAssignedExecutionSpace temp{};

    DOUT( dbg_arches_task, "[TaskFactoryBase]      Task: " << (*i_task)->get_task_name() );

    switch( type ){

      case (TaskInterface::INITIALIZE):
        {
          temp = (*i_task)->loadTaskInitializeFunctionPointers();
          (*i_task)->register_initialize( variable_registry, pack_tasks );
          taskNames[icount]=taskNames[icount]+"::INITIALIZE";
          time_substep = 0;
        }
        break;
      case (TaskInterface::RESTART_INITIALIZE):
        {
          temp = (*i_task)->loadTaskRestartInitFunctionPointers();
          (*i_task)->register_restart_initialize( variable_registry , pack_tasks);
          taskNames[icount]=taskNames[icount]+"::RESTART_INITIALIZE";
          time_substep = 0;
        }
        break;
      case (TaskInterface::TIMESTEP_INITIALIZE):
        {
          temp = (*i_task)->loadTaskTimestepInitFunctionPointers();
          (*i_task)->register_timestep_init( variable_registry, pack_tasks );
          taskNames[icount]=taskNames[icount]+"::TIMESTEP_INITIALIZE";
          time_substep = 0;
        }
        break;
      case (TaskInterface::TIMESTEP_EVAL):
        {
          temp = (*i_task)->loadTaskEvalFunctionPointers();
          (*i_task)->register_timestep_eval( variable_registry, time_substep, pack_tasks);
          taskNames[icount]=taskNames[icount]+"::EVAL";
        }
        break;
      case (TaskInterface::BC):
        {
          temp = (*i_task)->loadTaskComputeBCsFunctionPointers();
          (*i_task)->register_compute_bcs( variable_registry, time_substep , pack_tasks);
          taskNames[icount]=taskNames[icount]+"::COMPUTE_BCS";
        }
        break;
      case (TaskInterface::ATOMIC):
        {
          temp = (*i_task)->loadTaskEvalFunctionPointers();
          (*i_task)->register_timestep_eval( variable_registry, time_substep, pack_tasks );
          taskNames[icount]=taskNames[icount]+"::ATOMIC";
        }
        break;
      default:
        throw InvalidValue("Error: TASK_TYPE not recognized.",__FILE__,__LINE__);
        break;

    }

    taskExecSpace[icount] = temp;
    icount++;

    if ( assignedExecutionSpace != TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE && assignedExecutionSpace != temp && temp != TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE ) {
      archesTasksMixMemSpaces = true;
    } else if ( TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE != temp ){
      assignedExecutionSpace = temp;
    }
  }

  auto TaskDependencies = [&](Task *& tsk) {

    int counter = 0;
    for ( auto pivar = variable_registry.begin(); pivar != variable_registry.end(); pivar++ ){

      counter++;

      ArchesFieldContainer::VariableInformation& ivar = *pivar;
      insert_max_ghost(ivar, _factory_name+"::"+task_group_name+", "+type_string);

      switch(ivar.depend) {
      case ArchesFieldContainer::COMPUTES:
        {
          if ( time_substep == 0 ) {
            if ( reinitialize ){
              // Uncomment this code to fix all ghost cells.
              // const Uintah::PatchSet* const allPatches =
              //  sched->getLoadBalancer()->getPerProcessorPatchSet(level);
              // const Uintah::PatchSubset* const localPatches =
              //  allPatches->getSubset( Uintah::Parallel::getMPIRank() );
              // DOUT( dbg_arches_task, "[TaskFactoryBase]  modifying (wsg): " << ivar.name );
              // tsk->modifiesWithScratchGhost( ivar.label,
              //                                localPatches,
              //                                Uintah::Task::ThisLevel,
              //                                matls->getSubset(0), Uintah::Task::NormalDomain,
              //                                ivar.ghost_type, ivar.nGhost );
              // end uncomment
              DOUT( dbg_arches_task, "[TaskFactoryBase]      modifying: " << ivar.name );
              tsk->modifies( ivar.label );   // was computed upstream
            } else {
              DOUT( dbg_arches_task, "[TaskFactoryBase]      computing: " << ivar.name );
              // tsk->computesWithScratchGhost( ivar.label, matls->getSubset(0),
              //                                Uintah::Task::NormalDomain, ivar.ghost_type,
              //                                ivar.nGhost );
              tsk->computes( ivar.label );   //only compute on the zero time substep
            }
          } else {
            // Uncomment this code to fix all ghost cells.
            // const Uintah::PatchSet* const allPatches =
            //   sched->getLoadBalancer()->getPerProcessorPatchSet(level);
            // const Uintah::PatchSubset* const localPatches =
            //    allPatches->getSubset( Uintah::Parallel::getMPIRank() );
            // DOUT( dbg_arches_task, "[TaskFactoryBase]  modifying (wsg): " << ivar.name );
            // tsk->modifiesWithScratchGhost( ivar.label,
            //                                localPatches,
            //                                Uintah::Task::ThisLevel,
            //                                matls->getSubset(0), Uintah::Task::NormalDomain,
            //                                ivar.ghost_type, ivar.nGhost );
            // end ucomment
            DOUT( dbg_arches_task, "[TaskFactoryBase]      modifying: " << ivar.name );
            tsk->modifies( ivar.label );
        }}
        break;
    case ArchesFieldContainer::COMPUTESCRATCHGHOST:
      {
        if ( time_substep == 0 ){
          DOUT( dbg_arches_task, "[TaskFactoryBase]  computing (wsg): " << ivar.name );
          tsk->computesWithScratchGhost( ivar.label, matls->getSubset(0),
                                         Uintah::Task::NormalDomain, ivar.ghost_type,
                                         ivar.nGhost );
        } else {
          const Uintah::PatchSet* const allPatches =
            sched->getLoadBalancer()->getPerProcessorPatchSet(level);
          const Uintah::PatchSubset* const localPatches =
            allPatches->getSubset( Uintah::Parallel::getMPIRank() );
          DOUT( dbg_arches_task, "[TaskFactoryBase]  modifying (wsg): " << ivar.name );
          tsk->modifiesWithScratchGhost( ivar.label,
                                         localPatches,
                                         Uintah::Task::ThisLevel,
                                         matls->getSubset(0), Uintah::Task::NormalDomain,
                                         ivar.ghost_type, ivar.nGhost );
        }
      }
      break;
    case ArchesFieldContainer::MODIFIES:
      {
        // Uncomment this code to fix all ghost cells.
        // const Uintah::PatchSet* const allPatches =
        //     sched->getLoadBalancer()->getPerProcessorPatchSet(level);
        // const Uintah::PatchSubset* const localPatches =
        //     allPatches->getSubset( Uintah::Parallel::getMPIRank() );
        // DOUT( dbg_arches_task, "[TaskFactoryBase]  modifying (wsg): " << ivar.name );
        // tsk->modifiesWithScratchGhost( ivar.label,
        //                                 localPatches,
        //                                 Uintah::Task::ThisLevel,
        //                                 matls->getSubset(0), Uintah::Task::NormalDomain,
        //                                 ivar.ghost_type, ivar.nGhost );
        // end uncomment
        DOUT( dbg_arches_task, "[TaskFactoryBase]      modifying: " << ivar.name );
        tsk->modifies( ivar.label );
      }
      break;
    case ArchesFieldContainer::REQUIRES:
      {
        DOUT( dbg_arches_task, "[TaskFactoryBase]      requiring: " << ivar.name << " with ghosts: " << ivar.nGhost << " from DW: " << ivar.dw );
        tsk->needsLabel( ivar.uintah_task_dw, ivar.label, ivar.ghost_type, ivar.nGhost );
      }
      break;
    default:
      {
        std::stringstream msg;
        msg << "Arches Task Error: Cannot schedule task because "
            << "of incomplete variable dependency. \n";
        throw InvalidValue(msg.str(), __FILE__, __LINE__);
      }
      break;
      }
    }

    //other variables:
    if ( sched->get_dw(0) != nullptr ){
      tsk->needsLabel(Task::OldDW, VarLabel::find("delT"));
      tsk->needsLabel(Task::OldDW, VarLabel::find(simTime_name));
    }

    // This task had no work to perform.  Delete it.
    if ( counter == 0 ) {
      delete tsk;
      tsk = nullptr;
    } else {
      if (archesTasksMixMemSpaces) {
        std::cout << std::endl << " WARNING Different execution spaces specified.  All Arches tasks within a single Uintah task must share the same execution space." << std::endl << std::endl;
        for (unsigned int i=0; i<taskNames.size(); i++){
            std::cout <<taskNames[i] << " using execution space enum ->  "<< taskExecSpace[i]   <<" \n";
         } 
        throw InvalidValue("Error: Different execution spaces specified.  All Arches tasks within a single Uintah task must share the same execution space.",__FILE__,__LINE__);
      }
    }
  };

  bool non_const_pack_tasks = pack_tasks;

  // We must know which memory space(s) the Arches task embedded within the Uintah task will execute
  // so Uintah can ensure those simulation variables are prepared in that memory space prior to task execution.
  if (assignedExecutionSpace == TaskAssignedExecutionSpace::KOKKOS_OPENMP) {
    create_portable_tasks( TaskDependencies, this,
                          _factory_name + std::string("::") + task_group_name + std::string("::") + type_string,
                          &TaskFactoryBase::do_task<KOKKOS_OPENMP_TAG>,
                          sched, level->eachPatch(), matls, TASKGRAPH::DEFAULT,
                          variable_registry, arches_tasks, type, time_substep, non_const_pack_tasks);
  } else if (assignedExecutionSpace == TaskAssignedExecutionSpace::KOKKOS_DEVICE) {

    //some race condition in kokkos::parallel_reduce. So combine all patches together in a single reduction task to avoid the multiple cpu threads calling parallel_reduce
    //temp work around until the permanent solution
        if(task_group_name == "density_star" && type == (TaskInterface::TIMESTEP_EVAL)){
          create_portable_tasks( TaskDependencies, this,
                                                          _factory_name + std::string("::") + task_group_name + std::string("::") + type_string,
                                                          &TaskFactoryBase::do_task<KOKKOS_DEFAULT_DEVICE_TAG>,
                                                          sched, sched->getLoadBalancer()->getPerProcessorPatchSet(level), matls, TASKGRAPH::DEFAULT,
                                                          variable_registry, arches_tasks, type, time_substep, non_const_pack_tasks);
          //printf("warning: Creating per processor task for density_star due to race condition in kokkos parallel_reduce %s %d\n", __FILE__, __LINE__);
        }
        else{
      create_portable_tasks( TaskDependencies, this,
                          _factory_name + std::string("::") + task_group_name + std::string("::") + type_string,
                          &TaskFactoryBase::do_task<KOKKOS_DEFAULT_DEVICE_TAG>,
                          sched, level->eachPatch(), matls, TASKGRAPH::DEFAULT,
                          variable_registry, arches_tasks, type, time_substep, non_const_pack_tasks);
        }

  } else { //if (assignedExecutionSpace == TaskAssignedExecutionSpace::UINTAH_CPU) {
    create_portable_tasks( TaskDependencies, this,
                          _factory_name + std::string("::") + task_group_name + std::string("::") + type_string,
                          &TaskFactoryBase::do_task<UINTAH_CPU_TAG>,
                          sched, level->eachPatch(), matls, TASKGRAPH::DEFAULT,
                          variable_registry, arches_tasks, type, time_substep, non_const_pack_tasks);
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void TaskFactoryBase::do_task ( const PatchSubset* patches,
                                const MaterialSubset* matls,
                                OnDemandDataWarehouse* old_dw,
                                OnDemandDataWarehouse* new_dw,
                                UintahParams& uintahParams,
                                ExecutionObject<ExecSpace, MemSpace>& execObj,
                                std::vector<ArchesFieldContainer::VariableInformation> variable_registry,
                                std::vector<TaskInterface*> arches_tasks,
                                TaskInterface::TASK_TYPE type,
                                int time_substep,
                                const bool packed_tasks ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(patch, m_matl_index,
                                            variable_registry, old_dw, new_dw);

    SchedToTaskInfo info;

    if ( old_dw != nullptr ){
      //get the current dt
      delt_vartype DT;
      old_dw->get(DT, VarLabel::find("delT"));
      info.dt = DT;
      info.time_substep = time_substep;

      //get the time
      simTime_vartype simTime;
      old_dw->get(simTime, VarLabel::find(simTime_name));
      info.time = simTime;
    }

    info.packed_tasks = packed_tasks;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager( variable_registry,
                                                                         patch, info );

    tsk_info_mngr->set_field_container( field_container );

    for ( auto i_task = arches_tasks.begin(); i_task != arches_tasks.end(); i_task++ ){

      DOUT( dbg_arches_task, "[TaskFactoryBase]   " << _factory_name << " is executing "
        << (*i_task)->get_task_name() << " with function " << get_task_exec_str(type) );

      switch( type ){
        case (TaskInterface::INITIALIZE):
          {
            (*i_task)->initialize<ExecSpace, MemSpace>( patch, tsk_info_mngr, execObj );
          }
          break;
        case (TaskInterface::RESTART_INITIALIZE):
          {
            (*i_task)->restart_initialize<ExecSpace, MemSpace>( patch, tsk_info_mngr, execObj );
          }
          break;
        case (TaskInterface::TIMESTEP_INITIALIZE):
          {
            (*i_task)->timestep_init<ExecSpace, MemSpace>( patch, tsk_info_mngr, execObj );
            time_substep = 0;
          }
          break;
        case (TaskInterface::TIMESTEP_EVAL):
          {
            (*i_task)->eval<ExecSpace, MemSpace>( patch, tsk_info_mngr, execObj );
          }
          break;
        case (TaskInterface::BC):
          {
            (*i_task)->compute_bcs<ExecSpace, MemSpace>( patch, tsk_info_mngr, execObj );
          }
          break;
        case (TaskInterface::ATOMIC):
          {
            (*i_task)->eval<ExecSpace, MemSpace>( patch, tsk_info_mngr, execObj );
          }
          break;
        default:
          throw InvalidValue("Error: TASK_TYPE not recognized.",__FILE__,__LINE__);
          break;
      }
    }

    //clean up
    delete tsk_info_mngr;
    delete field_container;

  }
}

//--------------------------------------------------------------------------------------------------
void TaskFactoryBase::print_variable_max_ghost(){

  std::stringstream msg;
  msg << " :: Reporting max ghost cells per Factory :: " << std::endl;
  msg << "       Factory = " << _factory_name << std::endl;

  for ( auto i = m_variable_ghost_info.begin(); i != m_variable_ghost_info.end(); i++ ){
    msg << "   Variable: " << i->first << std::endl;
    if ( i->second.numTasksNewDW > 0 ){
      msg << "        Min NewDW Ghost: " << i->second.min_newdw_ghost << " Max NewDW Ghost: " << i->second.max_newdw_ghost <<
      " across " << i->second.numTasksNewDW << " tasks. " << std::endl;
      msg << "        In the following tasks: " << std::endl;
      for (auto niter = i->second.taskNamesNewDW.begin();
           niter != i->second.taskNamesNewDW.end(); niter++ ){
        msg << "         " << *niter << std::endl;
      }
    }
    if ( i->second.numTasksOldDW > 0 ){
      msg << "        Min OldDW Ghost: " << i->second.min_olddw_ghost << " Max OldDW Ghost: " << i->second.max_olddw_ghost <<
      " across " << i->second.numTasksOldDW << " tasks. " << std::endl;
      msg << "        In the following tasks: " << std::endl;
      for (auto niter = i->second.taskNamesOldDW.begin();
         niter != i->second.taskNamesOldDW.end(); niter++ ){
           msg << "         " << *niter << std::endl;
      }
    }
  }

  msg << " :: End report of max ghost cells :: " << std::endl;
  DOUT( dbg_fac_vartask_dep, msg.str());

}
