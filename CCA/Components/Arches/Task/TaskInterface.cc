#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>

//Uintah Includes:

//3P Includes:
//#include <boost/foreach.hpp>

using namespace Uintah;
namespace so = SpatialOps;

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
                                  VAR_TYPE type,
                                  VAR_DEPEND dep,
                                  int nGhost,
                                  WHICH_DW dw,
                                  std::vector<VariableInformation>& variable_registry,
                                  const int time_substep ){

  register_variable_work( name, type, dep, nGhost, dw, variable_registry, time_substep );

}

void
TaskInterface::register_variable( std::string name,
                                  VAR_TYPE type,
                                  VAR_DEPEND dep,
                                  int nGhost,
                                  WHICH_DW dw,
                                  std::vector<VariableInformation>& variable_registry ){

  register_variable_work( name, type, dep, nGhost, dw, variable_registry, 0 );

}

void
TaskInterface::register_variable( std::string name,
                                  VAR_TYPE type,
                                  VAR_DEPEND dep,
                                  std::vector<VariableInformation>& variable_registry ){

  WHICH_DW dw = NEWDW;
  int nGhost = 0;

  register_variable_work( name, type, dep, nGhost, dw, variable_registry, 0 );

}

void
TaskInterface::register_variable( std::string name,
                                  VAR_TYPE type,
                                  VAR_DEPEND dep,
                                  std::vector<VariableInformation>& variable_registry,
                                  const int timesubstep ){

  WHICH_DW dw = NEWDW;
  int nGhost = 0;

  register_variable_work( name, type, dep, nGhost, dw, variable_registry, timesubstep );

}

//====================================================================================
//
//====================================================================================
void
TaskInterface::register_variable_work( std::string name,
                                       VAR_TYPE type,
                                       VAR_DEPEND dep,
                                       int nGhost,
                                       WHICH_DW dw,
                                       std::vector<VariableInformation>& variable_registry,
                                       const int time_substep ){

  VariableInformation info;

  info.name   = name;
  info.depend = dep;
  info.dw     = dw;
  info.type   = type;
  info.nGhost = nGhost;
  info.dw_inquire = false;
  info.local = false;

  switch (dw) {

  case OLDDW:

    info.uintah_task_dw = Task::OldDW;
    break;

  case NEWDW:

    info.uintah_task_dw = Task::NewDW;
    break;

  case LATEST:

    info.dw_inquire = true;
    break;

  default:

    throw InvalidValue("Arches Task Error: Cannot determine the DW needed for variable: "+name, __FILE__, __LINE__);
    break;

  }

  //check for conflicts:
  if ( (dep == COMPUTES && dw == OLDDW) ||
       (dep == LOCAL_COMPUTES && dw == OLDDW) ) {
    throw InvalidValue("Arches Task Error: Cannot COMPUTE (COMPUTES) a variable from OldDW for variable: "+name, __FILE__, __LINE__);
  }

  if ( (dep == MODIFIES && dw == OLDDW) ) {
    throw InvalidValue("Arches Task Error: Cannot MODIFY a variable from OldDW for variable: "+name, __FILE__, __LINE__);
  }

  if ( dep == COMPUTES ||
       dep == LOCAL_COMPUTES ) {

    if ( nGhost > 0 ) {

      std::cout << "Arches Task Warning: Variable COMPUTE (COMPUTES) found that is requesting ghosts for: "+name+" Nghosts set to zero!" << std::endl;
      info.nGhost = 0;

    }
  }

  //create new varlabels if needed
  //NOTE: We aren't going to check here
  //to make sure that other variables are
  //created somewhere else.  That check
  //will be done later.
  if ( dep == LOCAL_COMPUTES ) {


    const VarLabel* test = NULL;
    test = VarLabel::find( name );

    if ( test == NULL && time_substep == 0 ) {

      if ( type == CC_INT ) {
        info.label = VarLabel::create( name, CCVariable<int>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == CC_DOUBLE ) {
        info.label = VarLabel::create( name, CCVariable<double>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == CC_VEC ) {
        info.label = VarLabel::create( name, CCVariable<Vector>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == FACEX ) {
        info.label = VarLabel::create( name, SFCXVariable<double>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == FACEY ) {
        info.label = VarLabel::create( name, SFCYVariable<double>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == FACEZ ) {
        info.label = VarLabel::create( name, SFCZVariable<double>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == SUM ) {
        info.label = VarLabel::create( name, sum_vartype::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == MAX ) {
        info.label = VarLabel::create( name, max_vartype::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == MIN ) {
        info.label = VarLabel::create( name, min_vartype::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      } else if ( type == PARTICLE ) {
        info.label = VarLabel::create( name, ParticleVariable<double>::getTypeDescription() );
        info.local = true;
        _local_labels.push_back(info.label);
      }

      info.depend = COMPUTES;
      dep = COMPUTES;

    } else if ( test != NULL && time_substep > 0 ) {

      //because computing happens on time_substep = 0
      //checking for duplicate labels occurred upstream
      info.depend = MODIFIES;
      dep = MODIFIES;

    } else {

      throw InvalidValue("Arches Task Error: Trying to create a local variable, "+name+", that already exists for Arches Task: "+_task_name, __FILE__, __LINE__);

    }

  }

  if ( dep == REQUIRES ) {

    if ( type == CC_INT ) {
      if ( nGhost == 0 ) {
        info.ghost_type = Ghost::None;
      } else {
        info.ghost_type = Ghost::AroundCells;
      }
    } else if ( type == CC_DOUBLE ) {
      if ( nGhost == 0 ) {
        info.ghost_type = Ghost::None;
      } else {
        info.ghost_type = Ghost::AroundCells;
      }
    } else if ( type == CC_VEC ) {
      if ( nGhost == 0 ) {
        info.ghost_type = Ghost::None;
      } else {
        info.ghost_type = Ghost::AroundCells;
      }
    } else if ( type == FACEX ) {
      if ( nGhost == 0 ) {
        info.ghost_type = Ghost::None;
      } else {
        info.ghost_type = Ghost::AroundFaces;
      }
    } else if ( type == FACEY ) {
      if ( nGhost == 0 ) {
        info.ghost_type = Ghost::None;
      } else {
        info.ghost_type = Ghost::AroundFaces;
      }
    } else if ( type == FACEZ ) {
      if ( nGhost == 0 ) {
        info.ghost_type = Ghost::None;
      } else {
        info.ghost_type = Ghost::AroundFaces;
      }
    }
  }

  //label will be matched later.
  //info.label = NULL;

  //load the variable on the registry:
  variable_registry.push_back( info );

}

//====================================================================================
//
//====================================================================================
void
TaskInterface::resolve_labels( std::vector<VariableInformation>& variable_registry ){

  BOOST_FOREACH( VariableInformation &ivar, variable_registry ){

    ivar.label = VarLabel::find( ivar.name );

    if ( ivar.label == NULL ) {
      throw InvalidValue("Arches Task Error: Cannot resolve variable label for task execution: "+ivar.name, __FILE__, __LINE__);
    }
  }
}

//====================================================================================
//
//====================================================================================
template <class T>
void TaskInterface::resolve_field_requires( DataWarehouse* old_dw,
                                            DataWarehouse* new_dw,
                                            T* field,
                                            VariableInformation& info,
                                            const Patch* patch,
                                            const int time_substep ){

  if ( info.dw_inquire ) {
    if ( time_substep > 0 ) {
      new_dw->get( *field, info.label, _matl_index, patch, info.ghost_type, info.nGhost );
    } else {
      old_dw->get( *field, info.label, _matl_index, patch, info.ghost_type, info.nGhost );
    }
  } else {
    if ( info.dw == OLDDW ) {
      old_dw->get( *field, info.label, _matl_index, patch, info.ghost_type, info.nGhost );
    } else {
      new_dw->get( *field, info.label, _matl_index, patch, info.ghost_type, info.nGhost );
    }
  }

}

//====================================================================================
//
//====================================================================================
template <class T>
void TaskInterface::resolve_field_modifycompute( DataWarehouse* old_dw, DataWarehouse* new_dw,
                                                 T* field, VariableInformation& info,
                                                 const Patch* patch, const int time_substep ){

  switch(info.depend) {

  case COMPUTES:

    new_dw->allocateAndPut( field, info.label, _matl_index, patch );
    break;

  case MODIFIES:

    new_dw->getModifiable( field, info.label, _matl_index, patch );
    break;

  default:

    throw InvalidValue("Arches Task Error: Cannot resolve DW dependency for variable: "+info.name, __FILE__, __LINE__);

  }

}

//====================================================================================
//
//====================================================================================
void TaskInterface::resolve_fields( DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const Patch* patch,
                                    ArchesFieldContainer* field_container,
                                    ArchesTaskInfoManager* f_collector,
                                    const bool doing_init ){


  std::vector<VariableInformation>& variable_registry = f_collector->get_variable_reg();

  int time_substep = f_collector->get_time_substep();

  //loop through all the fields and do the allocates, modifies, and gets
  //stuff the resultant fields into a map for later reference.
  BOOST_FOREACH( VariableInformation &ivar, variable_registry ){

    switch ( ivar.type ) {

    case CC_INT:
      if ( ivar.depend == REQUIRES ) {

        constCCVariable<int>* var = scinew constCCVariable<int>;
        resolve_field_requires( old_dw, new_dw, var, ivar, patch, time_substep );
        ArchesFieldContainer::ConstFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_INT);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_variable(ivar.name, icontain);


      } else if ( ivar.depend == MODIFIES ) {

        CCVariable<int>* var = scinew CCVariable<int>;
        new_dw->getModifiable( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_INT);
        field_container->add_variable(ivar.name, icontain);

      } else {

        CCVariable<int>* var = scinew CCVariable<int>;
        new_dw->allocateAndPut( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_INT);
        field_container->add_variable(ivar.name, icontain);

      }
      break;

    case CC_DOUBLE:

      if ( ivar.depend == REQUIRES ) {

        constCCVariable<double>* var = scinew constCCVariable<double>;
        resolve_field_requires( old_dw, new_dw, var, ivar, patch, time_substep );
        ArchesFieldContainer::ConstFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_DOUBLE);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_variable(ivar.name, icontain);

      } else if ( ivar.depend == MODIFIES ) {

        CCVariable<double>* var = scinew CCVariable<double>;
        new_dw->getModifiable( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_DOUBLE);
        field_container->add_variable(ivar.name, icontain);

      } else {

        CCVariable<double>* var = scinew CCVariable<double>;
        new_dw->allocateAndPut( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_DOUBLE);
        field_container->add_variable(ivar.name, icontain);

      }
      break;

    case CC_VEC:

      if ( ivar.depend == REQUIRES ) {

        constCCVariable<Vector>* var = scinew constCCVariable<Vector>;
        resolve_field_requires( old_dw, new_dw, var, ivar, patch, time_substep );
        ArchesFieldContainer::ConstFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_VEC);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_variable(ivar.name, icontain);

      } else if ( ivar.depend == MODIFIES ) {

        CCVariable<Vector>* var = scinew CCVariable<Vector>;
        new_dw->getModifiable( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_VEC);
        field_container->add_variable(ivar.name, icontain);

      } else {

        CCVariable<Vector>* var = scinew CCVariable<Vector>;
        new_dw->allocateAndPut( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::CC_VEC);
        field_container->add_variable(ivar.name, icontain);

      }
      break;

    case FACEX:

      if ( ivar.depend == REQUIRES ) {

        constSFCXVariable<double>* var = scinew constSFCXVariable<double>;
        resolve_field_requires( old_dw, new_dw, var, ivar, patch, time_substep );
        ArchesFieldContainer::ConstFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEX);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_variable(ivar.name, icontain);

      } else if ( ivar.depend == MODIFIES ) {

        SFCXVariable<double>* var = scinew SFCXVariable<double>;
        new_dw->getModifiable( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEX);
        field_container->add_variable(ivar.name, icontain);

      } else {

        SFCXVariable<double>* var = scinew SFCXVariable<double>;
        new_dw->allocateAndPut( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEX);
        field_container->add_variable(ivar.name, icontain);

      }
      break;

    case FACEY:

      if ( ivar.depend == REQUIRES ) {

        constSFCYVariable<double>* var = scinew constSFCYVariable<double>;
        resolve_field_requires( old_dw, new_dw, var, ivar, patch, time_substep );
        ArchesFieldContainer::ConstFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEY);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_variable(ivar.name, icontain);

      } else if ( ivar.depend == MODIFIES ) {

        SFCYVariable<double>* var = scinew SFCYVariable<double>;
        new_dw->getModifiable( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEY);
        field_container->add_variable(ivar.name, icontain);

      } else {

        SFCYVariable<double>* var = scinew SFCYVariable<double>;
        new_dw->allocateAndPut( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEY);
        field_container->add_variable(ivar.name, icontain);

      }
      break;

    case FACEZ:

      if ( ivar.depend == REQUIRES ) {

        constSFCZVariable<double>* var = scinew constSFCZVariable<double>;
        resolve_field_requires( old_dw, new_dw, var, ivar, patch, time_substep );
        ArchesFieldContainer::ConstFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEZ);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_variable(ivar.name, icontain);

      } else if ( ivar.depend == MODIFIES ) {

        SFCZVariable<double>* var = scinew SFCZVariable<double>;
        new_dw->getModifiable( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEZ);
        field_container->add_variable(ivar.name, icontain);

      } else {

        SFCZVariable<double>* var = scinew SFCZVariable<double>;
        new_dw->allocateAndPut( *var, ivar.label, _matl_index, patch );
        ArchesFieldContainer::FieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::FACEZ);
        field_container->add_variable(ivar.name, icontain);

      }
      break;

    case PARTICLE:

      if ( ivar.depend == REQUIRES ) {

        constParticleVariable<double>* var = scinew constParticleVariable<double>;

        if ( ivar.dw_inquire ) {
          if ( time_substep > 0 ) {
            ParticleSubset* subset = new_dw->getParticleSubset( _matl_index, patch );
            new_dw->get( *var, ivar.label, subset );
          } else {
            ParticleSubset* subset = old_dw->getParticleSubset( _matl_index, patch );
            old_dw->get( *var, ivar.label, subset );
          }
        } else {
          if ( ivar.dw == OLDDW ) {
            ParticleSubset* subset = old_dw->getParticleSubset( _matl_index, patch );
            old_dw->get( *var, ivar.label, subset );
          } else {
            ParticleSubset* subset = new_dw->getParticleSubset( _matl_index, patch );
            new_dw->get( *var, ivar.label, subset );
          }
        }

        ArchesFieldContainer::ConstParticleFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::PARTICLE);
        icontain.set_ghosts(ivar.nGhost);
        field_container->add_const_particle_variable(ivar.name, icontain);

      } else if ( ivar.depend == MODIFIES ) {

        //not sure what to do here about the particleSubset...
        //for now grabbing only from old:
        ParticleSubset* subset = old_dw->getParticleSubset( _matl_index, patch );
        ParticleVariable<double>* var = scinew ParticleVariable<double>;
        new_dw->getModifiable( *var, ivar.label, subset );
        ArchesFieldContainer::ParticleFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::PARTICLE);
        icontain.set_ghosts(0);
        field_container->add_particle_variable(ivar.name, icontain);

      } else {

        //not sure what to do here about the particleSubset...
        //for now grabbing only from old except for the case of initialization:
        ParticleSubset* subset;
        if ( doing_init ) {
          subset = new_dw->getParticleSubset( _matl_index, patch );
        } else {
          subset = old_dw->getParticleSubset( _matl_index, patch );
        }
        ParticleVariable<double>* var = scinew ParticleVariable<double>;
        new_dw->allocateAndPut( *var, ivar.label, subset );
        ArchesFieldContainer::ParticleFieldContainer icontain;
        icontain.set_field(var);
        icontain.set_field_type(ArchesFieldContainer::PARTICLE);
        icontain.set_ghosts(0);
        field_container->add_particle_variable(ivar.name, icontain);

      }
      break;

    default:
      throw InvalidValue("Arches Task Error: Cannot resolve DW dependency for variable: "+ivar.name, __FILE__, __LINE__);
      break;

    }
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

  std::vector<VariableInformation> variable_registry;

  register_timestep_eval( variable_registry, time_substep );

  resolve_labels( variable_registry );

  Task* tsk;

  if ( task_type == STANDARD_TASK )
    tsk = scinew Task( _task_name, this, &TaskInterface::do_task, variable_registry, time_substep );
  else if ( task_type == BC_TASK )
    tsk = scinew Task( _task_name+"_bc_task", this, &TaskInterface::do_bcs, variable_registry, time_substep );
  else
    throw InvalidValue("Error: Task type not recognized.",__FILE__,__LINE__);

  int counter = 0;

  BOOST_FOREACH( VariableInformation &ivar, variable_registry ){

    counter++;

    switch(ivar.depend) {

    case COMPUTES:
      if ( time_substep == 0 ) {
        tsk->computes( ivar.label );   //only compute on the zero time substep
      } else {
        tsk->modifies( ivar.label );
        ivar.dw = NEWDW;
        ivar.uintah_task_dw = Task::NewDW;
        ivar.depend = MODIFIES;
      }
      break;
    case MODIFIES:
      tsk->modifies( ivar.label );
      break;
    case REQUIRES:
      if ( ivar.dw_inquire ) {
        if ( time_substep > 0 ) {
          ivar.dw = NEWDW;
          ivar.uintah_task_dw = Task::NewDW;
        } else {
          ivar.dw = OLDDW;
          ivar.uintah_task_dw = Task::OldDW;
        }
      } else {
        if ( ivar.dw == OLDDW ) {
          ivar.uintah_task_dw = Task::OldDW;
        } else {
          ivar.uintah_task_dw = Task::NewDW;
        }
      }
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

  std::vector<VariableInformation> variable_registry;

  if ( is_restart ) {
    register_restart_initialize( variable_registry );
  } else {
    register_initialize( variable_registry );
  }

  resolve_labels( variable_registry );

  Task* tsk;
  if ( is_restart ) {
    tsk = scinew Task( _task_name+"_restart_initialize", this, &TaskInterface::do_restart_init, variable_registry );
  } else {
    tsk = scinew Task( _task_name+"_initialize", this, &TaskInterface::do_init, variable_registry );
  }

  int counter = 0;

  BOOST_FOREACH( VariableInformation &ivar, variable_registry ){

    counter++;

    if ( ivar.dw == OLDDW ) {
      throw InvalidValue("Arches Task Error: Cannot use OLDDW for initialization task: "+_task_name, __FILE__, __LINE__);
    }

    switch(ivar.depend) {

    case COMPUTES:
      tsk->computes( ivar.label );
      break;
    case MODIFIES:
      tsk->modifies( ivar.label );
      break;
    case REQUIRES:
      ivar.dw = NEWDW;
      ivar.uintah_task_dw = Task::NewDW;
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

  std::vector<VariableInformation> variable_registry;

  register_timestep_init( variable_registry );

  resolve_labels( variable_registry );

  Task* tsk = scinew Task( _task_name+"_timestep_initialize", this, &TaskInterface::do_timestep_init, variable_registry );

  int counter = 0;

  BOOST_FOREACH( VariableInformation &ivar, variable_registry ){

    counter++;

    switch(ivar.depend) {

    case COMPUTES:
      tsk->computes( ivar.label );   //only compute on the zero time substep
      break;
    case MODIFIES:
      tsk->modifies( ivar.label );
      break;
    case REQUIRES:
      if ( ivar.dw_inquire ) {
        ivar.dw = OLDDW;
        ivar.uintah_task_dw = Task::OldDW;
      } else {
        if ( ivar.dw == OLDDW ) {
          ivar.uintah_task_dw = Task::OldDW;
        } else {
          ivar.uintah_task_dw = Task::NewDW;
        }
      }
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
// (do tasks)
//====================================================================================
void TaskInterface::do_task( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             std::vector<VariableInformation> variable_registry,
                             int time_substep ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    const Wasatch::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch);

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = time_substep;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //doing DW gets...
    resolve_fields( old_dw, new_dw, patch, field_container, tsk_info_mngr, false );

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
                            std::vector<VariableInformation> variable_registry,
                            int time_substep ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    const Wasatch::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch);

    SchedToTaskInfo info;

    //get the current dt
    delt_vartype DT;
    old_dw->get(DT, VarLabel::find("delT"));
    info.dt = DT;
    info.time_substep = time_substep;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //doing DW gets...
    resolve_fields( old_dw, new_dw, patch, field_container, tsk_info_mngr, false );

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

void TaskInterface::do_init( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             std::vector<VariableInformation> variable_registry ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    const Wasatch::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch);

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //doing DW gets...
    resolve_fields( old_dw, new_dw, patch, field_container, tsk_info_mngr, true );

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
                                     std::vector<VariableInformation> variable_registry ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    const Wasatch::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch);

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //doing DW gets...
    resolve_fields( old_dw, new_dw, patch, field_container, tsk_info_mngr, true );

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
                                      std::vector<VariableInformation> variable_registry ){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    const Wasatch::AllocInfo ainfo( old_dw, new_dw, _matl_index, patch, pc );

    ArchesFieldContainer* field_container = scinew ArchesFieldContainer(ainfo, patch);

    SchedToTaskInfo info;

    //get the current dt
    info.dt = 0;
    info.time_substep = 0;

    ArchesTaskInfoManager* tsk_info_mngr = scinew ArchesTaskInfoManager(variable_registry, patch, info);

    //doing DW gets...
    resolve_fields( old_dw, new_dw, patch, field_container, tsk_info_mngr, false );

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
