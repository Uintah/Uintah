#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <CCA/Components/Arches/Utility/WaveFormInit.h>
#include <CCA/Components/Arches/Utility/RandParticleLoc.h>
#include <CCA/Components/Arches/Utility/ShunnMMS.h>
#include <CCA/Components/Arches/Utility/ShunnMMSP3.h>
#include <CCA/Components/Arches/Utility/AlmgrenMMS.h>
#include <CCA/Components/Arches/Utility/TaylorGreen3D.h>
#include <CCA/Components/Arches/Utility/InitLagrangianParticleVelocity.h>
#include <CCA/Components/Arches/Utility/InitLagrangianParticleSize.h>
#include <CCA/Components/Arches/Utility/FileInit.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah;

InitializeFactory::InitializeFactory( const ApplicationCommon* arches ) : TaskFactoryBase(arches)
{
  _factory_name = "InitializeFactory";
}

InitializeFactory::~InitializeFactory()
{}

void
InitializeFactory::register_all_tasks( ProblemSpecP& db )
{

  /*

    <Initialization>
      <task label="my_init_task">
        details
      </task>
    </Initialization>




  */

  if ( db->findBlock("Initialization") ){

    ProblemSpecP db_init = db->findBlock("Initialization");

    for (ProblemSpecP db_task = db_init->findBlock("task"); db_task != nullptr; db_task = db_task->findNextBlock("task")){

      std::string task_name;
      std::string eqn_name;
      std::string type;
      db_task->getAttribute("task_label",task_name );
      db_task->getAttribute("variable_label", eqn_name );
      db_task->getAttribute("type", type );

      if ( type == "wave" ){

        std::string variable_type;
        std::string indep_variable_type;

        db_task->findBlock("wave")->findBlock("grid")->getAttribute("type",variable_type);
        db_task->findBlock("wave")->findBlock("independent_variable")->getAttribute("type",indep_variable_type);

        if ( variable_type == "CC" && indep_variable_type == "CC"){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<CCVariable<double>, constCCVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else if ( variable_type == "FX" &&  indep_variable_type == "CC" ){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SFCXVariable<double>, constCCVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else if ( variable_type == "FY" &&  indep_variable_type == "CC" ){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SFCYVariable<double>, constCCVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else if ( variable_type == "FZ" && indep_variable_type == "CC" ){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SFCZVariable<double>, constCCVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else {
          throw InvalidValue("Error: Grid type not valid for WaveForm initializer: "+variable_type, __FILE__, __LINE__);
        }
        _unweighted_var_tasks.push_back(task_name);

      } else if ( type == "shunn_mms"){

        std::string var_type;
        db_task->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew ShunnMMS<CCVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FX" ){
          tsk = scinew ShunnMMS<SFCXVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FY" ){
          tsk = scinew ShunnMMS<SFCYVariable<double> >::Builder( task_name, 0, eqn_name );
        } else {
          tsk = scinew ShunnMMS<SFCZVariable<double> >::Builder( task_name, 0, eqn_name );
        }

        register_task( task_name, tsk );
        if (db_task ->findBlock("which_density")){
           _weighted_var_tasks.push_back(task_name);
        } else {
           _unweighted_var_tasks.push_back(task_name);
        }

      } else if ( type == "shunn_mms_p3"){

        std::string var_type;
        db_task->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew ShunnMMSP3<CCVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FX" ){
          tsk = scinew ShunnMMSP3<SFCXVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FY" ){
          tsk = scinew ShunnMMSP3<SFCYVariable<double> >::Builder( task_name, 0, eqn_name );
        } else {
          tsk = scinew ShunnMMSP3<SFCZVariable<double> >::Builder( task_name, 0, eqn_name );
        }

        register_task( task_name, tsk );
        if (db_task ->findBlock("which_density")){
           _weighted_var_tasks.push_back(task_name);
        } else {
           _unweighted_var_tasks.push_back(task_name);
        }
      } else if ( type == "almgren_mms"){

        std::string var_type;
        db_task->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew AlmgrenMMS<CCVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FX" ){
          tsk = scinew AlmgrenMMS<SFCXVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FY" ){
          tsk = scinew AlmgrenMMS<SFCYVariable<double> >::Builder( task_name, 0, eqn_name );
        } else {
          tsk = scinew AlmgrenMMS<SFCZVariable<double> >::Builder( task_name, 0, eqn_name );
        }

        register_task( task_name, tsk );
        _unweighted_var_tasks.push_back(task_name);

        } else if ( type == "taylor_green3d"){

          std::string var_type;
          db_task->findBlock("variable")->getAttribute("type", var_type);

          TaskInterface::TaskBuilder* tsk;

          if ( var_type == "FX" ){
            tsk = scinew TaylorGreen3D<SFCXVariable<double> >::Builder( task_name, 0, eqn_name );
          } else if ( var_type == "FY" ){
            tsk = scinew TaylorGreen3D<SFCYVariable<double> >::Builder( task_name, 0, eqn_name );
          } else {
            tsk = scinew TaylorGreen3D<SFCZVariable<double> >::Builder( task_name, 0, eqn_name );
          }

          register_task( task_name, tsk );
          _unweighted_var_tasks.push_back(task_name);


      } else if ( type == "random_lagrangian_particles"){

        TaskInterface::TaskBuilder* tsk = scinew RandParticleLoc::Builder( task_name, 0 );
        register_task( task_name, tsk );
        _unweighted_var_tasks.push_back(task_name);

      } else if ( type == "lagrangian_particle_velocity"){

        TaskInterface::TaskBuilder* tsk
          = scinew InitLagrangianParticleVelocity::Builder( task_name, 0 );
        register_task( task_name, tsk );
        _unweighted_var_tasks.push_back(task_name);

      } else if ( type == "lagrangian_particle_size"){

        TaskInterface::TaskBuilder* tsk
          = scinew InitLagrangianParticleSize::Builder( task_name, 0 );
        register_task( task_name, tsk );
        _unweighted_var_tasks.push_back(task_name);

      } else if ( type == "input_file" ){

        std::string var_type;
        db_task->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew FileInit<CCVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FX" ){
          tsk = scinew FileInit<SFCXVariable<double> >::Builder( task_name, 0, eqn_name );
        } else if ( var_type == "FY" ){
          tsk = scinew FileInit<SFCYVariable<double> >::Builder( task_name, 0, eqn_name );
        } else {
          tsk = scinew FileInit<SFCZVariable<double> >::Builder( task_name, 0, eqn_name );
        }

        register_task( task_name, tsk );
        _unweighted_var_tasks.push_back(task_name);

      } else {

        throw InvalidValue("Error: Initialization function not recognized: "+type,__FILE__,__LINE__);

      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
InitializeFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("Initialization") ){

    ProblemSpecP db_init = db->findBlock("Initialization");

    for (ProblemSpecP db_task = db_init->findBlock("task"); db_task != nullptr; db_task = db_task->findNextBlock("task")){

      std::string task_name;
      std::string type;
      db_task->getAttribute("task_label",task_name );
      db_task->getAttribute("type", type );

      print_task_setup_info( task_name, type );

      TaskInterface* tsk = retrieve_task(task_name);
      tsk->problemSetup( db_task );

      tsk->create_local_labels();

    }
  }
}

//--------------------------------------------------------------------------------------------------
void InitializeFactory::schedule_initialization( const LevelP& level,
                                                 SchedulerP& sched,
                                                 const MaterialSet* matls,
                                                 bool doing_restart ){

  const bool pack_tasks = false;
  schedule_task_group( "all_tasks", m_task_init_order, TaskInterface::INITIALIZE,
                       pack_tasks, level, sched, matls );
}
