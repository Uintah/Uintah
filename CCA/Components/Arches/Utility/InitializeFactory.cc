#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <CCA/Components/Arches/Utility/WaveFormInit.h>
#include <CCA/Components/Arches/Utility/RandParticleLoc.h>
#include <CCA/Components/Arches/Utility/InitLagrangianParticleVelocity.h>
#include <CCA/Components/Arches/Utility/InitLagrangianParticleSize.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah;

InitializeFactory::InitializeFactory()
{}

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

    for (ProblemSpecP db_task = db_init->findBlock("task"); db_task != 0;
        db_task = db_task->findNextBlock("task")){

      std::string task_name;
      std::string eqn_name;
      std::string type;
      db_task->getAttribute("task_label",task_name );
      db_task->getAttribute("variable_label", eqn_name );
      db_task->getAttribute("type", type );

      if ( type == "wave" ){

        std::string variable_type;

        db_task->findBlock("wave")->findBlock("grid")->getAttribute("type",variable_type);

        if ( variable_type == "CC"){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<CCVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else if ( variable_type == "FX" ){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SFCXVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else if ( variable_type == "FY" ){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SFCYVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else if ( variable_type == "FZ" ){
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SFCZVariable<double> >::Builder(task_name, 0, eqn_name);
          register_task( task_name, tsk );
        } else {
          throw InvalidValue("Error: Grid type not valid for WaveForm initializer: "+variable_type, __FILE__, __LINE__);
        }

      } else if ( type == "random_lagrangian_particles"){

        TaskInterface::TaskBuilder* tsk = scinew RandParticleLoc::Builder( task_name, 0 );
        register_task( task_name, tsk );

      } else if ( type == "lagrangian_particle_velocity"){

        TaskInterface::TaskBuilder* tsk = scinew InitLagrangianParticleVelocity::Builder( task_name, 0 );
        register_task( task_name, tsk );

      } else if ( type == "lagrangian_particle_size"){

        TaskInterface::TaskBuilder* tsk = scinew InitLagrangianParticleSize::Builder( task_name, 0 );
        register_task( task_name, tsk );

      } else {
        throw InvalidValue("Error: Initialization function not recognized: "+type,__FILE__,__LINE__);
      }
    }
  }
}

void
InitializeFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("Initialization") ){

    ProblemSpecP db_init = db->findBlock("Initialization");

    for (ProblemSpecP db_task = db_init->findBlock("task"); db_task != 0;
        db_task = db_task->findNextBlock("task")){

      std::string task_name;
      db_task->getAttribute("task_label",task_name );

      TaskInterface* tsk = retrieve_task(task_name);
      tsk->problemSetup( db_task );

      tsk->create_local_labels();

    }
  }
}
