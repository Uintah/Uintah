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
      db_task->getAttribute("label",task_name );
      db_task->getAttribute("eqn", eqn_name );
      db_task->getAttribute("type", type );

      if ( type == "wave" ){

        std::string dependent_type;
        std::string independent_type;

        db_task->findBlock("wave")->findBlock("grid")->getAttribute("dependent_type",dependent_type);
        db_task->findBlock("wave")->findBlock("grid")->getAttribute("independent_type", independent_type);

        typedef SpatialOps::SVolField SVol;
        typedef SpatialOps::XVolField XVol;
        typedef SpatialOps::YVolField YVol;
        typedef SpatialOps::ZVolField ZVol;

        if ( dependent_type == "svol"){

          if ( independent_type == "svol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<SVol, SVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else if ( independent_type == "xvol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<XVol, SVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else if ( independent_type == "yvol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<YVol, SVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else if ( independent_type == "zvol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<ZVol, SVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else {
            throw InvalidValue("Error: SpatalOps grid variable type not recognized for waveform.",__FILE__,__LINE__);
          }

        } else if ( dependent_type == "xvol"){

          if ( independent_type == "svol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<SVol, XVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else if ( independent_type == "xvol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<XVol, XVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else if ( independent_type == "yvol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<YVol, XVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else if ( independent_type == "zvol"){
            TaskInterface::TaskBuilder* tsk = new WaveFormInit<ZVol, XVol>::Builder(task_name, 0, eqn_name);
            register_task( task_name, tsk );
          } else {
            throw InvalidValue("Error: SpatalOps grid variable type not recognized for waveform.",__FILE__,__LINE__);
          }

        } else {
          throw InvalidValue("Error: Grid type not recognized (must be a SO field).",__FILE__,__LINE__);
        }

      } else if ( type == "random_lagrangian_particles"){

        TaskInterface::TaskBuilder* tsk = new RandParticleLoc::Builder( task_name, 0 );
        register_task( task_name, tsk );

      } else if ( type == "lagrangian_particle_velocity"){

        TaskInterface::TaskBuilder* tsk = new InitLagrangianParticleVelocity::Builder( task_name, 0 );
        register_task( task_name, tsk );

      } else if ( type == "lagrangian_particle_size"){

        TaskInterface::TaskBuilder* tsk = new InitLagrangianParticleSize::Builder( task_name, 0 );
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
      db_task->getAttribute("label",task_name );

      TaskInterface* tsk = retrieve_task(task_name);
      tsk->problemSetup( db_task );

      tsk->create_local_labels();

    }
  }
}
