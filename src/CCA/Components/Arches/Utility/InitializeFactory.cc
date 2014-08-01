#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <CCA/Components/Arches/Utility/WaveFormInit.h>
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

        std::string gtype; 
        db_task->findBlock("grid")->getAttribute("type",gtype);

        if ( gtype == "svol"){ 
          typedef SpatialOps::SVolField SVol;
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<SVol>::Builder(task_name, 0, eqn_name );
          register_task(task_name, tsk);

          _active_tasks.push_back(task_name); 

// Need to make operators to interpolate the grid positions to faces 
// in order for this to work. 
        } else if ( gtype == "xvol"){ 
          typedef SpatialOps::SSurfXField XVol;
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<XVol>::Builder(task_name, 0, eqn_name );
          register_task(task_name, tsk);

          _active_tasks.push_back(task_name); 
        } else if ( gtype == "yvol"){ 
          typedef SpatialOps::SSurfYField YVol;
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<YVol>::Builder(task_name, 0, eqn_name );
          register_task(task_name, tsk);

          _active_tasks.push_back(task_name); 
        } else if ( gtype == "zvol"){ 
          typedef SpatialOps::SSurfZField ZVol;
          TaskInterface::TaskBuilder* tsk = scinew WaveFormInit<ZVol>::Builder(task_name, 0, eqn_name );
          register_task(task_name, tsk);

          _active_tasks.push_back(task_name); 

        } else { 
          throw InvalidValue("Error: Grid type not recognized (must be a SO field).",__FILE__,__LINE__);
        }

      } else { 
        throw InvalidValue("Error: Initialization function not recognized.",__FILE__,__LINE__);
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
      std::string eqn_name; 
      std::string type; 
      db_task->getAttribute("label",task_name ); 
      db_task->getAttribute("eqn", eqn_name );
      db_task->getAttribute("type", type ); 

      if ( type == "wave" ){ 

        TaskInterface* tsk = retrieve_task(task_name); 
        tsk->problemSetup( db_task ); 

      } else { 
        throw InvalidValue("Error: Initialization function not recognized.",__FILE__,__LINE__);
      }


    }
  }
}
