#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

//Specific model headers: 
#include <CCA/Components/Arches/ParticleModels/ExampleParticleModel.h>

using namespace Uintah; 

ParticleModelFactory::ParticleModelFactory()
{}

ParticleModelFactory::~ParticleModelFactory()
{}

void 
ParticleModelFactory::register_all_tasks( ProblemSpecP& db )
{ 
  /*
   
    <ParticleModels>
      <model label="some unique name" type="defined type">
  
        <spec consistent with model type> 

      </model>
    </ParticleModels>


   */


  if ( db->findBlock("ParticleModels")){ 

    ProblemSpecP db_pm = db->findBlock("ParticleModels"); 

    for (ProblemSpecP db_model = db_pm->findBlock("model"); db_model != 0; 
         db_model = db_model->findNextBlock("model")){

      std::string model_name; 
      std::string type; 
      db_model->getAttribute("label",model_name ); 
      db_model->getAttribute("type", type ); 

      //stub for number of environments
      int N; 
      db_model->require("N",N); 

      typedef SpatialOps::SVolField SVol;

      if ( type == "simple_rate"){ 

        std::string dependent_type; 
        std::string independent_type; 
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type); 
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type); 

        if ( dependent_type == "svol" ){ 

          if ( independent_type == "svol"){ 

            std::string task_name = type + "_" + model_name; 

            TaskInterface::TaskBuilder* tsk = scinew 
              ExampleParticleModel<SVol,SVol>::Builder(task_name, 0, model_name, N); 

            register_task( task_name, tsk ); 
            _active_tasks.push_back(task_name); 

          } else { 
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else { 
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else { 
        throw InvalidValue("Error: Particle model not recognized.",__FILE__,__LINE__);
      }

    }
  }

}

void 
ParticleModelFactory::build_all_tasks( ProblemSpecP& db )
{ 

  if ( db->findBlock("ParticleModels")){ 

    ProblemSpecP db_pm = db->findBlock("ParticleModels"); 

    for (ProblemSpecP db_model = db_pm->findBlock("model"); db_model != 0; 
         db_model = db_model->findNextBlock("model")){

      std::string model_name; 
      std::string type; 
      db_model->getAttribute("label",model_name ); 
      db_model->getAttribute("type", type ); 

      std::string task_name = type + "_" + model_name; 

      TaskInterface* tsk = retrieve_task(task_name);

      tsk->problemSetup( db_model ); 

    }
  }
}
