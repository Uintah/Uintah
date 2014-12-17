#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

//Specific model headers: 
#include <CCA/Components/Arches/ParticleModels/ExampleParticleModel.h>
#include <CCA/Components/Arches/ParticleModels/DragModel.h>
#include <CCA/Components/Arches/ParticleModels/BodyForce.h>
#include <CCA/Components/Arches/ParticleModels/Constant.h>

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
  int N;

  if (db->findBlock("CQMOM") ) {
    std::vector<int> N_i;
    db->findBlock("CQMOM")->require("QuadratureNodes",N_i);
    N = 1;
    std::cout << "N: " << N << std::endl;
    for (unsigned int i = 0; i < N_i.size(); i++ ) {
      
      N *= N_i[i];
      std::cout << "N: " << N << std::endl;
    }
  } else if (db->findBlock("DQMOM") ) {
    db->findBlock("DQMOM")->require("number_quad_nodes",N);
  }

  if ( db->findBlock("ParticleModels")){

    ProblemSpecP db_pm = db->findBlock("ParticleModels"); 

    for (ProblemSpecP db_model = db_pm->findBlock("model"); db_model != 0; 
         db_model = db_model->findNextBlock("model")){

      std::string model_name; 
      std::string type; 
      db_model->getAttribute("label",model_name ); 
      db_model->getAttribute("type", type ); 

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

        
      } else if  ( type == "drag" ) {
      
        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);
        
        if ( dependent_type == "svol" ){
          
          if ( independent_type == "svol"){
            
            std::string task_name = type + "_" + model_name;
            
            TaskInterface::TaskBuilder* tsk = scinew
            DragModel<SVol,SVol>::Builder(task_name, 0, model_name, N);
            
            register_task( task_name, tsk );
            _active_tasks.push_back(task_name);
            
          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }
          
          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }
        
      } else if  ( type == "gravity" ) {
        
        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);
        
        if ( dependent_type == "svol" ){
          
          if ( independent_type == "svol"){
            
            std::string task_name = type + "_" + model_name;
            
            TaskInterface::TaskBuilder* tsk = scinew
            BodyForce<SVol,SVol>::Builder(task_name, 0, model_name, N);
            
            register_task( task_name, tsk );
            _active_tasks.push_back(task_name);
            
          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }
          
          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }
        
      } else if  ( type == "constant" ) {
        
        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        
        if ( dependent_type == "svol" ){
          
          if ( independent_type == "svol"){
            
            std::string task_name = type + "_" + model_name;
            
            TaskInterface::TaskBuilder* tsk = scinew
            Constant<SVol>::Builder(task_name, 0, model_name, N);
            
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

      tsk->create_local_labels(); 

    }
  }
}
