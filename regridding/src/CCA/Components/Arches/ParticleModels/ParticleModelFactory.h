#ifndef UT_ParticleModelFactory_h
#define UT_ParticleModelFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{ 

  class ParticleModelFactory : public TaskFactoryBase { 

  public: 

    ParticleModelFactory(); 
    ~ParticleModelFactory(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

    std::vector<std::string> retrieve_task_subset( const std::string subset ) { 

      if ( subset == "coal_models"){ 

        return _coal_models; 

      } else if ( subset == "post_update_coal"){ 

        return _post_update_coal_tasks; 

      } else if ( subset == "pre_update_particle_models"){

        return _pre_update_particle_tasks; 

      } else { 
        throw InvalidValue("Error: Task subset not recognized in ParticleModelFactory:  "+subset, __FILE__,__LINE__);
      }
    }

  protected: 


  private: 

    std::vector<std::string> _coal_models;                ///<Tasks associated with coal 
    std::vector<std::string> _post_update_coal_tasks;     ///<Tasks that execute at the end of the DQMOM update
    std::vector<std::string> _pre_update_particle_tasks;  ///<Tasks that execute at the start of an RK step 
  
  };
}
#endif 
