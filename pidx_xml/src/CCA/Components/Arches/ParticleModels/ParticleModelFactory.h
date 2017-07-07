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

      } else if ( subset == "post_update_particle_models"){

        return _post_update_particle_tasks;

      } else if ( subset == _all_tasks_str ){

        return _active_tasks;

      } else {
        throw InvalidValue("Error: Task subset not recognized in ParticleModelFactory:  "+subset, __FILE__,__LINE__);
      }
    }

  protected:


  private:

    std::vector<std::string> _coal_models;                 ///< Tasks associated with coal
    std::vector<std::string> _post_update_particle_tasks;  ///< Tasks that execute after the timeave
                                                            //  of the particle transport variables

  };
}
#endif
