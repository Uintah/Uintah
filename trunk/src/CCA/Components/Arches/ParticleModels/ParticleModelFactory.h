#ifndef UT_ParticleModelFactory_h
#define UT_ParticleModelFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class ParticleModelFactory : public TaskFactoryBase {

  public:

    ParticleModelFactory( const ApplicationCommon* arches );
    ~ParticleModelFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

      if ( subset == "particle_models"){

        return m_particle_models;

      } else if ( subset == "deposition_models" ){

        return m_deposition_models;

      } else if ( subset == "dqmom_transport_variables"){

        return m_dqmom_transport_variables;

      } else if ( subset == "particle_properties"){

        return m_particle_properties;

      } else if ( subset == _all_tasks_str ){

        return _active_tasks;

      } else {
        throw InvalidValue("Error: Task subset not recognized in ParticleModelFactory:  "+subset, __FILE__,__LINE__);
      }
    }

  private:

    std::vector<std::string> m_particle_models;             ///< Particle Models for IC's
    std::vector<std::string> m_deposition_models;           ///< Wall deposition models
    std::vector<std::string> m_particle_properties;         ///< IC dependent properties
    std::vector<std::string> m_dqmom_transport_variables;   ///< Variables needed for transport construction

  };
}
#endif
