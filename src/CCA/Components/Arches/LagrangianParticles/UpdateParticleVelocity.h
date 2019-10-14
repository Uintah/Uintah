#ifndef Uintah_Component_Arches_UpdateParticleVelocity_h
#define Uintah_Component_Arches_UpdateParticleVelocity_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class UpdateParticleVelocity : public TaskInterface {

public:

    UpdateParticleVelocity( std::string task_name, int matl_index );
    ~UpdateParticleVelocity();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ); 

    void create_local_labels(){}

    //Build instructions for this (UpdateParticleVelocity) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      UpdateParticleVelocity* build()
      { return scinew UpdateParticleVelocity( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    std::string _px_name;
    std::string _py_name;
    std::string _pz_name;

    std::string _u_name;
    std::string _v_name;
    std::string _w_name;

  };
}
#endif
