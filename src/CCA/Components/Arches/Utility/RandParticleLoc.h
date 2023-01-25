#ifndef Uintah_Component_Arches_RandParticleLoc_h
#define Uintah_Component_Arches_RandParticleLoc_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class RandParticleLoc : public TaskInterface {

public:

    RandParticleLoc( std::string task_name, int matl_index ):TaskInterface(task_name, matl_index){}
    ~RandParticleLoc(){}

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    }

    void create_local_labels(){}

    //Build instructions for this (RandParticleLoc) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      RandParticleLoc* build()
      { return scinew RandParticleLoc( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    double _value;

    std::string _px_name;
    std::string _py_name;
    std::string _pz_name;

  };
}
#endif
