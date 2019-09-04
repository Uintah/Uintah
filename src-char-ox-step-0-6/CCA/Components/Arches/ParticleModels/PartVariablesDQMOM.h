#ifndef Uintah_Component_Arches_PartVariablesDQMOM_h
#define Uintah_Component_Arches_PartVariablesDQMOM_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class PartVariablesDQMOM : public TaskInterface { 

public:

    PartVariablesDQMOM( std::string task_name, int matl_index );
    ~PartVariablesDQMOM(){}

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){};

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();


    //Build instructions for this (PartVariablesDQMOM) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      PartVariablesDQMOM* build()
      { return scinew PartVariablesDQMOM( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    int m_Nenv;
    std::string m_length_root;
    std::string m_number_density_name;
    std::string m_surfAreaF_root;



  };
}
#endif
