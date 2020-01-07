#ifndef Uintah_Component_Arches_ConstantStateProperties_h
#define Uintah_Component_Arches_ConstantStateProperties_h
#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class ConstantStateProperties : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    ConstantStateProperties( std::string task_name, int matl_index );
    ~ConstantStateProperties();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    void register_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    //Build instructions for this (ConstantStateProperties) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      ConstantStateProperties* build()
      { return scinew ConstantStateProperties( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    std::map<std::string, double> m_name_to_value;

  };
}
#endif
