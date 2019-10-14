#ifndef Uintah_Component_Arches_WallHFVariable_h
#define Uintah_Component_Arches_WallHFVariable_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/MaterialManager.h>

namespace Uintah{

  class WallHFVariable : public TaskInterface {

public:

    WallHFVariable( std::string task_name, int matl_index, MaterialManagerP materialManager );
    ~WallHFVariable();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_restart_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();


    //Build instructions for this (WallHFVariable) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, MaterialManagerP materialManager )
        : m_task_name(task_name), m_matl_index(matl_index), _materialManager(materialManager){}
      ~Builder(){}

      WallHFVariable* build()
      { return scinew WallHFVariable( m_task_name, m_matl_index, _materialManager ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      MaterialManagerP _materialManager;

    };

private:

    std::string _flux_x;
    std::string _flux_y;
    std::string _flux_z;
    std::string _net_power;
    std::string _area;

    double _eps;

    int _f;

    MaterialManagerP _materialManager;

    bool _new_variables;

  };
}
#endif
