#ifndef Uintah_Component_Arches_WallConstSmag_h
#define Uintah_Component_Arches_WallConstSmag_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class WallConstSmag : public TaskInterface {

public:

    WallConstSmag( std::string task_name, int matl_index, const ProblemSpecP db_turb_parent );
    ~WallConstSmag();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, const ProblemSpecP db_turb_parent )
        : m_task_name(task_name), m_matl_index(matl_index), m_db_turb_parent(db_turb_parent){}
      ~Builder(){}

      WallConstSmag* build()
      { return scinew WallConstSmag( m_task_name, m_matl_index, m_db_turb_parent ); }

      private:


      std::string m_task_name;
      int m_matl_index;

      const ProblemSpecP m_db_turb_parent;

    };

private:

    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;

    std::string m_IsI_name;
    std::string m_density_name;
    double m_Cs; //Wall constant
    int m_standoff; //Wall constant
    double m_molecular_visc;
    std::string m_volFraction_name{"volFraction"};
    std::vector<std::string> m_sigma_t_names;

    int Nghost_cells;

    const ProblemSpecP m_db_turb_parent; 

  };
}
#endif
