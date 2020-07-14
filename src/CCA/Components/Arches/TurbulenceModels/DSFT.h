#ifndef Uintah_Component_Arches_DSFT_h
#define Uintah_Component_Arches_DSFT_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

namespace Uintah{

  class DSFT : public TaskInterface {

public:

    DSFT( std::string task_name, int matl_index, const std::string turb_model_name );
    ~DSFT();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks){}

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

    //Build instructions for this (DSFT) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, const std::string turb_model_name )
        : m_task_name(task_name), m_matl_index(matl_index), m_turb_model_name(turb_model_name){}
      ~Builder(){}

      DSFT* build()
      { return scinew DSFT( m_task_name, m_matl_index, m_turb_model_name ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      const std::string m_turb_model_name;

    };

    template <typename ExecSpace, typename MemSpace>
    void computeModel( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

private:

    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;
    std::string m_density_name;
    std::string m_volFraction_name;

    std::string m_cc_u_vel_name;
    std::string m_cc_v_vel_name;
    std::string m_cc_w_vel_name;

    std::string m_rhou_vel_name;
    std::string m_rhov_vel_name;
    std::string m_rhow_vel_name;
    std::string m_IsI_name;

    const std::string m_turb_model_name;
    bool m_create_labels_IsI_t_viscosity{true};
    Uintah::ArchesCore::FILTER Type_filter;
    Uintah::ArchesCore::TestFilter m_Filter;
  };
}
#endif
