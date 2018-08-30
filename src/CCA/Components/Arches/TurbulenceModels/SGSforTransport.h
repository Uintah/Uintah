#ifndef Uintah_Component_Arches_SGSforTransport_h
#define Uintah_Component_Arches_SGSforTransport_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

namespace Uintah{

  class SGSforTransport : public TaskInterface {

    public:

      SGSforTransport( std::string task_name, int matl_index );
      ~SGSforTransport();

      TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

      TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

      TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

      TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();
    
      TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

      void problemSetup( ProblemSpecP& db );

      void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

      void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

      void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

      void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

      template <typename ExecutionSpace, typename MemorySpace>
      void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemorySpace>& executionObject ){}

      template <typename ExecutionSpace, typename MemorySpace>
      void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemorySpace>& executionObject );

      template<typename ExecutionSpace, typename MemSpace> void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace,MemSpace>& exObj);

      template <typename ExecutionSpace, typename MemorySpace>
      void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemorySpace>& executionObject );

      void create_local_labels();

      //Build instructions for this (SGSforTransport) class.
      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
          ~Builder(){}

          SGSforTransport* build()
          { return scinew SGSforTransport( _task_name, _matl_index ); }

        private:

          std::string _task_name;
          int _matl_index;
      };

    private:
      std::string m_u_vel_name;
      std::string m_v_vel_name;
      std::string m_w_vel_name;
      std::string m_density_name;

      std::string m_cc_u_vel_name;
      std::string m_cc_v_vel_name;
      std::string m_cc_w_vel_name;
      std::vector<std::string> m_SgsStress_names;
      std::vector<std::string> m_fmom_source_names;

      std::string m_rhou_vel_name;
      std::string m_rhov_vel_name;
      std::string m_rhow_vel_name;
      //int Type_filter ; 
      Uintah::FILTER Type_filter; 
  };
}
#endif
