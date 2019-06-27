#ifndef Uintah_Component_Arches_MMS_scalar_h
#define Uintah_Component_Arches_MMS_scalar_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/MaterialManager.h>

namespace Uintah{

  class MMS_scalar : public TaskInterface {

public:

    enum WAVE_TYPE { SINE, SINE_T, GCOSINE, T1, T2, T3 };

    MMS_scalar( std::string task_name, int matl_index, MaterialManagerP materialManager  );
    ~MMS_scalar();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry  , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry  , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep  , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep  , const bool packed_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();
          
    void MMS_SINE( double&, double&, double&, const double&, double );
    void MMS_SINE_T( double&, double&, double&, double&,const double&, double );
    void MMS_T1( double&, double&, double );
    void MMS_T2( double&, double&, double );
    void MMS_GCOSINE( double&, double&,  const double&, double );

    //Build instructions for this (MMS_scalar) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, MaterialManagerP materialManager ) : m_task_name(task_name), m_matl_index(matl_index), _materialManager(materialManager){}
      ~Builder(){}

      MMS_scalar* build()
      { return scinew MMS_scalar( m_task_name, m_matl_index, _materialManager  ); }

      private:

      std::string m_task_name;
      int m_matl_index;

      MaterialManagerP _materialManager;
    };

private:
    double A ;
    double F ;
    double offset ;
    double pi = acos(-1.0);
    double sigma;
    std::string ind_var_name;

    WAVE_TYPE _wtype;

    std::string m_MMS_label;
    std::string m_MMS_source_label;
    std::string m_MMS_source_diff_label;
    std::string m_MMS_source_t_label;

    MaterialManagerP _materialManager;

  };
}
#endif
