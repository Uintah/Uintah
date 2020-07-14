#ifndef Uintah_Component_Arches_DQMOMNoInversion_h
#define Uintah_Component_Arches_DQMOMNoInversion_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
//-------------------------------------------------------

/**
 * @class    DQMOMNoInversion
 * @author   Jeremy Thornock
 * @date     Feb 2018
 *
 * @brief    Construct source term for DQMOM transport
 *
 * @details  Constructs the source terms for DQMOM transport
 *           where the inversion has been factored out.
 *
 */

//-------------------------------------------------------

namespace Uintah{

  class DQMOMNoInversion : public TaskInterface {

  public:

    DQMOMNoInversion( std::string task_name, int matl_index, const int N );
    ~DQMOMNoInversion();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index, const int N ) :
      m_task_name(task_name), m_matl_index(matl_index), m_N(N){}
      ~Builder(){}

      DQMOMNoInversion* build()
      { return scinew DQMOMNoInversion( m_task_name, m_matl_index, m_N ); }

    private:

      std::string m_task_name;
      int m_matl_index;
      int m_N;

    };

//  protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

  private:

    const int m_N;                              ///< The number of "environments"
    std::vector<std::string> m_ic_names;        ///< Internal coordinate names
    std::vector<std::string> m_ic_qn_srcnames;  ///< ICname + _qn# + _src, which gets put into the transport eqn
    std::vector<std::string> m_ic_qn_rhsnames;  ///< ICname + _qn# + _RHS 
    std::map<std::string, std::vector<std::string> > m_ic_model_map;   ///< A map of ic names to models

  };
}

#endif
