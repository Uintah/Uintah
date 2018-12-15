#ifndef Uintah_Component_Arches_AddPressGradient_h
#define Uintah_Component_Arches_AddPressGradient_h

#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>

//===============================================================

/**
* @class  AddPressGradient
* @author Jeremy Thornock
* @date   2016
*
* @brief Adds pressure gradient to the RHS of the momentum eqn
*        This should be done *post* projection.
*
**/

//===============================================================

namespace Uintah{

  class AddPressGradient : public AtomicTaskInterface {

public:

    /** @brief Default constructor **/
    AddPressGradient( std::string task_name, int matl_index );

    /** @brief Default destructor **/
    ~AddPressGradient();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    /** @brief Create local labels for the task **/
    void create_local_labels();

    /** @brief Registers all variables with pertinent information for the
     *         uintah dw interface **/
    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
                        const int time_substep, const bool pack_tasks );

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    /** @brief Builder class containing instructions on how to build the task **/
    class Builder : public AtomicTaskInterface::AtomicTaskBuilder {

      public:

        Builder(std::string name, int matl_index):
        m_task_name(name), m_matl_index(matl_index){};

        ~Builder() {}

        AddPressGradient* build(){ return scinew AddPressGradient(m_task_name, m_matl_index);}

      protected:

        std::string m_task_name;
        int m_matl_index;

    };

private:

    std::string m_xmom;
    std::string m_ymom;
    std::string m_zmom;
    std::string m_press;
    std::string m_eps_name;


  };
} // namespace Uintah

#endif
