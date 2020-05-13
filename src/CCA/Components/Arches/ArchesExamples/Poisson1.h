#ifndef Uintah_Component_Arches_ARCHESEXAMPLES_Poisson1_h
#define Uintah_Component_Arches_ARCHESEXAMPLES_Poisson1_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Parallel/MasterLock.h>

namespace Uintah{
  namespace ArchesExamples{
  class Poisson1 : public TaskInterface {

  public:

    Poisson1( std::string task_name, int matl_index ) : TaskInterface( task_name,
      matl_index){};
    ~Poisson1(){};

    typedef std::vector<ArchesFieldContainer::VariableInformation> ArchesVIVector;

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( ArchesVIVector& variable_registry , const bool packed_tasks);

    void register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks){};

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep, const bool packed_tasks );

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep, const bool packed_tasks ){};

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();

    //Build instructions for this (KScalarRHS) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
      : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      Poisson1* build()
      { return scinew Poisson1( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

  private:

    std::vector<std::string> m_var_names;


  }; // class Poisson1
  }//end namespace ArchesExamples
} //end namespace Uintah

#endif
