#ifndef Uintah_Component_Arches_ARCHESEXAMPLES_Poisson1_h
#define Uintah_Component_Arches_ARCHESEXAMPLES_Poisson1_h

/*
 * Steps to create a new example using Arches
 * 1. Create a .h and .cc files (and classes) based on ArchesExamples/Poisson1.h and .cc.
 * 2. Update functions problemSetup, create_local_labels, loadTask*, register_* and
 *    actual functions doing the work as needed. Good place to start is to use following
 *    mapping from this example:
 *    CCA/Components/Examples/Poisson1.cc   CCA/Components/Arches/ArchesExamples/Poisson1.cc
 *    Poisson1                              Poisson1 or create_local_labels
 *    problemSetup                          problemSetup
 *    initialize                            initialize
 *    timeAdvance                           eval
 *    computeStableTimeStep                 Not needed. Arches will take care of it.
 *    scheduleInitialize                    loadTaskTimestepInitFunctionPointers and register_initialize
 *    scheduleTimeAdvance                   loadTaskEvalFunctionPointers and register_timestep_eval
 *    scheduleComputeStableTimeStep         Not needed. Arches will take care of it.
 *    Optionally compute_bcs and timestep_init can be used along with their load and register
 *    functions to process initialization during every timestep or boundary conditions.
 *    This can simplify eval
 * 3. Update ExampleFactory.h and .cc. Check ExampleFactory.h for more comments.
 * 4. Once CPU only version works, port to OpenMP and GPU. Check the porting steps in:
 *    CCA/Components/Arches/SampleTask.cc
 */


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

    void register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks);

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep, const bool packed_tasks );

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep, const bool packed_tasks );

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

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



  }; // class Poisson1
  }//end namespace ArchesExamples
} //end namespace Uintah

#endif
