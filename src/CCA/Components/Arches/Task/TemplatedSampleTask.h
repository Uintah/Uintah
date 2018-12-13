#ifndef Uintah_Component_Arches_TemplatedSampleTask_h
#define Uintah_Component_Arches_TemplatedSampleTask_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  template <typename T>
  class TemplatedSampleTask : public TaskInterface {

public:

    TemplatedSampleTask<T>( std::string task_name, int matl_index );
    ~TemplatedSampleTask<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (TemplatedSampleTask) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      TemplatedSampleTask* build()
      { return scinew TemplatedSampleTask<T>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks ){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    template <typename ExecutionSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& execObj );

    template <typename ExecutionSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& execObj );

    template <typename ExecutionSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace,MemSpace>& execObj ){}

    template <typename ExecutionSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& execObj );

    void create_local_labels();

private:

  };

  //Function definitions:

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TemplatedSampleTask<T>::TemplatedSampleTask( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TemplatedSampleTask<T>::~TemplatedSampleTask()
  {
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TemplatedSampleTask<T>::loadTaskComputeBCsFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::BC>( this
                                       , &TemplatedSampleTask<T>::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &TemplatedSampleTask<T>::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &TemplatedSampleTask<T>::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TemplatedSampleTask<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &TemplatedSampleTask<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &TemplatedSampleTask<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &TemplatedSampleTask<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TemplatedSampleTask<T>::loadTaskEvalFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                       , &TemplatedSampleTask<T>::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &TemplatedSampleTask<T>::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &TemplatedSampleTask<T>::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TemplatedSampleTask<T>::loadTaskTimestepInitFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                       , &TemplatedSampleTask<T>::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &TemplatedSampleTask<T>::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &TemplatedSampleTask<T>::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::OpenMP builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TemplatedSampleTask<T>::loadTaskRestartInitFunctionPointers()
  {
    return  TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  void TemplatedSampleTask<T>::problemSetup( ProblemSpecP& db ){
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  void TemplatedSampleTask<T>::create_local_labels(){

    register_new_variable<T>( "templated_variable" );

  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  void TemplatedSampleTask<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
                                                    const bool packed_tasks ){

    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

  }

  //--------------------------------------------------------------------------------------------------
  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  template<typename ExecutionSpace, typename MemSpace>
  void TemplatedSampleTask<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& execObj ){

    T& field = *(tsk_info->get_uintah_field<T>( "templated_variable" ));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      field(i,j,k) = 3.2;
    });

  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  void TemplatedSampleTask<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  template<typename ExecutionSpace, typename MemSpace>
  void TemplatedSampleTask<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& execObj ){

    T& field = *(tsk_info->get_uintah_field<T>( "templated_variable" ));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      field(i,j,k) = 23.4;
    });

  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  void TemplatedSampleTask<T>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  template<typename ExecutionSpace, typename MemSpace>
  void TemplatedSampleTask<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& execObj ){}

}
#endif
