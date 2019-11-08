#include <CCA/Components/Arches/Task/SampleTask.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
SampleTask::SampleTask( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
SampleTask::~SampleTask(){
}

//--------------------------------------------------------------------------------------------------
//  loadTask<task>FunctionPointers is used to indicate the build(s) supported by a given Arches task
//
//  For empty tasks, use the below:
//
//    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
//
//  For non-empty tasks, use the below with unsupported tags commented out:
//
//    return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
//                                       , &SampleTask::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
//                                       , &SampleTask::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
//                                       //, &SampleTask::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
//                                       );
//
//  * Tag all non-empty tasks with UINTAH_CPU_TAG
//
//  * Tag non-empty tasks refactored to support Kokkos::OpenMP builds with KOKKOS_OPENMP_TAG
//    - e.g. Thread-safe tasks using Uintah::parallel_<pattern>
//
//  * Tag non-empty tasks refactored to support Kokkos::Cuda builds with KOKKOS_CUDA_TAG
//    - e.g. Thread-safe tasks using Uintah::parallel_<pattern> that use only C/C++
//           functionality support by CUDA
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskComputeBCsFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::BC>( this
                                     , &SampleTask::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &SampleTask::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SampleTask::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &SampleTask::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SampleTask::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SampleTask::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &SampleTask::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SampleTask::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SampleTask::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &SampleTask::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SampleTask::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SampleTask::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::problemSetup( ProblemSpecP& db ){

  _value = 1.0;
  //db->findBlock("sample_task")->getAttribute("value",_value);

}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_timestep_init(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks ){
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void SampleTask::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks ){

  typedef ArchesFieldContainer AFC;

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_field", AFC::MODIFIES, variable_registry, m_task_name );
  register_variable( "a_result_field", AFC::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void SampleTask::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  //CCVariable<double>& field = tsk_info->get_field<CCVariable<double> >( "a_sample_field" );
  //CCVariable<double>& result = tsk_info->get_field<CCVariable<double> >( "a_result_field" );

  //constCCVariable<double>& field = tsk_info->get_field<constCCVariable<double> >("a_sample_field");
  CCVariable<double>& field = tsk_info->get_field<CCVariable<double> >("a_sample_field");
  CCVariable<double>& result = tsk_info->get_field<CCVariable<double> >("a_result_field");

  //traditional functor:
  struct mySpecialOper{
    //constructor
    mySpecialOper( CCVariable<double>& var ) : m_var(var){}
    //operator
    void
    operator()(int i, int j, int k) const {

      m_var(i,j,k) = 2.0;

    }
    private:
    CCVariable<double>& m_var;
  };

  mySpecialOper actual_oper(result);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  Uintah::parallel_for( range, actual_oper );

  // lambda style
  Uintah::parallel_for( range, [&](int i, int j, int k){
    field(i,j,k) = 1.1;
    result(i,j,k) = 2.1;
  });

}

//--------------------------------------------------------------------------------------------------
//Register all variables both local and those needed from elsewhere that are required for this task.
void
SampleTask::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  typedef ArchesFieldContainer AFC;

  //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
  register_variable( "a_sample_field", AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "a_result_field", AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "density",        AFC::REQUIRES, 1, AFC::LATEST, variable_registry, time_substep, m_task_name );

  register_variable( "A", AFC::COMPUTES, variable_registry, time_substep, m_task_name );

}

//--------------------------------------------------------------------------------------------------
//This is the work for the task.  First, get the variables. Second, do the work!
template <typename ExecSpace, typename MemSpace>
void SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& field = tsk_info->get_field<CCVariable<double> >( "a_sample_field" );
  CCVariable<double>& result = tsk_info->get_field<CCVariable<double> >( "a_result_field" );
  CCVariable<double>& density = tsk_info->get_field<CCVariable<double> >( "density" );

  //Three ways to get variables:
  // Note that there are 'const' versions of these access calls to tsk_info as well. Just use a
  // tsk_info->get_const_*
  // By reference
  //CCVariable<double>& A_ref = tsk_info->get_field<CCVariable<double> >("A");
  // Pointer
  //CCVariable<double>* A_ptr = tsk_info->get_uintah_field<CCVariable<double> >("A");
  // Traditional Uintah Style
  // But, in this case you lose some of the convenient feature of the Arches Task Interface
  // which may or may not be important to you.
  //CCVariable<double> A_trad;

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    field(i,j,k) = _value * ( density(i,j,k));
    result(i,j,k) = field(i,j,k)*field(i,j,k);
  });
}
