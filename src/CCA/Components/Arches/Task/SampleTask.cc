#include <CCA/Components/Arches/Task/SampleTask.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
//  Arches Task Porting Overview
//--------------------------------------------------------------------------------------------------
//
//  (01) Add helper function(s) to enable task tagging, e.g.,:
//
//       TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();
//
//       TaskAssignedExecutionSpace SampleTask::loadTaskEvalFunctionPointers()
//       {
//         return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
//                                            , &SampleTask::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
//                                            //, &SampleTask::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
//                                            //, &SampleTask::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
//                                            );
//       }
//
//  (02) Uncomment tag(s) for target build(s) to support
//
//  (03) Template target task(s) on ExecSpace and MemSpace then add execObj to the parameter list, e.g.,:
//
//       void SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
//
//       Becomes:
//
//       template <typename ExecSpace, typename MemSpace>
//       void SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
//
//  (04) Replace type-specific get_field data warehouse accessors with auto type get_field data warehouse accessors, e.g.,:
//
//       constCCVariable<double>& CCuVel = tsk_info->get_field<constCCVariable<double>>( m_cc_u_vel_name );
//
//       Becomes:
//
//       auto CCuVel = tsk_info->get_field<CT, const double, MemSpace>( m_cc_u_vel_name );
//
//       * Portable get_field calls require 3 template parameters: (1) legacy Uintah type, (2) underlying plain-old-data type, and (3) memory space
//
//  (05) Replace initialize with Uintah::parallel_initialize, e.g.,:
//
//       CCuVel.initialize(0.0);
//       CCvVel.initialize(0.0);
//       CCwVel.initialize(0.0);
//
//       Becomes:
//
//       Uintah::parallel_initialize( execObj, 0.0, CCuVel, CCvVel, CCwVel );
//
//       * Variables must be of the same type when passing multiple variables into a single Uintah::parallel_initialize
//       * Alternatively, initialize variables in the subsequent Uintah::parallel_<pattern> instead of a standalone Uintah::parallel_initialize
//
//  (06) Replace vectors with fixed-size arrays, e.g.,:
//
//       std::vector< CT* > species;
//
//       for ( int ns = 0; ns < _NUM_species; ns++ ) {
//         CT* species_p = &(tsk_info->get_field< CT >( _species_names[ns] ));
//         species.push_back( species_p );
//       }
//
//       Becomes:
//
//       auto species = createContainer<CT, const double, max_species_count, MemSpace>(species_count);
//
//       for ( int ns = 0; ns < _NUM_species; ns++ ) {
//         species[ns] = tsk_info->get_field<CT, const double, MemSpace>( _species_names[ns] );
//       }
//
//       * Portable createContainer calls require 4 template parameters: (1) legacy Uintah type, (2) underlying plain-old-data type, (3) maximum size, and (4) memory space
//
//  (07) Specify the range of cells for the Uintah::parallel_<pattern> to iterate over, e.g.,:
//
//       Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
//
//  (08) Replace CellIterator-based loops with Uintah::parallel_<pattern>, e.g.,:
//
//       for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ) { // Loop Body }
//
//       Becomes:
//
//       Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA( int i, int j, int k ){ // Loop Body });
//
//       * Deprecated parallel_for calls are still used throughout Uintah, add execObj to the parameter list to ensure use of the latest
//       * Currently supported Uintah::parallel_<pattern> variants can be found in src/Core/Parallel/LoopExecution.hpp
//
//  (09) Move temporary per-cell variables specific to a each i,j,k inside of loops for thread-safety, e.g.,:
//
//       // NOT thread-safe when each i,j,k needs its own dudx, dudy, dudz
//       // double dudx = 0.0;
//       // double dudy = 0.0;
//       // double dudz = 0.0;
//
//       Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA( int i, int j, int k ){
//         // Thread-safe when each i,j,k needs its own dudx, dudy, dudz
//         double dudx = 0.0;
//         double dudy = 0.0;
//         double dudz = 0.0;
//       });
//
//  (10) Eliminate use of C++ standard library classes and functions that do not have CUDA equivalents, e.g.,:
//
//       Replace std::cout with printf, replace std::string with null-terminated arrays of characters, hard-code std::accumulate, etc
//
//       * A collection of supported C/C++ functionality can be found at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support
//
//  (11) Drop use of std:: before C++ standard library classes and functions that do have CUDA equivalents, e.g.,:
//
//       Replace std::fabs with fabs, replace std::fmax with fmax, replace std::fmin with fmin, etc
//
//       * A collection of C/C++ standard library math functions supported in CUDA device code can be found at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix
//
//--------------------------------------------------------------------------------------------------
//
//  General Suggestions:
//
//  * Avoid hard-coded execution and memory spaces
//  * Avoid use of intermediate variables mapped to underlying strings for portable get_field calls
//  * Keep code within portable loops as simple as possible
//  * Keep formatting and whitespace consistent across tasks for searchability
//  * Port tasks incrementally one-by-one
//  * Search uncommented tags for portable task examples (current best example supporting non-Kokkos, Kokkos::OpenMP, and Kokkos::CUDA builds is src/CCA/Components/Arches/ParticleModels/CharOxidationps.h)
//  * Verify correctness before and after changes for portability across multiple inputs and platforms
//  * Verify execution takes place where expected (e.g., using htop, ps, etc on host, nvpp on device, etc)
//  * When in doubt, port by brute force (e.g., uncomment tag(s), try to build, fix build-breaking portability barriers, repeat)
//
//--------------------------------------------------------------------------------------------------

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
//    - e.g., Thread-safe tasks using Uintah::parallel_<pattern>
//
//  * Tag non-empty tasks refactored to support Kokkos::Cuda builds with KOKKOS_CUDA_TAG
//    - e.g., Thread-safe tasks using Uintah::parallel_<pattern> that use only C/C++ functionality support by CUDA
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
