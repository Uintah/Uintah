#include <CCA/Components/Arches/Task/SampleTask.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
//  Arches Task Porting Overview
//--------------------------------------------------------------------------------------------------
//
//  DISCLAIMER: Arches tasks are in varying states of portability with some using deprecated infrastructure.
//              Please verify use of the latest portable infrastructure (e.g., 04, 05) when porting.
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
//  (12) Convert private and protected functions using Uintah::parallel_<pattern> calls to public.
//
//       * More on extended lambda restrictions can be found at https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda-restrictions
//
//  (13) Copy class member variables used in Uintah::parallel_<pattern> calls into temporary variables declared outside of calls.
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
SampleTask::SampleTask( std::string task_name, int matl_index )
  : TaskInterface( task_name, matl_index )
{
}

//--------------------------------------------------------------------------------------------------
SampleTask::~SampleTask()
{
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
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &SampleTask::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &SampleTask::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SampleTask::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &SampleTask::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &SampleTask::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SampleTask::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SampleTask::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::problemSetup( ProblemSpecP& db )
{
  _value = 1.0;
  //db->findBlock("sample_task")->getAttribute("value",_value);
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks )
{
  // Register all data warehouse variables used in task SampleTask::initialize
  register_variable( "a_sample_field", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  register_variable( "a_result_field", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

  // NOTES:
  // * Pass underlying strings into register_variable where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
  // * Supported parameter lists can be found in src/CCA/Components/Arches/Task/TaskVariableTools.cc
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void
SampleTask::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj )
{
  // Get all data warehouse variables used in SampleTask::initialize
  auto field  = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "a_sample_field" );
  auto result = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "a_result_field" );

  // Initialize data warehouse variables
  Uintah::parallel_initialize( execObj, 1.1, field );
  Uintah::parallel_initialize( execObj, 2.1, result );

  // NOTES:
  // * Portable get_field calls require 3 template parameters: (1) legacy Uintah type, (2) underlying plain-old-data type, and (3) memory space
  // * Pass underlying strings into get_field where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
}

//--------------------------------------------------------------------------------------------------
void
SampleTask::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep, const bool packed_tasks )
{
  // Register all data warehouse variables used in SampleTask::eval
  register_variable( "a_sample_field", ArchesFieldContainer::COMPUTES, /* Ghost Cell Quantity, Data Warehouse, */            variable_registry, time_substep, m_task_name );
  register_variable( "a_result_field", ArchesFieldContainer::COMPUTES, /* Ghost Cell Quantity, Data Warehouse, */            variable_registry, time_substep, m_task_name );
  register_variable( "density",        ArchesFieldContainer::REQUIRES, 1,                      ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );

  // NOTES:
  // * Pass underlying strings into register_variable where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
  // * Supported parameter lists can be found in src/CCA/Components/Arches/Task/TaskVariableTools.cc
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void
SampleTask::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj )
{
  // Get all data warehouse variables used in SampleTask::eval
  auto field   = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "a_sample_field" );
  auto result  = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "a_result_field" );
  auto density = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "density" );

  // Setup the range of cells to iterate over
  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  // Setup the loop that iterates over cells
  Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA( int i, int j, int k ){
    field(i,j,k)  = _value * density(i,j,k);
    result(i,j,k) = field(i,j,k) * field(i,j,k);
  });

  // NOTES:
  // * Portable get_field calls require 3 template parameters: (1) legacy Uintah type, (2) underlying plain-old-data type, and (3) memory space
  // * Pass underlying strings into get_field where possible to improve searchability (e.g., "a_sample_field")
  // * Uintah infrastructure uses underlying strings for debugging output and exceptions
  // * Portable Uintah::parallel_for calls pass execObj and are executed using the supported back-end(s)
}
