#include <CCA/Components/Arches/ParticleModels/TotNumDensity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
TotNumDensity::TotNumDensity( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace TotNumDensity::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace TotNumDensity::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &TotNumDensity::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &TotNumDensity::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &TotNumDensity::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace TotNumDensity::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &TotNumDensity::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &TotNumDensity::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &TotNumDensity::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace TotNumDensity::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace TotNumDensity::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::problemSetup( ProblemSpecP& db ){

  bool doing_dqmom = ArchesCore::check_for_particle_method(db,ArchesCore::DQMOM_METHOD);
  bool doing_cqmom = ArchesCore::check_for_particle_method(db,ArchesCore::CQMOM_METHOD);

  if ( doing_dqmom ){
    _Nenv = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );
  } else if ( doing_cqmom ){
    _Nenv = ArchesCore::get_num_env( db, ArchesCore::CQMOM_METHOD );
  } else {
    throw ProblemSetupException(
      "Error: This method only working for DQMOM/CQMOM.",__FILE__,__LINE__);
  }

}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_task_name );

}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool pack_tasks ){

  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){
    const std::string weight_name  = ArchesCore::append_env( "w", ienv);
    register_variable(
      weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void TotNumDensity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& num_den = tsk_info->get_field<CCVariable<double> >( m_task_name );
  num_den.initialize(0.0);

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){


    const std::string weight_name = ArchesCore::append_env( "w", ienv);
    constCCVariable<double>& weight = tsk_info->get_field<constCCVariable<double> >( weight_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){

      num_den(i,j,k) += weight(i,j,k);

    });
  }
}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){
    const std::string weight_name  = ArchesCore::append_env( "w", ienv);
    register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::NEWDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void TotNumDensity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& num_den = tsk_info->get_field<CCVariable<double> >( m_task_name );
  num_den.initialize(0.0);

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){


    const std::string weight_name = ArchesCore::append_env( "w", ienv);
    constCCVariable<double>& weight = tsk_info->get_field<constCCVariable<double> >( weight_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){

      num_den(i,j,k) += weight(i,j,k);

    });
  }
}
} //namespace Uintah
