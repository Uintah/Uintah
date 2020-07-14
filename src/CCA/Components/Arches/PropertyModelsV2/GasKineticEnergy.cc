#include <CCA/Components/Arches/PropertyModelsV2/GasKineticEnergy.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
GasKineticEnergy::GasKineticEnergy( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
GasKineticEnergy::~GasKineticEnergy(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GasKineticEnergy::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GasKineticEnergy::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &GasKineticEnergy::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &GasKineticEnergy::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &GasKineticEnergy::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GasKineticEnergy::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &GasKineticEnergy::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &GasKineticEnergy::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &GasKineticEnergy::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GasKineticEnergy::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GasKineticEnergy::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::problemSetup( ProblemSpecP& db ){

  m_u_vel_name = parse_ups_for_role( Uintah::ArchesCore::CCUVELOCITY_ROLE, db, "CCUVelocity" );
  m_v_vel_name = parse_ups_for_role( Uintah::ArchesCore::CCVVELOCITY_ROLE, db, "CCVVelocity" );
  m_w_vel_name = parse_ups_for_role( Uintah::ArchesCore::CCWVELOCITY_ROLE, db, "CCWVelocity" );
  m_kinetic_energy = "gas_kinetic_energy";
  m_max_ke = 1e9 ;
}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_kinetic_energy );

}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_kinetic_energy , ArchesFieldContainer::COMPUTES, variable_registry , m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void GasKineticEnergy::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto ke = tsk_info->get_field<CCVariable<double>, double, MemSpace >( m_kinetic_energy );
  parallel_initialize(execObj,0.0,ke);

}



//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );

  register_variable( m_kinetic_energy , ArchesFieldContainer::COMPUTES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void GasKineticEnergy::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  // cc gas velocities
  auto u = tsk_info->get_field<constCCVariable<double>,const double, MemSpace >(m_w_vel_name);
  auto v = tsk_info->get_field<constCCVariable<double>,const double, MemSpace >(m_w_vel_name);
  auto w = tsk_info->get_field<constCCVariable<double>,const double, MemSpace >(m_w_vel_name);

  auto ke = tsk_info->get_field<CCVariable<double>, double, MemSpace >( m_kinetic_energy );
  parallel_initialize(execObj,0.0,ke);
  double ke_p = 0;
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  //Uintah::parallel_reduce_min(execObj, range, KOKKOS_LAMBDA (const int i, const int j, const int k, double & m_dt ){
  Uintah::parallel_reduce_sum(execObj, range, KOKKOS_LAMBDA (const int i, const int j, const int k, double& ke_sum){
    ke(i,j,k) = 0.5*(u(i,j,k)*u(i,j,k) + v(i,j,k)*v(i,j,k) +w(i,j,k)*w(i,j,k));
    ke_sum += ke(i,j,k);
  }, ke_p);
  // check if ke is diverging in this patch
  if ( ke_p > m_max_ke )
    throw InvalidValue("Error: KE is diverging.",__FILE__,__LINE__);
}
} //namespace Uintah
