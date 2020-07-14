#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <Core/Grid/Box.h>

using namespace Uintah;

typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GridInfo::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GridInfo::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &GridInfo::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &GridInfo::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &GridInfo::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GridInfo::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GridInfo::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &GridInfo::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &GridInfo::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &GridInfo::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace GridInfo::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
GridInfo::create_local_labels(){

  register_new_variable<CCVariable<double> >( "gridX" );
  register_new_variable<CCVariable<double> >( "gridY" );
  register_new_variable<CCVariable<double> >( "gridZ" );
  register_new_variable<CCVariable<double> >( "ucellX" );
  register_new_variable<CCVariable<double> >( "vcellY" );
  register_new_variable<CCVariable<double> >( "wcellZ" );

}

//--------------------------------------------------------------------------------------------------
void GridInfo::register_initialize( std::vector<AFC::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( "gridX", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "gridY", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "gridZ", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "ucellX", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "vcellY", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "wcellZ", AFC::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void GridInfo::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto gridX = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "gridX" );
  auto gridY = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "gridY" );
  auto gridZ = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "gridZ" );
  auto ucellX = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "ucellX" );
  auto vcellY = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "vcellY" );
  auto wcellZ = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "wcellZ" );

  Vector Dx = patch->dCell();
  const double dx = Dx.x();
  const double dy = Dx.y();
  const double dz = Dx.z();
  const double dx2 = Dx.x()/2.;
  const double dy2 = Dx.y()/2.;
  const double dz2 = Dx.z()/2.;

  const Level* lvl = patch->getLevel();
  IntVector min; IntVector max;
  lvl->getGrid()->getLevel(0)->findCellIndexRange(min,max);
  IntVector period_bc = IntVector(1,1,1) - lvl->getPeriodicBoundaries();
  Box domainBox = lvl->getBox(min+period_bc, max-period_bc);
  const double lowx = domainBox.lower().x();
  const double lowy = domainBox.lower().y();
  const double lowz = domainBox.lower().z();

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    gridX(i,j,k) = lowx + i * dx + dx2;
    gridY(i,j,k) = lowy + j * dy + dy2;
    gridZ(i,j,k) = lowz + k * dz + dz2;

    ucellX(i,j,k) = gridX(i,j,k) - dx2;
    vcellY(i,j,k) = gridY(i,j,k) - dy2;
    wcellZ(i,j,k) = gridZ(i,j,k) - dz2;
  });
}

//--------------------------------------------------------------------------------------------------
void
GridInfo::register_timestep_init( std::vector<AFC::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( "gridX" , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "gridY" , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "gridZ" , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "ucellX" , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "vcellY" , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "wcellZ" , AFC::COMPUTES, variable_registry, m_task_name );

  register_variable( "gridX" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry, m_task_name );
  register_variable( "gridY" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry, m_task_name );
  register_variable( "gridZ" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry, m_task_name );
  register_variable( "ucellX" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry, m_task_name );
  register_variable( "vcellY" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry, m_task_name );
  register_variable( "wcellZ" , AFC::REQUIRES , 0 , AFC::OLDDW , variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
GridInfo::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto gridX = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "gridX" );
  auto gridY = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "gridY" );
  auto gridZ = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "gridZ" );
  auto ucellX = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "ucellX" );
  auto vcellY = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "vcellY" );
  auto wcellZ = tsk_info->get_field<CCVariable<double>, double, MemSpace>( "wcellZ" );

  auto old_gridX = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( "gridX" );
  auto old_gridY = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( "gridY" );
  auto old_gridZ = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( "gridZ" );
  auto old_ucellX = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( "ucellX" );
  auto old_vcellY = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( "vcellY" );
  auto old_wcellZ = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( "wcellZ" );

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    gridX(i,j,k) = old_gridX(i,j,k);
    gridY(i,j,k) = old_gridY(i,j,k);
    gridZ(i,j,k) = old_gridZ(i,j,k);
    ucellX(i,j,k) = old_ucellX(i,j,k);
    vcellY(i,j,k) = old_vcellY(i,j,k);
    wcellZ(i,j,k) = old_wcellZ(i,j,k);
  });
}
