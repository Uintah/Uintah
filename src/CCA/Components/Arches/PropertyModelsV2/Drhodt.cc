#include <CCA/Components/Arches/PropertyModelsV2/Drhodt.h>
#include <CCA/Components/Arches/KokkosTools.h>
#include <CCA/Components/Arches/UPSHelper.h>


namespace Uintah{

//--------------------------------------------------------------------------------------------------
Drhodt::Drhodt( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
Drhodt::~Drhodt(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Drhodt::loadTaskComputeBCsFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::BC>( this
                                     , &Drhodt::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &Drhodt::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &Drhodt::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Drhodt::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &Drhodt::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &Drhodt::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &Drhodt::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Drhodt::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &Drhodt::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &Drhodt::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &Drhodt::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Drhodt::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &Drhodt::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &Drhodt::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &Drhodt::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::OpenMP builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Drhodt::loadTaskRestartInitFunctionPointers()
{
  return  TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
Drhodt::problemSetup( ProblemSpecP& db ){

  m_label_drhodt = "drhodt";
  using namespace ArchesCore;
  m_label_density = parse_ups_for_role( DENSITY, db, "density" );

}

//--------------------------------------------------------------------------------------------------
void
Drhodt::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_drhodt );

}

//--------------------------------------------------------------------------------------------------
void
Drhodt::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_drhodt , ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void Drhodt::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  auto drhodt = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >( m_label_drhodt );
  parallel_initialize(exObj,0.0,drhodt);

}

//--------------------------------------------------------------------------------------------------
void
Drhodt::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  register_variable( m_label_drhodt , ArchesFieldContainer::COMPUTES,  variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void Drhodt::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  auto rho = tsk_info->get_const_uintah_field_add<constCCVariable<double>, const double, MemSpace >( m_label_density );
  auto old_rho = tsk_info->get_const_uintah_field_add<constCCVariable<double>, const double, MemSpace >( m_label_density, ArchesFieldContainer::OLDDW);

  auto drhodt = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >( m_label_drhodt );
  parallel_initialize(exObj,0.0,drhodt);
  const double dt = tsk_info->get_dt();
  //Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for(exObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    drhodt(i,j,k)   = (rho(i,j,k) - old_rho(i,j,k))/dt;
  });
}
//--------------------------------------------------------------------------------------------------

} //namespace Uintah
