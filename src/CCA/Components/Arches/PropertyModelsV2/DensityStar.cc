#include <CCA/Components/Arches/PropertyModelsV2/DensityStar.h>
#include <CCA/Components/Arches/UPSHelper.h>


namespace Uintah{

//--------------------------------------------------------------------------------------------------
DensityStar::DensityStar( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
DensityStar::~DensityStar(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityStar::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityStar::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &DensityStar::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityStar::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityStar::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityStar::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DensityStar::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityStar::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityStar::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityStar::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &DensityStar::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityStar::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityStar::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityStar::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
DensityStar::problemSetup( ProblemSpecP& db ){

  using namespace ArchesCore;
  m_label_density = parse_ups_for_role( DENSITY_ROLE, db, "density" );
  m_label_densityStar = m_label_density + "_star" ;

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_densityStar );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DensityStar::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto rhoStar = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_densityStar );
  parallel_initialize(execObj,0.0,rhoStar);

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  //register_variable( m_label_densityStar , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::OLDDW, variable_registry, m_task_name );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::OLDDW, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
DensityStar::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto rhoStar = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_densityStar );
  auto old_rho = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( m_label_density );

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for(execObj,range, KOKKOS_LAMBDA(int i, int j, int k){
    rhoStar(i,j,k)=old_rho(i,j,k);
  });

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( ArchesCore::default_uMom_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( ArchesCore::default_vMom_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( ArchesCore::default_wMom_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_label_density , ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

  register_variable( m_label_densityStar, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DensityStar::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto xmom = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(ArchesCore::default_uMom_name);
  auto ymom = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(ArchesCore::default_vMom_name);
  auto zmom = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(ArchesCore::default_wMom_name);

  auto rho = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_density );
  auto rhoStar = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_densityStar );

  const double dt = tsk_info->get_dt();

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double vol       = DX.x()*DX.y()*DX.z();

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

  double check_guess_density_out = 0;
  Uintah::parallel_reduce_sum( execObj, range, KOKKOS_LAMBDA (int i, int j, int k, double& check_guess_density)
  {
	rhoStar(i,j,k)   = rho(i,j,k) - ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
									  area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) )+
									  area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) )) * dt / vol;
    check_guess_density += (rhoStar(i,j,k) < 0);

  }, check_guess_density_out);

  if (check_guess_density_out > 0){
    std::cout << "NOTICE: Negative density guess(es) occurred. Reverting to old density."<< std::endl ;
  } else {
    Uintah::parallel_for(execObj,range, KOKKOS_LAMBDA(int i, int j, int k){
      rho(i,j,k)  = rhoStar(i,j,k); // I am copy density guess in density
    });
  }

}

} //namespace Uintah
