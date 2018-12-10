#include <CCA/Components/Arches/PropertyModelsV2/DensityStar.h>
#include <CCA/Components/Arches/KokkosTools.h>
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
  return create_portable_arches_tasks<TaskInterface::BC>( this
                                     , &DensityStar::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityStar::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityStar::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
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

TaskAssignedExecutionSpace DensityStar::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &DensityStar::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityStar::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityStar::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::OpenMP builds
                                     );
}

TaskAssignedExecutionSpace DensityStar::loadTaskRestartInitFunctionPointers()
{
  return  TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}


//--------------------------------------------------------------------------------------------------
void
DensityStar::problemSetup( ProblemSpecP& db ){

  using namespace ArchesCore;
  m_label_density = parse_ups_for_role( DENSITY, db, "density" );
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

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::NEWDW, variable_registry);

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void DensityStar::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  auto rhoStar = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >( m_label_densityStar );
  auto rho = tsk_info->get_const_uintah_field_add<constCCVariable<double> ,const double, MemSpace>( m_label_density );

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for(exObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    rhoStar(i,j,k)   = rho(i,j,k);
  });

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_label_densityStar , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::OLDDW, variable_registry);

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace> void
DensityStar::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  auto rhoStar = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >( m_label_densityStar );
  auto old_rhoStar = tsk_info->get_const_uintah_field_add<constCCVariable<double>, const double, MemSpace >( m_label_densityStar );

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for(exObj,range, KOKKOS_LAMBDA(int i, int j, int k){
    rhoStar(i,j,k)=old_rhoStar(i,j,k);
  });

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( "x-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "y-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "z-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_label_density , ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

  register_variable( m_label_densityStar, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void DensityStar::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  auto xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double>, const double, MemSpace >("x-mom");
  auto ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double>, const double, MemSpace >("y-mom");
  auto zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double>, const double, MemSpace >("z-mom");

  auto rho = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace  >( m_label_density );
  auto rhoStar = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace  >( m_label_densityStar );

  const double dt = tsk_info->get_dt();

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double vol       = DX.x()*DX.y()*DX.z();

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for(exObj,range, KOKKOS_LAMBDA(int i, int j, int k){

    rhoStar(i,j,k)   = rho(i,j,k) - ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                                      area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) )+
                                      area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) )) * dt / vol;

  });

  double check_guess_density_out = 0;
  Uintah::parallel_reduce_sum( exObj, range, KOKKOS_LAMBDA (int i, int j, int k, double& check_guess_density)
  {
   check_guess_density += (rhoStar(i,j,k) < 0) ;

  }, check_guess_density_out); 

  if (check_guess_density_out > 0){
    std::cout << "NOTICE: Negative density guess(es) occurred. Reverting to old density."<< std::endl ;
  } else {
    Uintah::parallel_for(exObj,range, KOKKOS_LAMBDA(int i, int j, int k){  
      rho(i,j,k)  = rhoStar(i,j,k); // I am copy density guess in density
    });
  }
  
}

} //namespace Uintah
