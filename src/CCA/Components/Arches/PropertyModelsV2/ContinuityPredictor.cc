#include <CCA/Components/Arches/PropertyModelsV2/ContinuityPredictor.h>
#include <CCA/Components/Arches/KokkosTools.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
ContinuityPredictor::ContinuityPredictor( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
ContinuityPredictor::~ContinuityPredictor(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ContinuityPredictor::loadTaskComputeBCsFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::BC>( this
                                     , &ContinuityPredictor::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &ContinuityPredictor::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &ContinuityPredictor::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ContinuityPredictor::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &ContinuityPredictor::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &ContinuityPredictor::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &ContinuityPredictor::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ContinuityPredictor::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &ContinuityPredictor::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ContinuityPredictor::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &ContinuityPredictor::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace ContinuityPredictor::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &ContinuityPredictor::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ContinuityPredictor::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     );
}

TaskAssignedExecutionSpace ContinuityPredictor::loadTaskRestartInitFunctionPointers()
{
  return  TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}
//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::problemSetup( ProblemSpecP& db ){

  m_label_balance = "continuity_balance";

  if (db->findBlock("KMomentum")->findBlock("use_drhodt")){

    db->findBlock("KMomentum")->findBlock("use_drhodt")->getAttribute("label",m_label_drhodt);

  } else {

    m_label_drhodt = "drhodt";

  }

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_balance );

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_balance , ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemorySpace>
void ContinuityPredictor::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemorySpace>& executionObject ){

  CCVariable<double>& Balance = *(tsk_info->get_uintah_field<CCVariable<double> >( m_label_balance ));
  Balance.initialize(0.0);

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace> void
ContinuityPredictor::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& executionObject ){

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( "x-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "y-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "z-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_label_drhodt , ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( m_label_balance , ArchesFieldContainer::COMPUTES, variable_registry, time_substep );


}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemorySpace>
void ContinuityPredictor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemorySpace>& executionObject ){

  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");

  constCCVariable<double>& drho_dt = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_label_drhodt ));
  CCVariable<double>& Balance = *(tsk_info->get_uintah_field<CCVariable<double> >( m_label_balance ));
  Balance.initialize(0.0);
  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double vol     = DX.x()*DX.y()*DX.z();

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    Balance(i,j,k) = vol*drho_dt(i,j,k) + ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                                            area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) )+
                                            area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) ));
  });
}
} //namespace Uintah
