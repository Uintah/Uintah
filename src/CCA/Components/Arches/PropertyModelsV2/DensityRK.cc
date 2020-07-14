#include <CCA/Components/Arches/PropertyModelsV2/DensityRK.h>
#include <CCA/Components/Arches/UPSHelper.h>


namespace Uintah{

//--------------------------------------------------------------------------------------------------
DensityRK::DensityRK( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
DensityRK::~DensityRK(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityRK::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityRK::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &DensityRK::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityRK::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityRK::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityRK::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DensityRK::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DensityRK::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DensityRK::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityRK::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityRK::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
DensityRK::problemSetup( ProblemSpecP& db ){

  using namespace ArchesCore;
  m_label_density = parse_ups_for_role( DENSITY_ROLE, db, "density" );
  m_label_densityRK = m_label_density + "_rk" ;
  ProblemSpecP db_root = db->getRootNode();

  db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->getAttribute("order", _time_order);

  if ( _time_order == 1 ){

    _alpha.resize(1);
    _beta.resize(1);
    _time_factor.resize(1);

    _alpha[0] = 0.0;

    _beta[0]  = 1.0;

    _time_factor[0] = 1.0;

  } else if ( _time_order == 2 ) {

    _alpha.resize(2);
    _beta.resize(2);
    _time_factor.resize(2);

    _alpha[0]= 0.0;
    _alpha[1]= 0.5;

    _beta[0]  = 1.0;
    _beta[1]  = 0.5;

    _time_factor[0] = 1.0;
    _time_factor[1] = 1.0;

  } else if ( _time_order == 3 ) {

    _alpha.resize(3);
    _beta.resize(3);
    _time_factor.resize(3);

    _alpha[0] = 0.0;
    _alpha[1] = 0.75;
    _alpha[2] = 1.0/3.0;

    _beta[0]  = 1.0;
    _beta[1]  = 0.25;
    _beta[2]  = 2.0/3.0;

    _time_factor[0] = 1.0;
    _time_factor[1] = 0.5;
    _time_factor[2] = 1.0;

  } else {
    throw InvalidValue("Error: <TimeIntegrator> must have value: 1, 2, or 3 (representing the order).",__FILE__, __LINE__);
  }

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_densityRK );

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityRK , ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DensityRK::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto rhoRK = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_densityRK );
  parallel_initialize(execObj,0.0,rhoRK);

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( m_label_density , ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );

  register_variable( m_label_densityRK, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DensityRK::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto old_rho = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( m_label_density, ArchesFieldContainer::OLDDW );
  auto rho = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_density );
  auto rhoRK = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_label_densityRK );

  const int time_substep = tsk_info->get_time_substep();

  const double alpha =_alpha[time_substep];
  const double beta = _beta[time_substep];

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for(execObj,range, KOKKOS_LAMBDA(int i, int j, int k){

    rhoRK(i,j,k) = alpha * old_rho(i,j,k) +  beta * rho(i,j,k);

    rho(i,j,k)  = rhoRK(i,j,k); // I am copy density guess in density

  });
}

} //namespace Uintah
