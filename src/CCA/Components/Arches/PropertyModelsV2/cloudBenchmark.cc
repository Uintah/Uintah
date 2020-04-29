#include <CCA/Components/Arches/PropertyModelsV2/cloudBenchmark.h>
#include <ostream>
#include <cmath>
namespace Uintah{

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace cloudBenchmark::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace cloudBenchmark::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &cloudBenchmark::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &cloudBenchmark::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &cloudBenchmark::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace cloudBenchmark::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace cloudBenchmark::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &cloudBenchmark::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &cloudBenchmark::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &cloudBenchmark::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace cloudBenchmark::loadTaskRestartInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::RESTART_INITIALIZE>( this
                                     , &cloudBenchmark::restart_initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &cloudBenchmark::restart_initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &cloudBenchmark::restart_initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
void
cloudBenchmark::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_prop = db;
  db_prop->getWithDefault("min", m_min, m_notSetMin);
  db_prop->getWithDefault("max", m_max, m_notSetMax);

  // bulletproofing  min & max must be set
  if( ( m_min == m_notSetMin && m_max != m_notSetMax) ||
      ( m_min != m_notSetMin && m_max == m_notSetMax) ){
    std::ostringstream warn;
    warn << "\nERROR:<property_calculator type=burns_christon>\n "
         << "You must specify both a min: "<< m_min << " & std point: "<< m_max <<".";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

    db_prop->getAttribute("label",m_abskg_name);

}

//--------------------------------------------------------------------------------------------------
void
cloudBenchmark::create_local_labels(){

    register_new_variable<CCVariable<double> >(m_abskg_name);
    register_new_variable<CCVariable<double> >("temperature");

}

//--------------------------------------------------------------------------------------------------
void
cloudBenchmark::register_initialize( VIVec& variable_registry , const bool pack_tasks){

  register_variable( m_abskg_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "temperature", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "gridX", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "gridY", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "gridZ", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void cloudBenchmark::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  BBox domain(m_min,m_max);
  if( m_min == m_notSetMin  ||  m_max == m_notSetMax ){
    const Level* level = patch->getLevel();
    GridP grid  = level->getGrid();
    grid->getInteriorSpatialRange(domain);
    m_min = domain.min();
    m_max = domain.max();
  }

  Point midPt( (m_max - m_min)/2. + m_min);

  CCVariable<double>& abskg = tsk_info->get_field<CCVariable<double> >(m_abskg_name);
  CCVariable<double>& radT  = tsk_info->get_field<CCVariable<double> >("temperature");
  constCCVariable<double>& x = tsk_info->get_field<constCCVariable<double> >("gridX");
  constCCVariable<double>& y = tsk_info->get_field<constCCVariable<double> >("gridY");
  constCCVariable<double>& z = tsk_info->get_field<constCCVariable<double> >("gridZ");

  abskg.initialize(1.0);
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());
  Uintah::parallel_for( range, [&](int i, int j, int k){

  double xL=std::fabs( x(i,j,k));
  double yL=std::fabs( y(i,j,k));
  double zL=std::fabs( z(i,j,k));
  double kabsg_base=5.0;

  abskg(i,j,k)=(std::max(std::sin(xL*5.0*M_PI)*std::abs((std::sin(zL*3.0*M_PI+M_PI)) + (std::sin(yL *3.0*M_PI))),0.0)+std::max(std::sin(xL*5.0*M_PI+M_PI),0.0)*std::max(std::sin(zL*3.0*M_PI+M_PI) + std::sin(yL*3.0*M_PI+M_PI),0.0))*kabsg_base;

  });

  radT.initialize(0.0);
  Uintah::parallel_for( range, [&](int i, int j, int k){
  double xL=std::fabs( x(i,j,k) );
  double yL=std::fabs( y(i,j,k) );
  double zL=std::fabs( z(i,j,k) );
  double tempBase=1000;
  radT(i,j,k)= (std::max(sin(xL*5.0*M_PI)*std::max(sin((zL)*3.0*M_PI) + sin((yL) *3.0*M_PI),0.0),0.0)+std::max(sin(xL*5.0*M_PI),0.0)*std::max(sin(zL*3.0*M_PI+M_PI) + sin(yL*3.0*M_PI+M_PI),0.0))*tempBase;
  });

}

//--------------------------------------------------------------------------------------------------
void cloudBenchmark::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){
  register_initialize(variable_registry, false);
}

template <typename ExecSpace, typename MemSpace>
void cloudBenchmark::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  initialize(patch, tsk_info, execObj);
}

//--------------------------------------------------------------------------------------------------
void cloudBenchmark::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  register_variable( m_abskg_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_abskg_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "temperature", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "temperature", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace> void
cloudBenchmark::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& abskg = tsk_info->get_field<CCVariable<double> >(m_abskg_name);
  constCCVariable<double>& old_abskg = tsk_info->get_field<constCCVariable<double> >(m_abskg_name);
  CCVariable<double>& temp = tsk_info->get_field<CCVariable<double> >("temperature");
  constCCVariable<double>& old_temp = tsk_info->get_field<constCCVariable<double> >("temperature");

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for( range, [&](int i, int j, int k){

    abskg(i,j,k) = old_abskg(i,j,k);
    temp(i,j,k) = old_temp(i,j,k);

  });
}

} //namespace Uintah
