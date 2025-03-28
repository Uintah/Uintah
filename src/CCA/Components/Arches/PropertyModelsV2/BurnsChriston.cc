#include <CCA/Components/Arches/PropertyModelsV2/BurnsChriston.h>
#include <ostream>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace BurnsChriston::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace BurnsChriston::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &BurnsChriston::initialize<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &BurnsChriston::initialize<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &BurnsChriston::initialize<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &BurnsChriston::initialize<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &BurnsChriston::initialize<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace BurnsChriston::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace BurnsChriston::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &BurnsChriston::timestep_init<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &BurnsChriston::timestep_init<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &BurnsChriston::timestep_init<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &BurnsChriston::timestep_init<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &BurnsChriston::timestep_init<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace BurnsChriston::loadTaskRestartInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::RESTART_INITIALIZE>( this
                                     , &BurnsChriston::restart_initialize<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &BurnsChriston::restart_initialize<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &BurnsChriston::restart_initialize<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &BurnsChriston::restart_initialize<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &BurnsChriston::restart_initialize<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
void
BurnsChriston::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_prop = db;
  db_prop->getWithDefault("min", m_min, m_notSetMin);
  db_prop->getWithDefault("max", m_max, m_notSetMax);

  // bulletproofing  min & max must be set
  if( ( m_min == m_notSetMin && m_max != m_notSetMax) ||
      ( m_min != m_notSetMin && m_max == m_notSetMax) ){
    std::ostringstream warn;
    warn << "\nERROR:<property_calculator type=burns_christon>\n "
         << "You must specify both a min: "<< m_min << " & max point: "<< m_max <<".";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

    db_prop->getAttribute("label",m_abskg_name);

}

//--------------------------------------------------------------------------------------------------
void
BurnsChriston::create_local_labels(){

    register_new_variable<CCVariable<double> >(m_abskg_name);
    register_new_variable<CCVariable<double> >("temperature");

}

//--------------------------------------------------------------------------------------------------
void
BurnsChriston::register_initialize( VIVec& variable_registry , const bool pack_tasks){

    register_variable( m_abskg_name, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( "temperature", ArchesFieldContainer::COMPUTES, variable_registry );

    register_variable( "gridX", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( "gridY", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( "gridZ", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::NEWDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void BurnsChriston::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

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

    bool inside = false;
    if ( x(i,j,k) >= m_min.x() && y(i,j,k) >= m_min.y() && z(i,j,k) >= m_min.z() ){
      if ( x(i,j,k) <= m_max.x() && y(i,j,k) <= m_max.y() && z(i,j,k) <= m_max.z() ){
        inside = true;
      }
    }

    abskg(i,j,k) = ( inside == false ) ? 0.0 :
                   0.90 * ( 1.0 - 2.0 * std::fabs( x(i,j,k) - midPt.x() ) )
                   * ( 1.0 - 2.0 * std::fabs( y(i,j,k) - midPt.y() ) )
                   * ( 1.0 - 2.0 * std::fabs( z(i,j,k) - midPt.z() ) )
                   + 0.1;

  });

  radT.initialize(-pow(1.0/5.670367e-8,0.25));
  Uintah::parallel_for( range, [&](int i, int j, int k){
    radT(i,j,k) = pow(1.0/5.670367e-8,0.25);
  });

}

//--------------------------------------------------------------------------------------------------
void BurnsChriston::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

  register_variable( m_abskg_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "temperature", ArchesFieldContainer::COMPUTES, variable_registry );

  register_variable( "gridX", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "gridY", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "gridZ", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::NEWDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void BurnsChriston::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

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
    bool inside = false;
    if ( x(i,j,k) >= m_min.x() && y(i,j,k) >= m_min.y() && z(i,j,k) >= m_min.z() ){
      if ( x(i,j,k) <= m_max.x() && y(i,j,k) <= m_max.y() && z(i,j,k) <= m_max.z() ){
        inside = true;
      }
    }

    abskg(i,j,k) = ( inside == false ) ? 0.0 :
                   0.90 * ( 1.0 - 2.0 * std::fabs( x(i,j,k) - midPt.x() ) )
                   * ( 1.0 - 2.0 * std::fabs( y(i,j,k) - midPt.y() ) )
                   * ( 1.0 - 2.0 * std::fabs( z(i,j,k) - midPt.z() ) )
                   + 0.1;

  });

  radT.initialize(-pow(1.0/5.670367e-8,0.25));
  Uintah::parallel_for( range, [&](int i, int j, int k){
    radT(i,j,k) = pow(1.0/5.670367e-8,0.25);
  });
}

//--------------------------------------------------------------------------------------------------
void BurnsChriston::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  register_variable( m_abskg_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_abskg_name, ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "temperature", ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "temperature", ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::OLDDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace> void
BurnsChriston::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

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
