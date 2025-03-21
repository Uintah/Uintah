#include <CCA/Components/Arches/ChemMixV2/ConstantStateProperties.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
ConstantStateProperties::ConstantStateProperties( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
ConstantStateProperties::~ConstantStateProperties(){

}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ConstantStateProperties::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ConstantStateProperties::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &ConstantStateProperties::initialize<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &ConstantStateProperties::initialize<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &ConstantStateProperties::initialize<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &ConstantStateProperties::initialize<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &ConstantStateProperties::initialize<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ConstantStateProperties::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ConstantStateProperties::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &ConstantStateProperties::timestep_init<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &ConstantStateProperties::timestep_init<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &ConstantStateProperties::timestep_init<KOKKOS_DEFAULT_HOST_TAG>    // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &ConstantStateProperties::timestep_init<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &ConstantStateProperties::timestep_init<KOKKOS_DEFAULT_DEVICE_TAG>            // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ConstantStateProperties::loadTaskRestartInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::RESTART_INITIALIZE>( this
                                     , &ConstantStateProperties::restart_initialize<UINTAH_CPU_TAG>               // Task supports non-Kokkos builds
                                     //, &ConstantStateProperties::restart_initialize<KOKKOS_OPENMP_TAG>          // Task supports Kokkos::OpenMP builds
                                     //, &ConstantStateProperties::restart_initializet<KOKKOS_DEFAULT_HOST_TAG>   // Task supports Kokkos::DefaultHostExecutionSpace builds
                                     //, &ConstantStateProperties::restart_initialize<KOKKOS_DEFAULT_DEVICE_TAG>  // Task supports Kokkos::DefaultExecutionSpace builds
                                     //, &ColdFlowProperties::timestep_init<KOKKOS_DEFAULT_DEVICE_TAG>    // Task supports Kokkos builds
                                     );
}

//--------------------------------------------------------------------------------------------------
void ConstantStateProperties::problemSetup( ProblemSpecP& db ){

  for ( ProblemSpecP db_prop = db->findBlock("const_property");
        db_prop.get_rep() != nullptr;
        db_prop = db_prop->findNextBlock("const_property") ){

    std::string label;
    double value;

    db_prop->getAttribute("label", label);
    db_prop->getAttribute("value", value);

    m_name_to_value.insert( std::make_pair( label, value));

  }

}

//--------------------------------------------------------------------------------------------------
void ConstantStateProperties::create_local_labels(){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_new_variable<CCVariable<double> >( i->first );
  }

}

//--------------------------------------------------------------------------------------------------
void ConstantStateProperties::register_initialize( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void ConstantStateProperties::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( i->first );
    var.initialize(i->second);
  }

}

//--------------------------------------------------------------------------------------------------
void ConstantStateProperties::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( i->first, ArchesFieldContainer::NEEDSLABEL, 0, ArchesFieldContainer::OLDDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void ConstantStateProperties::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    constCCVariable<double>& old_var = tsk_info->get_field<constCCVariable<double> >( i->first );
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( i->first );

    var.copyData(old_var);
  }

}

//--------------------------------------------------------------------------------------------------
void ConstantStateProperties::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void ConstantStateProperties::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    CCVariable<double>& var = tsk_info->get_field<CCVariable<double> >( i->first );
    var.initialize(i->second);
  }

}
