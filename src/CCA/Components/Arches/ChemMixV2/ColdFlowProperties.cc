#include <CCA/Components/Arches/ChemMixV2/ColdFlowProperties.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
ColdFlowProperties::ColdFlowProperties( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
ColdFlowProperties::~ColdFlowProperties(){

}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ColdFlowProperties::loadTaskComputeBCsFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::BC>( this
                                     , &ColdFlowProperties::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ColdFlowProperties::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &ColdFlowProperties::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ColdFlowProperties::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &ColdFlowProperties::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ColdFlowProperties::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &ColdFlowProperties::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ColdFlowProperties::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &ColdFlowProperties::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ColdFlowProperties::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &ColdFlowProperties::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ColdFlowProperties::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &ColdFlowProperties::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ColdFlowProperties::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &ColdFlowProperties::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ColdFlowProperties::loadTaskRestartInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::RESTART_INITIALIZE>( this
                                     , &ColdFlowProperties::restart_initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &ColdFlowProperties::restart_initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &ColdFlowProperties::restart_initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::problemSetup( ProblemSpecP& db ){

  for ( ProblemSpecP db_prop = db->findBlock("property");
        db_prop.get_rep() != nullptr;
        db_prop = db_prop->findNextBlock("property") ){

    std::string label;
    bool inverted = false;
    double value0;
    double value1;

    db_prop->getAttribute("label", label);
    db_prop->getAttribute("stream_0", value0);
    db_prop->getAttribute("stream_1", value1);
    inverted = db_prop->findBlock("volumetric");

    SpeciesInfo info{ value0, value1, inverted };

    m_name_to_value.insert( std::make_pair( label, info ));

  }

  db->findBlock("mixture_fraction")->getAttribute("label", m_mixfrac_label );

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::create_local_labels(){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_new_variable<CCVariable<double> >( i->first );
  }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::register_initialize( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void ColdFlowProperties::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    auto var = tsk_info->get_field<CCVariable<double>, double, MemSpace>( i->first );
    parallel_initialize(execObj, 0.0, var);
  }

  get_properties(execObj, patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( i->first, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void ColdFlowProperties::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    auto old_var = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( i->first );
    auto var = tsk_info->get_field<CCVariable<double>, double, MemSpace>( i->first );

    parallel_for(execObj, BlockRange(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex()) , KOKKOS_LAMBDA (int i,int j,int k){
      var(i,j,k) = old_var(i,j,k);
    });
  }
}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void ColdFlowProperties::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  get_properties(execObj, patch, tsk_info );

}

void ColdFlowProperties::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::MODIFIES, variable_registry );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void ColdFlowProperties::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  get_properties(execObj, patch, tsk_info );

}

void ColdFlowProperties::register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::MODIFIES, variable_registry );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void ColdFlowProperties::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto f = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( m_mixfrac_label );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());
    if ( !on_this_patch ) continue;

    //Get the iterator
    Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
    std::string facename = i_bc->second.name;

    IntVector iDir = patch->faceDirection( i_bc->second.face );

    for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){

      auto prop = tsk_info->get_field<CCVariable<double>, double, MemSpace>( i->first );
      const SpeciesInfo info = i->second;

      const bool volumetric = info.volumetric;
      const double stream_1 = info.stream_1;
      const double stream_2 = info.stream_2;

      parallel_for_unstructured(execObj,cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), KOKKOS_LAMBDA (int i,int j,int k) {

        int ip = i-iDir[0];
        int jp = j-iDir[1];
        int kp = k-iDir[2];

        const double f_interp = 0.5 *( f(i,j,k) + f(ip,jp,kp) );

        const double value = volumetric ?
                             1./(f_interp / stream_1 + ( 1. - f_interp ) / stream_2) :
                             f_interp * stream_1 + ( 1. - f_interp ) * stream_2;

        prop(i,j,k) = 2. * value - prop(ip,jp,kp);

      });
    }
  }
}

template <typename ExecSpace, typename MemSpace>
void ColdFlowProperties::get_properties( ExecutionObject<ExecSpace, MemSpace>& execObj, const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  auto f = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( m_mixfrac_label );

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){

    auto prop = tsk_info->get_field<CCVariable<double>, double, MemSpace>( i->first );
    const SpeciesInfo info = i->second;

    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

   const double stream_1 = info.stream_1;
   const double stream_2 = info.stream_2;

   if (info.volumetric){
    Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA ( int i, int j, int k ){
      prop(i,j,k) =1./(f(i,j,k) / stream_1 + ( 1. - f(i,j,k) ) / stream_2);
    });
   }else{
    Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA ( int i, int j, int k ){
      prop(i,j,k) =f(i,j,k) * stream_1 + ( 1. - f(i,j,k) ) * stream_2;
    });
   }

  }
}
