#include <CCA/Components/Arches/Transport/PressureBC.h>

using namespace Uintah::ArchesCore;
using namespace Uintah;

typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
PressureBC::PressureBC( std::string task_name, int matl_index ) :
AtomicTaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
PressureBC::~PressureBC()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureBC::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureBC::loadTaskInitializeFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureBC::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &PressureBC::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &PressureBC::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &PressureBC::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureBC::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureBC::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void PressureBC::problemSetup( ProblemSpecP& db ){
  m_press = "pressure";
}

//--------------------------------------------------------------------------------------------------
void PressureBC::create_local_labels()
{}

//--------------------------------------------------------------------------------------------------
void PressureBC::register_timestep_eval( std::vector<AFC::VariableInformation>& variable_registry,
                                 const int time_substep, const bool pack_tasks ){

  register_variable( m_press, AFC::MODIFIES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void PressureBC::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto p = tsk_info->get_field<CCVariable<double>, double, MemSpace>( m_press );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());
    if ( !on_this_patch ) continue;

    Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
    IntVector iDir = patch->faceDirection( i_bc->second.face );
    BndTypeEnum my_type = i_bc->second.type;

    if ( my_type == WALL_BC || my_type == INLET_BC  ){

      parallel_for_unstructured(execObj, cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), KOKKOS_LAMBDA (const int i,const int j,const int k) {
        // enforce dp/dn = 0
        p(i,j,k) = p(i-iDir[0],j-iDir[1],k-iDir[2]);

      });

    } else if ( my_type == OUTLET_BC || my_type == PRESSURE_BC ) {

      //enforce p = 0
      parallel_for_unstructured(execObj, cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), KOKKOS_LAMBDA (const int i,const int j,const int k) {
        p(i,j,k) = -p(i-iDir[0],j-iDir[1],k-iDir[2]);
      });
    }
  }
}
