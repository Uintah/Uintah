#include <CCA/Components/Arches/Utility/SurfaceNormals.h>
#include <CCA/Components/Arches/GridTools.h>
#include <math.h>

using namespace Uintah;

//helper
typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

//--------------------------------------------------------------------------------------------------
SurfaceNormals::SurfaceNormals( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

SurfaceNormals::~SurfaceNormals(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceNormals::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceNormals::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &SurfaceNormals::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SurfaceNormals::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &SurfaceNormals::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceNormals::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceNormals::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &SurfaceNormals::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SurfaceNormals::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &SurfaceNormals::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceNormals::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::problemSetup( ProblemSpecP& db ){
}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::create_local_labels(){

  //outward facing normals
  //
  //    gas    <--|--   solid
  //
  register_new_variable<SFCXVariable<double> >( "surf_out_normX" );
  register_new_variable<SFCYVariable<double> >( "surf_out_normY" );
  register_new_variable<SFCZVariable<double> >( "surf_out_normZ" );

  //inward facing normals
  //
  //    gas    --|-->   solid
  //
  register_new_variable<SFCXVariable<double> >( "surf_in_normX" );
  register_new_variable<SFCYVariable<double> >( "surf_in_normY" );
  register_new_variable<SFCZVariable<double> >( "surf_in_normZ" );

}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::register_initialize( VIVec& variable_registry , const bool packed_tasks){

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "volFraction", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW,
                      variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void SurfaceNormals::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto vol_fraction = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>("volFraction");

  auto n_in_x = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("surf_in_normX");
  auto n_in_y = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("surf_in_normY");
  auto n_in_z = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("surf_in_normZ");

  auto n_out_x = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("surf_out_normX");
  auto n_out_y = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("surf_out_normY");
  auto n_out_z = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("surf_out_normZ");

  Uintah::BlockRange full_range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  Uintah::parallel_for( execObj, full_range, KOKKOS_LAMBDA( int i, int j, int k ){
    n_in_x(i,j,k) = 0.0;
    n_out_x(i,j,k) = 0.0;
    n_in_y(i,j,k) = 0.0;
    n_out_y(i,j,k) = 0.0;
    n_in_z(i,j,k) = 0.0;
    n_out_z(i,j,k) = 0.0;
  });

  const double noise = 1e-10;

  // X-dimension

  GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(0,1)
  Uintah::BlockRange fx_range( low_fx_patch_range, high_fx_patch_range );

  Uintah::parallel_for( execObj, fx_range, KOKKOS_LAMBDA( int i, int j, int k ){
    n_out_x(i,j,k) = ( vol_fraction(i,j,k) - vol_fraction(i-1,j,k) )
                     / abs( vol_fraction(i,j,k) - vol_fraction(i-1,j,k) + noise);
    n_in_x(i,j,k) = ( vol_fraction(i-1,j,k) - vol_fraction(i,j,k) ) /
                    abs( vol_fraction(i-1,j,k) - vol_fraction(i,j,k) + noise);
  });

  // Y-dimension

  GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(0,1)
  Uintah::BlockRange fy_range( low_fy_patch_range, high_fy_patch_range );

  Uintah::parallel_for( execObj, fy_range, KOKKOS_LAMBDA( int i, int j, int k ){
    n_out_y(i,j,k) = ( vol_fraction(i,j,k) - vol_fraction(i,j-1,k) )
                     / abs( vol_fraction(i,j,k) - vol_fraction(i,j-1,k) + noise);
    n_in_y(i,j,k) = ( vol_fraction(i,j-1,k) - vol_fraction(i,j,k) ) /
                    abs( vol_fraction(i,j,k) - vol_fraction(i,j-1,k) + noise);
  });

  // Z-dimension

  GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(0,1)
  Uintah::BlockRange fz_range( low_fz_patch_range, high_fz_patch_range );

  Uintah::parallel_for( execObj, fz_range, KOKKOS_LAMBDA( int i, int j, int k ){
    n_out_z(i,j,k) = ( vol_fraction(i,j,k) - vol_fraction(i,j,k-1) )
                     / abs( vol_fraction(i,j,k) - vol_fraction(i,j,k-1) + noise);
    n_in_z(i,j,k) = ( vol_fraction(i,j,k-1) - vol_fraction(i,j,k) ) /
                    abs( vol_fraction(i,j,k-1) - vol_fraction(i,j,k) + noise);
  });

}

//--------------------------------------------------------------------------------------------------
void
SurfaceNormals::register_timestep_init( VIVec& variable_registry, const bool packed_tasks){

  register_variable( "surf_out_normX" , ArchesFieldContainer::REQUIRES , 0,
                      ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::REQUIRES , 0,
                      ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::REQUIRES , 0,
                      ArchesFieldContainer::OLDDW, variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::OLDDW , variable_registry );

  register_variable( "surf_out_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_out_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

  register_variable( "surf_in_normX" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normY" , ArchesFieldContainer::COMPUTES , variable_registry );
  register_variable( "surf_in_normZ" , ArchesFieldContainer::COMPUTES , variable_registry );

}

template <typename ExecSpace, typename MemSpace> void
SurfaceNormals::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto n_in_x = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("surf_in_normX");
  auto n_in_y = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("surf_in_normY");
  auto n_in_z = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("surf_in_normZ");
  auto old_n_in_x = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>("surf_in_normX");
  auto old_n_in_y = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>("surf_in_normY");
  auto old_n_in_z = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>("surf_in_normZ");

  auto n_out_x = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("surf_out_normX");
  auto n_out_y = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("surf_out_normY");
  auto n_out_z = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("surf_out_normZ");
  auto old_n_out_x = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>("surf_out_normX");
  auto old_n_out_y = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>("surf_out_normY");
  auto old_n_out_z = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>("surf_out_normZ");

  Uintah::BlockRange full_range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  Uintah::parallel_for( execObj, full_range, KOKKOS_LAMBDA( int i, int j, int k ){
    n_in_x(i,j,k)  = old_n_in_x(i,j,k);
    n_out_x(i,j,k) = old_n_out_x(i,j,k);
    n_in_y(i,j,k)  = old_n_in_y(i,j,k);
    n_out_y(i,j,k) = old_n_out_y(i,j,k);
    n_in_z(i,j,k)  = old_n_in_z(i,j,k);
    n_out_z(i,j,k) = old_n_out_z(i,j,k);
  });
}
