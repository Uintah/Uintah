#include <CCA/Components/Arches/PropertyModelsV2/FaceVelocities.h>
#include <CCA/Components/Arches/UPSHelper.h>

using namespace Uintah;
using namespace ArchesCore;

//--------------------------------------------------------------------------------------------------
FaceVelocities::FaceVelocities( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){

  m_vel_names.resize(9);
  m_vel_names[0] = "ucell_xvel";
  m_vel_names[1] = "ucell_yvel";
  m_vel_names[2] = "ucell_zvel";
  m_vel_names[3] = "vcell_xvel";
  m_vel_names[4] = "vcell_yvel";
  m_vel_names[5] = "vcell_zvel";
  m_vel_names[6] = "wcell_xvel";
  m_vel_names[7] = "wcell_yvel";
  m_vel_names[8] = "wcell_zvel";

}

//--------------------------------------------------------------------------------------------------
FaceVelocities::~FaceVelocities(){}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace FaceVelocities::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace FaceVelocities::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &FaceVelocities::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &FaceVelocities::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &FaceVelocities::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace FaceVelocities::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &FaceVelocities::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &FaceVelocities::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &FaceVelocities::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace FaceVelocities::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace FaceVelocities::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void FaceVelocities::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
  m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
  m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );

  m_int_scheme = ArchesCore::get_interpolant_from_string( "second" ); //default second order
  m_ghost_cells = 1; //default for 2nd order

  if ( db->findBlock("KMomentum") ){
    if (db->findBlock("KMomentum")->findBlock("convection")){

      std::string conv_scheme;
      db->findBlock("KMomentum")->findBlock("convection")->getAttribute("scheme", conv_scheme);

      if (conv_scheme == "fourth"){
        m_ghost_cells=2;
        m_int_scheme = ArchesCore::get_interpolant_from_string( conv_scheme );
      }
    }
  }

}

//--------------------------------------------------------------------------------------------------
void FaceVelocities::create_local_labels(){
  //U-CELL LABELS:
  register_new_variable<SFCXVariable<double> >("ucell_xvel");
  register_new_variable<SFCXVariable<double> >("ucell_yvel");
  register_new_variable<SFCXVariable<double> >("ucell_zvel");
  //V-CELL LABELS:
  register_new_variable<SFCYVariable<double> >("vcell_xvel");
  register_new_variable<SFCYVariable<double> >("vcell_yvel");
  register_new_variable<SFCYVariable<double> >("vcell_zvel");
  //W-CELL LABELS:
  register_new_variable<SFCZVariable<double> >("wcell_xvel");
  register_new_variable<SFCZVariable<double> >("wcell_yvel");
  register_new_variable<SFCZVariable<double> >("wcell_zvel");
}

//--------------------------------------------------------------------------------------------------
void FaceVelocities::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){
  for (auto iter = m_vel_names.begin(); iter != m_vel_names.end(); iter++ ){
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void FaceVelocities::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto ucell_xvel = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("ucell_xvel");
  auto ucell_yvel = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("ucell_yvel");
  auto ucell_zvel = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("ucell_zvel");

  auto vcell_xvel = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("vcell_xvel");
  auto vcell_yvel = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("vcell_yvel");
  auto vcell_zvel = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("vcell_zvel");

  auto wcell_xvel = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("wcell_xvel");
  auto wcell_yvel = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("wcell_yvel");
  auto wcell_zvel = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("wcell_zvel");

  parallel_initialize( execObj, 0.0
                     , ucell_xvel, ucell_yvel, ucell_zvel
                     , vcell_xvel, vcell_yvel, vcell_zvel
                     , wcell_xvel, wcell_yvel, wcell_zvel
                     );
}

//--------------------------------------------------------------------------------------------------
void FaceVelocities::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){
  for (auto iter = m_vel_names.begin(); iter != m_vel_names.end(); iter++ ){
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
  }
  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES,m_ghost_cells , ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES,m_ghost_cells , ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES,m_ghost_cells , ArchesFieldContainer::LATEST, variable_registry, time_substep, m_task_name );
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void FaceVelocities::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto uVel = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(m_u_vel_name);
  auto vVel = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(m_v_vel_name);
  auto wVel = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(m_w_vel_name);

  auto ucell_xvel = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("ucell_xvel");
  auto ucell_yvel = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("ucell_yvel");
  auto ucell_zvel = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("ucell_zvel");

  auto vcell_xvel = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("vcell_xvel");
  auto vcell_yvel = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("vcell_yvel");
  auto vcell_zvel = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("vcell_zvel");

  auto wcell_xvel = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("wcell_xvel");
  auto wcell_yvel = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("wcell_yvel");
  auto wcell_zvel = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("wcell_zvel");

  parallel_initialize( execObj, 0.0
                     , ucell_xvel, ucell_yvel, ucell_zvel
                     , vcell_xvel, vcell_yvel, vcell_zvel
                     , wcell_xvel, wcell_yvel, wcell_zvel
                     );

  // bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  // bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  // bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  // bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  // bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  // bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  IntVector low = patch->getCellLowIndex();
  IntVector high = patch->getCellHighIndex();

  //x-direction:
  GET_WALL_BUFFERED_PATCH_RANGE(low,high,1,1,0,1,0,1);
  Uintah::BlockRange x_range(low, high);

  ArchesCore::doInterpolation( execObj, x_range, ucell_xvel, uVel, -1, 0, 0, m_int_scheme );
  ArchesCore::doInterpolation( execObj, x_range, ucell_yvel, vVel, -1, 0, 0, m_int_scheme );
  ArchesCore::doInterpolation( execObj, x_range, ucell_zvel, wVel, -1, 0, 0, m_int_scheme );

  //y-direction:
  low = patch->getCellLowIndex();
  high = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(low,high,0,1,1,1,0,1);
  Uintah::BlockRange y_range(low, high);

  ArchesCore::doInterpolation( execObj, y_range, vcell_xvel, uVel, 0, -1, 0, m_int_scheme );
  ArchesCore::doInterpolation( execObj, y_range, vcell_yvel, vVel, 0, -1, 0, m_int_scheme );
  ArchesCore::doInterpolation( execObj, y_range, vcell_zvel, wVel, 0, -1, 0, m_int_scheme );

  //z-direction:
  low = patch->getCellLowIndex();
  high = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(low,high,0,1,0,1,1,1);
  Uintah::BlockRange z_range(low, high);

  ArchesCore::doInterpolation( execObj, z_range, wcell_xvel, uVel, 0, 0, -1, m_int_scheme );
  ArchesCore::doInterpolation( execObj, z_range, wcell_yvel, vVel, 0, 0, -1, m_int_scheme );
  ArchesCore::doInterpolation( execObj, z_range, wcell_zvel, wVel, 0, 0, -1, m_int_scheme );

}
