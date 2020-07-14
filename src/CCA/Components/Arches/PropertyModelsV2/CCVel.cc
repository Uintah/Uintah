#include <CCA/Components/Arches/PropertyModelsV2/CCVel.h>
#include <CCA/Components/Arches/UPSHelper.h>

using namespace Uintah;
using namespace ArchesCore;

//--------------------------------------------------------------------------------------------------
CCVel::CCVel( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){}

//--------------------------------------------------------------------------------------------------
CCVel::~CCVel(){}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CCVel::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CCVel::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &CCVel::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &CCVel::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &CCVel::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CCVel::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &CCVel::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &CCVel::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &CCVel::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CCVel::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &CCVel::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &CCVel::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &CCVel::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CCVel::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void CCVel::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
  m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
  m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );

  m_u_vel_name_cc = m_u_vel_name + "_cc";
  m_v_vel_name_cc = m_v_vel_name + "_cc";
  m_w_vel_name_cc = m_w_vel_name + "_cc";

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
  if ( db->findBlock("TurbulenceModels")){
    if ( db->findBlock("TurbulenceModels")->findBlock("model")){
      std::string turb_closure_model;
      std::string conv_scheme;
      db->findBlock("TurbulenceModels")->findBlock("model")->getAttribute("type", turb_closure_model);
      if ( turb_closure_model == "multifractal" ){
        if (db->findBlock("KMomentum")->findBlock("convection")){
          std::stringstream msg;
          msg << "ERROR: Cannot use KMomentum->convection if you are using the multifracal nles closure." << std::endl;
          throw InvalidValue(msg.str(),__FILE__,__LINE__);
        } else {
            m_ghost_cells=2;
            conv_scheme="fourth";
            m_int_scheme = ArchesCore::get_interpolant_from_string( conv_scheme );
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void CCVel::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_u_vel_name_cc);
  register_new_variable<CCVariable<double> >( m_v_vel_name_cc);
  register_new_variable<CCVariable<double> >( m_w_vel_name_cc);

  register_new_variable<CCVariable<double> >( "x_vorticity" );
  register_new_variable<CCVariable<double> >( "y_vorticity" );
  register_new_variable<CCVariable<double> >( "z_vorticity" );

}

//--------------------------------------------------------------------------------------------------
void CCVel::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name, AFC::REQUIRES,m_ghost_cells , AFC::NEWDW, variable_registry, m_task_name );
  register_variable( m_v_vel_name, AFC::REQUIRES,m_ghost_cells , AFC::NEWDW, variable_registry, m_task_name );
  register_variable( m_w_vel_name, AFC::REQUIRES,m_ghost_cells , AFC::NEWDW, variable_registry, m_task_name );
  register_variable( m_u_vel_name_cc, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_v_vel_name_cc, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_w_vel_name_cc, AFC::COMPUTES, variable_registry, m_task_name );

  register_variable( "x_vorticity", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "y_vorticity", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "z_vorticity", AFC::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CCVel::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  compute_velocities( execObj, patch, tsk_info );

  compute_vorticity( execObj, patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
void CCVel::register_timestep_init( AVarInfo& variable_registry , const bool pack_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name_cc, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_v_vel_name_cc, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_w_vel_name_cc, AFC::COMPUTES, variable_registry, m_task_name );

  register_variable( m_u_vel_name_cc, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, m_task_name );
  register_variable( m_v_vel_name_cc, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, m_task_name );
  register_variable( m_w_vel_name_cc, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
CCVel::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto old_u_cc = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_u_vel_name_cc);
  auto old_v_cc = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_v_vel_name_cc);
  auto old_w_cc = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_w_vel_name_cc);

  auto u_cc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_u_vel_name_cc);
  auto v_cc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_v_vel_name_cc);
  auto w_cc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_w_vel_name_cc);

  Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
      u_cc(i,j,k)=old_u_cc(i,j,k);
      v_cc(i,j,k)=old_v_cc(i,j,k);
      w_cc(i,j,k)=old_w_cc(i,j,k);
    });

}

//--------------------------------------------------------------------------------------------------
void CCVel::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name, AFC::REQUIRES, m_ghost_cells, AFC::NEWDW, variable_registry, time_substep ,m_task_name );
  register_variable( m_v_vel_name, AFC::REQUIRES, m_ghost_cells, AFC::NEWDW, variable_registry, time_substep ,m_task_name );
  register_variable( m_w_vel_name, AFC::REQUIRES, m_ghost_cells, AFC::NEWDW, variable_registry, time_substep ,m_task_name );

  register_variable( m_u_vel_name_cc, AFC::MODIFIES, variable_registry, time_substep , m_task_name );
  register_variable( m_v_vel_name_cc, AFC::MODIFIES, variable_registry, time_substep , m_task_name );
  register_variable( m_w_vel_name_cc, AFC::MODIFIES, variable_registry, time_substep , m_task_name );

  register_variable( "x_vorticity", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "y_vorticity", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "z_vorticity", AFC::COMPUTES, variable_registry, time_substep , m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CCVel::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  compute_velocities( execObj, patch, tsk_info );

  compute_vorticity( execObj, patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CCVel::compute_velocities(ExecutionObject<ExecSpace, MemSpace>& execObj, const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  auto u = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(m_u_vel_name);
  auto v = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(m_v_vel_name);
  auto w = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(m_w_vel_name);

  auto u_cc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_u_vel_name_cc);
  auto v_cc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_v_vel_name_cc);
  auto w_cc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_w_vel_name_cc);

  parallel_initialize(execObj,0.0,u_cc,v_cc,w_cc);

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  ArchesCore::doInterpolation( execObj, range, u_cc, u , 1, 0, 0, m_int_scheme );
  ArchesCore::doInterpolation( execObj, range, v_cc, v , 0, 1, 0, m_int_scheme );
  ArchesCore::doInterpolation( execObj, range, w_cc, w , 0, 0, 1, m_int_scheme );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CCVel::compute_vorticity(ExecutionObject<ExecSpace, MemSpace>& execObj, const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  auto u = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(m_u_vel_name);
  auto v = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(m_v_vel_name);
  auto w = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(m_w_vel_name);

  auto w_x = tsk_info->get_field<CCVariable<double>, double, MemSpace>("x_vorticity");
  auto w_y = tsk_info->get_field<CCVariable<double>, double, MemSpace>("y_vorticity");
  auto w_z = tsk_info->get_field<CCVariable<double>, double, MemSpace>("z_vorticity");

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  Vector DX = patch->dCell();

  // const double Fdx = 4*DX[0];
  // const double Fdy = 4*DX[1];
  // const double Fdz = 4*DX[2];

  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){

    // Cell-centered values for cell-centered vorticity - but has the negative
    //  behavior of decaying the vorticity in the higher wavenumbers.
    // const double dudy = ((u(i,j+1,k) + u(i+1,j+1,k)) - (u(i,j-1,k)+u(i+1,j-1,k))) / Fdy;
    // const double dudz = ((u(i,j,k+1) + u(i+1,j,k+1)) - (u(i,j,k-1)+u(i+1,j,k-1))) / Fdz;
    // const double dvdx = ((v(i+1,j,k) + v(i+1,j+1,k)) - (v(i-1,j,k)+v(i-1,j+1,k))) / Fdx;
    // const double dvdz = ((v(i,j,k+1) + v(i,j+1,k+1)) - (v(i,j,k-1)+v(i,j+1,k-1))) / Fdz;
    // const double dwdx = ((w(i+1,j,k) + w(i+1,j,k+1)) - (w(i-1,j,k)+w(i-1,j,k+1))) / Fdx;
    // const double dwdy = ((w(i,j+1,k) + w(i,j+1,k+1)) - (w(i,j-1,k)+w(i,j-1,k+1))) / Fdy;

    //Vorticity is 'edge-centered' to avoid artificial decay in
    // higher wave numbers due to numerics. Uncomment the above lines
    // to get cell-centered vorticity.
    const double dudy = (u(i,j,k) - u(i,j-1,k)) / DX[1];
    const double dudz = (u(i,j,k) - u(i,j,k-1)) / DX[2];

    const double dvdx = (v(i,j,k) - v(i-1,j,k)) / DX[0];
    const double dvdz = (v(i,j,k) - v(i,j,k-1)) / DX[2];

    const double dwdx = (w(i,j,k) - w(i-1,j,k)) / DX[0];
    const double dwdy = (w(i,j,k) - w(i,j-1,k)) / DX[1];

    // Here are the actual vorticity components
    w_x(i,j,k) = dwdy - dvdz;
    w_y(i,j,k) = dudz - dwdx;
    w_z(i,j,k) = dvdx - dudy;

  });
}
