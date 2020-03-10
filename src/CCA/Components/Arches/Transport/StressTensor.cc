#include <CCA/Components/Arches/Transport/StressTensor.h>
#include <CCA/Components/Arches/GridTools.h>

using namespace Uintah;
using namespace ArchesCore;

//--------------------------------------------------------------------------------------------------
StressTensor::StressTensor( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){

  m_sigma_t_names.resize(6);
  m_sigma_t_names[0] = "sigma11";
  m_sigma_t_names[1] = "sigma12";
  m_sigma_t_names[2] = "sigma13";
  m_sigma_t_names[3] = "sigma22";
  m_sigma_t_names[4] = "sigma23";
  m_sigma_t_names[5] = "sigma33";

}

//--------------------------------------------------------------------------------------------------
StressTensor::~StressTensor(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace StressTensor::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace StressTensor::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &StressTensor::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &StressTensor::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &StressTensor::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace StressTensor::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &StressTensor::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &StressTensor::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &StressTensor::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace StressTensor::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace StressTensor::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void StressTensor::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

    m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
    m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
    m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );
    m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db );
//
  /* It is going to use central scheme as default   */
  diff_scheme = "central";
  Nghost_cells = 1;
  ArchesCore::GridVarMap< SFCXVariable<double> > var_map_x;
  var_map_x.problemSetup( db );
  m_eps_x_name = var_map_x.vol_frac_name;

  ArchesCore::GridVarMap< SFCYVariable<double> > var_map_y;
  var_map_y.problemSetup( db );
  m_eps_y_name = var_map_y.vol_frac_name;

  ArchesCore::GridVarMap< SFCZVariable<double> > var_map_z;
  var_map_z.problemSetup( db );
  m_eps_z_name = var_map_z.vol_frac_name;
}

//--------------------------------------------------------------------------------------------------
void StressTensor::create_local_labels(){
  for (auto iter = m_sigma_t_names.begin(); iter != m_sigma_t_names.end(); iter++ ){
    register_new_variable<CCVariable<double> >(*iter);
  }
}

//--------------------------------------------------------------------------------------------------
void StressTensor::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){
  for (auto iter = m_sigma_t_names.begin(); iter != m_sigma_t_names.end(); iter++ ){
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void StressTensor::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){


  auto sigma11 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[0]);
  auto sigma12 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[1]);
  auto sigma13 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[2]);
  auto sigma22 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[3]);
  auto sigma23 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[4]);
  auto sigma33 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[5]);

  parallel_initialize(execObj, 0.0, sigma11, sigma12, sigma13, sigma22, sigma23, sigma33);
}

//--------------------------------------------------------------------------------------------------
void StressTensor::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){
  // time_substep?
  for (auto iter = m_sigma_t_names.begin(); iter != m_sigma_t_names.end(); iter++ ){
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  }
  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::LATEST, variable_registry, time_substep);
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::LATEST, variable_registry, time_substep);
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::LATEST, variable_registry, time_substep);
  register_variable( m_t_vis_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_eps_x_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::OLDDW, variable_registry, time_substep);
  register_variable( m_eps_y_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::OLDDW, variable_registry, time_substep);
  register_variable( m_eps_z_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::OLDDW, variable_registry, time_substep);
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void StressTensor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto uVel = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(m_u_vel_name);
  auto vVel = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(m_v_vel_name);
  auto wVel = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(m_w_vel_name);
  auto D  = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_t_vis_name);
  auto eps_x = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(m_eps_x_name);
  auto eps_y = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(m_eps_y_name);
  auto eps_z = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(m_eps_z_name);

  auto sigma11 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[0]);
  auto sigma12 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[1]);
  auto sigma13 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[2]);
  auto sigma22 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[3]);
  auto sigma23 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[4]);
  auto sigma33 = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_sigma_t_names[5]);

  // initialize all velocities
  parallel_initialize(execObj, 0.0, sigma11, sigma12, sigma13, sigma22, sigma23, sigma33);

  Vector Dx = patch->dCell();

  IntVector low = patch->getCellLowIndex();
  IntVector high = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(low, high,0,1,0,1,0,1);
  Uintah::BlockRange x_range(low, high);
 
  //auto apply_uVelStencil=functorCreationWrapper(  uVel,  Dx); // non-macro approach gives cuda streaming error downstream
  //auto apply_vVelStencil=functorCreationWrapper(  vVel,  Dx);
  //auto apply_wVelStencil=functorCreationWrapper(  wVel,  Dx);

  Uintah::parallel_for(execObj, x_range, KOKKOS_LAMBDA (int i, int j, int k){

    double dudy = 0.0;
    double dudz = 0.0;
    double dvdx = 0.0;
    double dvdz = 0.0;
    double dwdx = 0.0;
    double dwdy = 0.0;

    const double mu12  = 0.5 * ( 0.5 * (D(i-1,j,k)+D(i,j,k))
                               + 0.5 * (D(i-1,j-1,k)+D(i,j-1,k)) );
    const double mu13  = 0.5 * ( 0.5 * (D(i-1,j,k-1)+D(i,j,k-1))
                               + 0.5 * (D(i-1,j,k)+D(i,j,k)) );
    const double mu23  = 0.5 * ( 0.5 * ( D(i,j,k)+D(i,j,k-1))
                               + 0.5 * (D(i,j-1,k)+D(i,j-1,k-1)) );

    //apply_uVelStencil(dudx,dudy,dudz,i,j,k);  // non-macro approach gives cuda streaming error downstream, likely due to saving templated value as reference instead of value. But must save by reference to suppor legacy code.  poosibly Use getKokkosView in functor constructor
    //apply_vVelStencil(dvdx,dvdy,dvdz,i,j,k);
    //apply_wVelStencil(dwdx,dwdy,dwdz,i,j,k);

    {
      STENCIL3_1D(1);
      dudy = eps_x(IJK_)*eps_x(IJK_M_)*(uVel(IJK_) - uVel(IJK_M_))/Dx.y();
    }
    {
      STENCIL3_1D(2);
      dudz = eps_x(IJK_)*eps_x(IJK_M_)*(uVel(IJK_) - uVel(IJK_M_))/Dx.z();\
    }
    {\
      STENCIL3_1D(0);\
      dvdx = eps_y(IJK_)*eps_y(IJK_M_)*(vVel(IJK_) - vVel(IJK_M_))/Dx.x();\
    }\
    {\
      STENCIL3_1D(2);\
      dvdz = eps_y(IJK_)*eps_y(IJK_M_)*(vVel(IJK_) - vVel(IJK_M_))/Dx.z();
    }
    {\
      STENCIL3_1D(0);\
      dwdx = eps_z(IJK_)*eps_z(IJK_M_)*(wVel(IJK_) - wVel(IJK_M_))/Dx.x();\
    }\
    {\
      STENCIL3_1D(1);\
      dwdy = eps_z(IJK_)*eps_z(IJK_M_)*(wVel(IJK_) - wVel(IJK_M_))/Dx.y();\
    }\

    sigma12(i,j,k) =  mu12 * (dudy + dvdx );
    sigma13(i,j,k) =  mu13 * (dudz + dwdx );
    sigma23(i,j,k) =  mu23 * (dvdz + dwdy );

  });

  IntVector lowNx = patch->getCellLowIndex();
  IntVector highNx = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(lowNx, highNx,1,1,0,0,0,0);
  Uintah::BlockRange range1(lowNx, highNx);
  Uintah::parallel_for(execObj, range1, KOKKOS_LAMBDA (int i, int j, int k){

    const double mu11  = D(i-1,j,k); // it does not need interpolation
    const double dudx  = eps_x(i,j,k)*eps_x(i-1,j,k) * (uVel(i,j,k) - uVel(i-1,j,k))/Dx.x();
    sigma11(i,j,k)     =  mu11 * 2.0*dudx;

  });

  IntVector lowNy = patch->getCellLowIndex();
  IntVector highNy = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(lowNy, highNy,0,0,1,1,0,0);
  Uintah::BlockRange range2(lowNy, highNy);
  Uintah::parallel_for(execObj, range2, KOKKOS_LAMBDA (int i, int j, int k){
    const double mu22 = D(i,j-1,k);  // it does not need interpolation
    const double dvdy  = eps_y(i,j,k)*eps_y(i,j-1,k) * (vVel(i,j,k) - vVel(i,j-1,k))/Dx.y();
    sigma22(i,j,k) =  mu22 * 2.0*dvdy;

  });

  IntVector lowNz = patch->getCellLowIndex();
  IntVector highNz = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(lowNz, highNz,0,0,0,0,1,1);
  Uintah::BlockRange range3(lowNz, highNz);
  Uintah::parallel_for(execObj, range3, KOKKOS_LAMBDA (int i, int j, int k){
    const double mu33 = D(i,j,k-1);  // it does not need interpolation
    const double dwdz  = eps_z(i,j,k)*eps_z(i,j,k-1) * (wVel(i,j,k) - wVel(i,j,k-1))/Dx.z();
    sigma33(i,j,k) = mu33 * 2.0*dwdz;

  });
}
