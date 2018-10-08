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
void StressTensor::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

    m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
    m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
    m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );
    m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY, db );
//
  /* It is going to use central scheme as default   */
  diff_scheme = "central";
  Nghost_cells = 1;
  m_eps_name = "volFraction";
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
void StressTensor::initialize( const Patch*, ArchesTaskInfoManager* tsk_info ){


  CCVariable<double>& sigma11 = *(tsk_info->get_uintah_field<CCVariable<double> >(m_sigma_t_names[0]));
  CCVariable<double>& sigma12 = *(tsk_info->get_uintah_field<CCVariable<double> >(m_sigma_t_names[1]));
  CCVariable<double>& sigma13 = *(tsk_info->get_uintah_field<CCVariable<double> >(m_sigma_t_names[2]));
  CCVariable<double>& sigma22 = *(tsk_info->get_uintah_field<CCVariable<double> >(m_sigma_t_names[3]));
  CCVariable<double>& sigma23 = *(tsk_info->get_uintah_field<CCVariable<double> >(m_sigma_t_names[4]));
  CCVariable<double>& sigma33 = *(tsk_info->get_uintah_field<CCVariable<double> >(m_sigma_t_names[5]));

  sigma11.initialize(0.0);
  sigma12.initialize(0.0);
  sigma13.initialize(0.0);
  sigma22.initialize(0.0);
  sigma23.initialize(0.0);
  sigma33.initialize(0.0);
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
  register_variable( m_eps_name, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::LATEST, variable_registry, time_substep);
}

//--------------------------------------------------------------------------------------------------
void StressTensor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(m_w_vel_name);
  constCCVariable<double>&     D  = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_t_vis_name);
  constCCVariable<double>&   eps  = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_eps_name);

  CCVariable<double>& sigma11 = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigma_t_names[0]);
  CCVariable<double>& sigma12 = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigma_t_names[1]);
  CCVariable<double>& sigma13 = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigma_t_names[2]);
  CCVariable<double>& sigma22 = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigma_t_names[3]);
  CCVariable<double>& sigma23 = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigma_t_names[4]);
  CCVariable<double>& sigma33 = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigma_t_names[5]);

  // initialize all velocities
  sigma11.initialize(0.0);
  sigma12.initialize(0.0);
  sigma13.initialize(0.0);
  sigma22.initialize(0.0);
  sigma23.initialize(0.0);
  sigma33.initialize(0.0);

  Vector Dx = patch->dCell();

  IntVector low = patch->getCellLowIndex();
  IntVector high = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(low, high,0,1,0,1,0,1);  
  Uintah::BlockRange x_range(low, high);

  const int xdir[3] = {1,0,0};
  const int ydir[3] = {0,1,0};
  const int zdir[3] = {0,0,1};

  Uintah::parallel_for( x_range, [&](int i, int j, int k){

    double dudx = 0.0;
    double dudy = 0.0;
    double dudz = 0.0;
    double dvdx = 0.0;
    double dvdy = 0.0;
    double dvdz = 0.0;
    double dwdx = 0.0;
    double dwdy = 0.0;
    double dwdz = 0.0;
    double mu12 = 0.0;
    double mu13 = 0.0;
    double mu23 = 0.0;

    mu12  = 0.5*(D(i-1,j,k)+D(i,j,k)); // First interpolation at j
    mu12 += 0.5*(D(i-1,j-1,k)+D(i,j-1,k));// Second interpolation at j-1
    mu12 *= 0.5;
    mu13  = 0.5*(D(i-1,j,k-1)+D(i,j,k-1));//First interpolation at k-1
    mu13 += 0.5*(D(i-1,j,k)+D(i,j,k));//Second interpolation at k
    mu13 *= 0.5;
    mu23  = 0.5*(D(i,j,k)+D(i,j,k-1));// First interpolation at j
    mu23 += 0.5*(D(i,j-1,k)+D(i,j-1,k-1));// Second interpolation at j-1
    mu23 *= 0.5;

    VelocityDerivative_central(dudx,dudy,dudz,uVel,Dx,i,j,k);
    VelocityDerivative_central(dvdx,dvdy,dvdz,vVel,Dx,i,j,k);
    VelocityDerivative_central(dwdx,dwdy,dwdz,wVel,Dx,i,j,k);
    
    const double afc12 = get_eps(eps, i, j, k, xdir, ydir);
    const double afc13 = get_eps(eps, i, j, k, xdir, zdir);
    const double afc23 = get_eps(eps, i, j, k, ydir, zdir);

    sigma12(i,j,k) = afc12 * mu12 * (dudy + dvdx );
    sigma13(i,j,k) = afc13 * mu13 * (dudz + dwdx );
    sigma23(i,j,k) = afc23 * mu23 * (dvdz + dwdy );

  });

  IntVector lowNx = patch->getCellLowIndex();
  IntVector highNx = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(lowNx, highNx,1,1,0,0,0,0);  
  Uintah::BlockRange range1(lowNx, highNx);
  Uintah::parallel_for( range1, [&](int i, int j, int k){

    const double mu11  = D(i-1,j,k); // it does not need interpolation
    const double dudx  = (uVel(i,j,k) - uVel(i-1,j,k))/Dx.x();
    const double afc11 = get_eps(eps, i, j, k, xdir, xdir);
    sigma11(i,j,k)     = afc11 * mu11 * 2.0*dudx;

  });

  IntVector lowNy = patch->getCellLowIndex();
  IntVector highNy = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(lowNy, highNy,0,0,1,1,0,0);  
  Uintah::BlockRange range2(lowNy, highNy);
  Uintah::parallel_for( range2, [&](int i, int j, int k){
    const double mu22 = D(i,j-1,k);  // it does not need interpolation
    const double afc22 = get_eps(eps, i, j, k, ydir, ydir);
    const double dvdy  = (vVel(i,j,k) - vVel(i,j-1,k))/Dx.y();
    sigma22(i,j,k) = afc22 * mu22 * 2.0*dvdy;

  });

  IntVector lowNz = patch->getCellLowIndex();
  IntVector highNz = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(lowNz, highNz,0,0,0,0,1,1);  
  Uintah::BlockRange range3(lowNz, highNz);
  Uintah::parallel_for( range3, [&](int i, int j, int k){
    const double mu33 = D(i,j,k-1);  // it does not need interpolation
    const double afc33 = get_eps(eps, i, j, k, zdir, zdir);
    const double dwdz  = (wVel(i,j,k) - wVel(i,j,k-1))/Dx.z();
    sigma33(i,j,k) = afc33 * mu33 * 2.0*dwdz;

  });
}

void StressTensor::VelocityDerivative_central(double &dudx, double &dudy, double &dudz, const Array3<double> &u, const Vector& Dx, int i, int j, int k)
{

  using namespace Uintah::ArchesCore;

  {
  STENCIL3_1D(0);
    dudx = (u(IJK_) - u(IJK_M_))/Dx.x();
  }
  {
  STENCIL3_1D(1);
    dudy = (u(IJK_) - u(IJK_M_))/Dx.y();
  }
  {
  STENCIL3_1D(2);
    dudz = (u(IJK_) - u(IJK_M_))/Dx.z();
  }
}
