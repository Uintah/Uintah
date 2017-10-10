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
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
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
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
  }
  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_t_vis_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
}

//--------------------------------------------------------------------------------------------------
void StressTensor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(m_w_vel_name);
  constCCVariable<double>&     D  = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_t_vis_name);

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
  double dudx =0.0;
  double dudy =0.0;
  double dudz =0.0;
  double dvdx =0.0;
  double dvdy =0.0;
  double dvdz =0.0;
  double dwdx =0.0;
  double dwdy =0.0;
  double dwdz =0.0;
  double mu11 = 0.0;
  double mu12 = 0.0;
  double mu13 = 0.0;
  double mu22 = 0.0;
  double mu23 = 0.0;
  double mu33 = 0.0;

  Uintah::parallel_for( x_range, [&](int i, int j, int k){

    mu11 = D(i-1,j,k); // it does not need interpolation
    mu22 = D(i,j-1,k);  // it does not need interpolation
    mu33 = D(i,j,k-1);  // it does not need interpolation
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

    sigma11(i,j,k) =  mu11 * 2.0*dudx;
    sigma12(i,j,k) =  mu12 * (dudy + dvdx );
    sigma13(i,j,k) =  mu13 * (dudz + dwdx );
    sigma22(i,j,k) =  mu22 * 2.0*dvdy;
    sigma23(i,j,k) =  mu23 * (dvdz + dwdy );
    sigma33(i,j,k) =  mu33 * 2.0*dwdz;

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
