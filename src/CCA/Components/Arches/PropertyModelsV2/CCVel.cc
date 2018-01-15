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
void CCVel::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );

  m_u_vel_name_cc = m_u_vel_name + "_cc";
  m_v_vel_name_cc = m_v_vel_name + "_cc";
  m_w_vel_name_cc = m_w_vel_name + "_cc";

}

//--------------------------------------------------------------------------------------------------
void CCVel::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_u_vel_name_cc);
  register_new_variable<CCVariable<double> >( m_v_vel_name_cc);
  register_new_variable<CCVariable<double> >( m_w_vel_name_cc);

}

//--------------------------------------------------------------------------------------------------
void CCVel::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, _task_name );
  register_variable( m_v_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, _task_name );
  register_variable( m_w_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, _task_name );
  register_variable( m_u_vel_name_cc, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_v_vel_name_cc, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_w_vel_name_cc, AFC::COMPUTES, variable_registry, _task_name );

}

//--------------------------------------------------------------------------------------------------
void CCVel::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_velocities( patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
void CCVel::register_timestep_init( AVarInfo& variable_registry , const bool pack_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name_cc, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_v_vel_name_cc, AFC::COMPUTES, variable_registry, _task_name );
  register_variable( m_w_vel_name_cc, AFC::COMPUTES, variable_registry, _task_name );

  register_variable( m_u_vel_name_cc, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, _task_name );
  register_variable( m_v_vel_name_cc, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, _task_name );
  register_variable( m_w_vel_name_cc, AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, _task_name );
  
}

//--------------------------------------------------------------------------------------------------
void CCVel::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& old_u_cc = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_u_vel_name_cc);
  constCCVariable<double>& old_v_cc = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_v_vel_name_cc);
  constCCVariable<double>& old_w_cc = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_w_vel_name_cc);

  CCVariable<double>& u_cc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_u_vel_name_cc);
  CCVariable<double>& v_cc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_v_vel_name_cc);
  CCVariable<double>& w_cc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_w_vel_name_cc);

  u_cc.copy(old_u_cc);
  v_cc.copy(old_v_cc);
  w_cc.copy(old_w_cc);

}

//--------------------------------------------------------------------------------------------------
void CCVel::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep ,_task_name );
  register_variable( m_v_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep ,_task_name );
  register_variable( m_w_vel_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep ,_task_name );

  register_variable( m_u_vel_name_cc, AFC::MODIFIES, variable_registry, time_substep , _task_name );
  register_variable( m_v_vel_name_cc, AFC::MODIFIES, variable_registry, time_substep , _task_name );
  register_variable( m_w_vel_name_cc, AFC::MODIFIES, variable_registry, time_substep , _task_name );

}

//--------------------------------------------------------------------------------------------------
void CCVel::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  compute_velocities( patch, tsk_info );

}

//--------------------------------------------------------------------------------------------------
void CCVel::compute_velocities( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& u = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& v = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& w = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(m_w_vel_name);
  CCVariable<double>& u_cc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_u_vel_name_cc);
  CCVariable<double>& v_cc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_v_vel_name_cc);
  CCVariable<double>& w_cc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_w_vel_name_cc);

  u_cc.initialize(0.0);
  v_cc.initialize(0.0);
  w_cc.initialize(0.0);

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    u_cc(i,j,k) = (u(i,j,k) + u(i+1,j,k)) / 2.;
    v_cc(i,j,k) = (v(i,j,k) + v(i+1,j,k)) / 2.;
    w_cc(i,j,k) = (w(i,j,k) + w(i+1,j,k)) / 2.;
  });

}
