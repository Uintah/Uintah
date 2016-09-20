#include <CCA/Components/Arches/PropertyModelsV2/FaceVelocities.h>

using namespace Uintah;

FaceVelocities::FaceVelocities( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){

  //hard coded velocity names:
  m_u_vel_name = "uVelocitySPBC";
  m_v_vel_name = "vVelocitySPBC";
  m_w_vel_name = "wVelocitySPBC";

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

FaceVelocities::~FaceVelocities(){}

void FaceVelocities::problemSetup( ProblemSpecP& db ){
}

void FaceVelocities::create_local_labels(){
  //U-CELL LABELS:
  register_new_variable<CCVariable<double> >(  "ucell_xvel");
  register_new_variable<SFCXVariable<double> >("ucell_yvel");
  register_new_variable<SFCXVariable<double> >("ucell_zvel");
  //V-CELL LABELS:
  register_new_variable<SFCYVariable<double> >("vcell_xvel");
  register_new_variable<CCVariable<double> >(  "vcell_yvel");
  register_new_variable<SFCYVariable<double> >("vcell_zvel");
  //W-CELL LABELS:
  register_new_variable<SFCZVariable<double> >("wcell_xvel");
  register_new_variable<SFCZVariable<double> >("wcell_yvel");
  register_new_variable<CCVariable<double> >(  "wcell_zvel");
}

//--------------------------------------------------------------------------------------------------

void FaceVelocities::register_initialize( AVarInfo& variable_registry ){
  for (auto iter = m_vel_names.begin(); iter != m_vel_names.end(); iter++ ){
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry );
  }
}

void FaceVelocities::initialize( const Patch*, ArchesTaskInfoManager* tsk_info ){


  CCVariable<double>&   ucell_xvel = *(tsk_info->get_uintah_field<CCVariable<double> >("ucell_xvel"));
  SFCXVariable<double>& ucell_yvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yvel"));
  SFCXVariable<double>& ucell_zvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zvel"));
  ucell_xvel.initialize(0.0);
  ucell_yvel.initialize(0.0);
  ucell_zvel.initialize(0.0);

  SFCYVariable<double>& vcell_xvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xvel"));
  CCVariable<double>&   vcell_yvel = *(tsk_info->get_uintah_field<CCVariable<double> >("vcell_yvel"));
  SFCYVariable<double>& vcell_zvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zvel"));
  vcell_xvel.initialize(0.0);
  vcell_yvel.initialize(0.0);
  vcell_zvel.initialize(0.0);

  SFCZVariable<double>& wcell_xvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xvel"));
  SFCZVariable<double>& wcell_yvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yvel"));
  CCVariable<double>&   wcell_zvel = *(tsk_info->get_uintah_field<CCVariable<double> >("wcell_zvel"));
  wcell_xvel.initialize(0.0);
  wcell_yvel.initialize(0.0);
  wcell_zvel.initialize(0.0);
}

//--------------------------------------------------------------------------------------------------

void FaceVelocities::register_timestep_eval( VIVec& variable_registry, const int time_substep ){
  for (auto iter = m_vel_names.begin(); iter != m_vel_names.end(); iter++ ){
    register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry );
  }
  register_variable(m_u_vel_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry);
  register_variable(m_v_vel_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry);
  register_variable(m_w_vel_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry);
}

void FaceVelocities::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(m_u_vel_name));
  constSFCYVariable<double>& vVel = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(m_v_vel_name));
  constSFCZVariable<double>& wVel = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(m_w_vel_name));

  CCVariable<double>&   ucell_xvel = *(tsk_info->get_uintah_field<CCVariable<double> >("ucell_xvel"));
  SFCXVariable<double>& ucell_yvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yvel"));
  SFCXVariable<double>& ucell_zvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zvel"));

  SFCYVariable<double>& vcell_xvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xvel"));
  CCVariable<double>&   vcell_yvel = *(tsk_info->get_uintah_field<CCVariable<double> >("vcell_yvel"));
  SFCYVariable<double>& vcell_zvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zvel"));

  SFCZVariable<double>& wcell_xvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xvel"));
  SFCZVariable<double>& wcell_yvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yvel"));
  CCVariable<double>&   wcell_zvel = *(tsk_info->get_uintah_field<CCVariable<double> >("wcell_zvel"));

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

    ucell_xvel(i,j,k) = 0.5*(uVel(i,j,k) + uVel(i-1,j,k));
    ucell_yvel(i,j,k) = 0.5*(vVel(i,j,k) + vVel(i-1,j,k));
    ucell_zvel(i,j,k) = 0.5*(wVel(i,j,k) + wVel(i-1,j,k));

    vcell_xvel(i,j,k) = 0.5*(uVel(i,j,k) + uVel(i,j-1,k));
    vcell_yvel(i,j,k) = 0.5*(vVel(i,j,k) + vVel(i,j-1,k));
    vcell_zvel(i,j,k) = 0.5*(wVel(i,j,k) + wVel(i,j-1,k));

    wcell_xvel(i,j,k) = 0.5*(uVel(i,j,k) + uVel(i,j,k-1));
    wcell_yvel(i,j,k) = 0.5*(vVel(i,j,k) + vVel(i,j,k-1));
    wcell_zvel(i,j,k) = 0.5*(wVel(i,j,k) + wVel(i,j,k-1));

  });
}
