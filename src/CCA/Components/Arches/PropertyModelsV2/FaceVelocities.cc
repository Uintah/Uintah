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
void FaceVelocities::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );

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
void FaceVelocities::initialize( const Patch*, ArchesTaskInfoManager* tsk_info ){

  SFCXVariable<double>& ucell_xvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xvel"));
  SFCXVariable<double>& ucell_yvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yvel"));
  SFCXVariable<double>& ucell_zvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zvel"));
  ucell_xvel.initialize(0.0);
  ucell_yvel.initialize(0.0);
  ucell_zvel.initialize(0.0);

  SFCYVariable<double>& vcell_xvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xvel"));
  SFCYVariable<double>& vcell_yvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_yvel"));
  SFCYVariable<double>& vcell_zvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zvel"));
  vcell_xvel.initialize(0.0);
  vcell_yvel.initialize(0.0);
  vcell_zvel.initialize(0.0);

  SFCZVariable<double>& wcell_xvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xvel"));
  SFCZVariable<double>& wcell_yvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yvel"));
  SFCZVariable<double>& wcell_zvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zvel"));
  wcell_xvel.initialize(0.0);
  wcell_yvel.initialize(0.0);
  wcell_zvel.initialize(0.0);

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
void FaceVelocities::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(m_u_vel_name));
  constSFCYVariable<double>& vVel = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(m_v_vel_name));
  constSFCZVariable<double>& wVel = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(m_w_vel_name));

  SFCXVariable<double>& ucell_xvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xvel"));
  SFCXVariable<double>& ucell_yvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yvel"));
  SFCXVariable<double>& ucell_zvel = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zvel"));

  SFCYVariable<double>& vcell_xvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xvel"));
  SFCYVariable<double>& vcell_yvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_yvel"));
  SFCYVariable<double>& vcell_zvel = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zvel"));

  SFCZVariable<double>& wcell_xvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xvel"));
  SFCZVariable<double>& wcell_yvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yvel"));
  SFCZVariable<double>& wcell_zvel = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zvel"));

  // initialize all velocities
  ucell_xvel.initialize(0.0);
  ucell_yvel.initialize(0.0);
  ucell_zvel.initialize(0.0);
  vcell_xvel.initialize(0.0);
  vcell_yvel.initialize(0.0);
  vcell_zvel.initialize(0.0);
  wcell_xvel.initialize(0.0);
  wcell_yvel.initialize(0.0);
  wcell_zvel.initialize(0.0);

  // bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  // bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  // bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  // bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  // bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  // bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  ArchesCore::OneDInterpolator my_interpolant_ucellx( ucell_xvel, uVel, -1, 0, 0 );
  ArchesCore::OneDInterpolator my_interpolant_ucelly( ucell_yvel, vVel, -1, 0, 0 );
  ArchesCore::OneDInterpolator my_interpolant_ucellz( ucell_zvel, wVel, -1, 0, 0 );

  ArchesCore::OneDInterpolator my_interpolant_vcellx( vcell_xvel, uVel,0,-1, 0 );
  ArchesCore::OneDInterpolator my_interpolant_vcelly( vcell_yvel, vVel,0,-1, 0 );
  ArchesCore::OneDInterpolator my_interpolant_vcellz( vcell_zvel, wVel,0,-1, 0 );

  ArchesCore::OneDInterpolator my_interpolant_wcellx( wcell_xvel, uVel,0, 0, -1);
  ArchesCore::OneDInterpolator my_interpolant_wcelly( wcell_yvel, vVel,0, 0, -1);
  ArchesCore::OneDInterpolator my_interpolant_wcellz( wcell_zvel, wVel,0, 0, -1);

  IntVector low = patch->getCellLowIndex();
  IntVector high = patch->getCellHighIndex();

  //x-direction:
  GET_WALL_BUFFERED_PATCH_RANGE(low,high,1,1,0,1,0,1);
  Uintah::BlockRange x_range(low, high);
    if ( m_int_scheme == ArchesCore::SECONDCENTRAL ) {

      ArchesCore::SecondCentral ci;
      Uintah::parallel_for( x_range, my_interpolant_ucellx, ci );
      Uintah::parallel_for( x_range, my_interpolant_ucelly, ci );
      Uintah::parallel_for( x_range, my_interpolant_ucellz, ci );

    } else if ( m_int_scheme== ArchesCore::FOURTHCENTRAL ){

      ArchesCore::FourthCentral ci;
      Uintah::parallel_for( x_range, my_interpolant_ucellx, ci );
      Uintah::parallel_for( x_range, my_interpolant_ucelly, ci );
      Uintah::parallel_for( x_range, my_interpolant_ucellz, ci );

  }

  //y-direction:
  low = patch->getCellLowIndex();
  high = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(low,high,0,1,1,1,0,1);
  Uintah::BlockRange y_range(low, high);

  if ( m_int_scheme == ArchesCore::SECONDCENTRAL ) {

      ArchesCore::SecondCentral ci;
      Uintah::parallel_for( y_range, my_interpolant_vcellx, ci );
      Uintah::parallel_for( y_range, my_interpolant_vcelly, ci );
      Uintah::parallel_for( y_range, my_interpolant_vcellz, ci );

    } else if ( m_int_scheme== ArchesCore::FOURTHCENTRAL ){

      //std::cout<<"fourth order face"<<std::endl;
      ArchesCore::FourthCentral ci;
      Uintah::parallel_for( y_range, my_interpolant_vcellx, ci );
      Uintah::parallel_for( y_range, my_interpolant_vcelly, ci );
      Uintah::parallel_for( y_range, my_interpolant_vcellz, ci );

  }

  //z-direction:
  low = patch->getCellLowIndex();
  high = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(low,high,0,1,0,1,1,1);
  Uintah::BlockRange z_range(low, high);

  if ( m_int_scheme == ArchesCore::SECONDCENTRAL ) {

      ArchesCore::SecondCentral ci;
      Uintah::parallel_for( z_range, my_interpolant_wcellx, ci );
      Uintah::parallel_for( z_range, my_interpolant_wcelly, ci );
      Uintah::parallel_for( z_range, my_interpolant_wcellz, ci );

    } else if ( m_int_scheme== ArchesCore::FOURTHCENTRAL ){

      ArchesCore::FourthCentral ci;
      Uintah::parallel_for( z_range, my_interpolant_wcellx, ci );
      Uintah::parallel_for( z_range, my_interpolant_wcelly, ci );
      Uintah::parallel_for( z_range, my_interpolant_wcellz, ci );

  }

}
