#include <CCA/Components/Arches/TurbulenceModels/DSFT.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
DSFT::DSFT( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

}

//--------------------------------------------------------------------------------------------------
DSFT::~DSFT(){
}

//--------------------------------------------------------------------------------------------------
void
DSFT::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;
  // u, v , w velocities
  m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );
  m_density_name = parse_ups_for_role( DENSITY, db, "density" );

  m_rhou_vel_name = "x-mom";
  m_rhov_vel_name = "y-mom";
  m_rhow_vel_name = "z-mom" ;

  m_cc_u_vel_name = m_u_vel_name + "_cc";
  m_cc_v_vel_name = m_v_vel_name + "_cc";
  m_cc_w_vel_name = m_w_vel_name + "_cc";

  std::string m_Type_filter_name;
  db->findBlock("filter")->getAttribute("type",m_Type_filter_name);

  Type_filter = get_filter_from_string( m_Type_filter_name );

}

//--------------------------------------------------------------------------------------------------
void
DSFT::create_local_labels(){

  register_new_variable<CCVariable<double> >("IsI");
  register_new_variable<CCVariable<double> >("s11");
  register_new_variable<CCVariable<double> >("s12");
  register_new_variable<CCVariable<double> >("s13");
  register_new_variable<CCVariable<double> >("s22");
  register_new_variable<CCVariable<double> >("s23");
  register_new_variable<CCVariable<double> >("s33");
  register_new_variable<CCVariable<double> >("Beta11");
  register_new_variable<CCVariable<double> >("Beta12");
  register_new_variable<CCVariable<double> >("Beta13");
  register_new_variable<CCVariable<double> >("Beta22");
  register_new_variable<CCVariable<double> >("Beta23");
  register_new_variable<CCVariable<double> >("Beta33");

  register_new_variable<CCVariable<double> >( "Filterrho");
  register_new_variable<CCVariable<double> >( "rhoUU");
  register_new_variable<CCVariable<double> >( "rhoVV");
  register_new_variable<CCVariable<double> >( "rhoWW");
  register_new_variable<CCVariable<double> >( "rhoUV");
  register_new_variable<CCVariable<double> >( "rhoUW");
  register_new_variable<CCVariable<double> >( "rhoVW");
  register_new_variable<CCVariable<double> >( "rhoU");
  register_new_variable<CCVariable<double> >( "rhoV");
  register_new_variable<CCVariable<double> >( "rhoW");

}

//--------------------------------------------------------------------------------------------------
void
DSFT::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry , const bool packed_tasks){

  register_variable( "Filterrho", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
}

//--------------------------------------------------------------------------------------------------
void
DSFT::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& filterRho = tsk_info->get_uintah_field_add< CCVariable<double> >("Filterrho");
  filterRho.initialize(0.0);
}
//--------------------------------------------------------------------------------------------------
void
DSFT::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry , const bool packed_tasks){

}

//--------------------------------------------------------------------------------------------------
void
DSFT::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

//--------------------------------------------------------------------------------------------------
void
DSFT::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep , const bool packed_tasks){
  int nG = 1;
  if (packed_tasks ){
   nG = 3;
  }
  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, nG , ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_density_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  register_variable( m_cc_u_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  register_variable( "IsI", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "Beta11", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "Beta12", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "Beta13", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "Beta22", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "Beta23", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "Beta33", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);

  register_variable( "s11", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "s12", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "s13", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "s22", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "s23", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);
  register_variable( "s33", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks);

  register_variable( "Filterrho", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , _task_name, packed_tasks);
  register_variable( "rhoUU", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , _task_name, packed_tasks);
  register_variable( "rhoVV", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , _task_name, packed_tasks);
  register_variable( "rhoWW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );
  register_variable( "rhoUV", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );
  register_variable( "rhoUW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );
  register_variable( "rhoVW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );
  register_variable( "rhoU", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );
  register_variable( "rhoV", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );
  register_variable( "rhoW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , _task_name, packed_tasks );

}

//--------------------------------------------------------------------------------------------------
void
DSFT::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(m_u_vel_name));
  constSFCYVariable<double>& vVel = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(m_v_vel_name));
  constSFCZVariable<double>& wVel = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(m_w_vel_name));
  constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));

  constCCVariable<double>& CCuVel = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_cc_u_vel_name));
  constCCVariable<double>& CCvVel = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_cc_v_vel_name));
  constCCVariable<double>& CCwVel = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_cc_w_vel_name));

  const Vector Dx = patch->dCell(); //
  int nG1 = 0;
  int nG2 = 0;
  // Compute IsI and Beta

  int nGhosts2 = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)
  int nGhosts1 = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)

  if ( tsk_info->packed_tasks() ){
    nGhosts2 = 2;
    nGhosts1 = 1;
    nG1 = nGhosts1;
    nG2 = nGhosts2;
  }

  IntVector low_filter = patch->getCellLowIndex() + IntVector(-nG1,-nG1,-nG1);
  IntVector high_filter = patch->getCellHighIndex() + IntVector(nG1,nG1,nG1);
  IntVector low_filter2 = patch->getCellLowIndex() + IntVector(-nG2,-nG2,-nG2);
  IntVector high_filter2 = patch->getCellHighIndex() + IntVector(nG2,nG2,nG2);
  Uintah::BlockRange range2(low_filter2, high_filter2 );
  Uintah::BlockRange range1(low_filter, high_filter );

  CCVariable<double>& IsI = tsk_info->get_uintah_field_add< CCVariable<double> >("IsI",nGhosts2 );
  CCVariable<double>& s11 = tsk_info->get_uintah_field_add< CCVariable<double> >("s11",nGhosts2 );
  CCVariable<double>& s12 = tsk_info->get_uintah_field_add< CCVariable<double> >("s12",nGhosts2 );
  CCVariable<double>& s13 = tsk_info->get_uintah_field_add< CCVariable<double> >("s13",nGhosts2 );
  CCVariable<double>& s22 = tsk_info->get_uintah_field_add< CCVariable<double> >("s22",nGhosts2 );
  CCVariable<double>& s23 = tsk_info->get_uintah_field_add< CCVariable<double> >("s23",nGhosts2 );
  CCVariable<double>& s33 = tsk_info->get_uintah_field_add< CCVariable<double> >("s33",nGhosts2 );

  CCVariable<double>& Beta11 = tsk_info->get_uintah_field_add< CCVariable<double> >("Beta11",nGhosts2 );
  CCVariable<double>& Beta12 = tsk_info->get_uintah_field_add< CCVariable<double> >("Beta12",nGhosts2 );
  CCVariable<double>& Beta13 = tsk_info->get_uintah_field_add< CCVariable<double> >("Beta13",nGhosts2 );
  CCVariable<double>& Beta22 = tsk_info->get_uintah_field_add< CCVariable<double> >("Beta22",nGhosts2 );
  CCVariable<double>& Beta23 = tsk_info->get_uintah_field_add< CCVariable<double> >("Beta23",nGhosts2 );
  CCVariable<double>& Beta33 = tsk_info->get_uintah_field_add< CCVariable<double> >("Beta33",nGhosts2 );

  s11.initialize(0.0);
  s12.initialize(0.0);
  s13.initialize(0.0);
  s22.initialize(0.0);
  s23.initialize(0.0);
  s33.initialize(0.0);
  Beta11.initialize(0.0);
  Beta12.initialize(0.0);
  Beta13.initialize(0.0);
  Beta22.initialize(0.0);
  Beta23.initialize(0.0);
  Beta33.initialize(0.0);

  computeIsInsij get_IsIsij(IsI, s11, s22, s33, s12, s13, s23, uVel, vVel, wVel, CCuVel, CCvVel, CCwVel, Dx);
  Uintah::parallel_for(range2,get_IsIsij);

  Uintah::parallel_for( range2, [&](int i, int j, int k){
    Beta11(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s11(i,j,k);
    Beta22(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s22(i,j,k);
    Beta33(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s33(i,j,k);
    Beta12(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s12(i,j,k);
    Beta13(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s13(i,j,k);
    Beta23(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s23(i,j,k);
  });

  // Filter rho
  CCVariable<double>& filterRho = tsk_info->get_uintah_field_add< CCVariable<double> >("Filterrho", nGhosts1);
  filterRho.initialize(0.0);
  Uintah::FilterVarT< constCCVariable<double> > get_frho(rho, filterRho, Type_filter);
  Uintah::parallel_for(range1,get_frho);

  // Compute rhouiuj at cc
  CCVariable<double>& rhoUU = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoUU",nGhosts2);
  CCVariable<double>& rhoVV = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoVV",nGhosts2);
  CCVariable<double>& rhoWW = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoWW",nGhosts2);
  CCVariable<double>& rhoUV = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoUV",nGhosts2);
  CCVariable<double>& rhoUW = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoUW",nGhosts2);
  CCVariable<double>& rhoVW = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoVW",nGhosts2);
  CCVariable<double>& rhoU = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoU",nGhosts2);
  CCVariable<double>& rhoV = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoV",nGhosts2);
  CCVariable<double>& rhoW = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoW",nGhosts2);

  rhoUU.initialize(0.0);
  rhoVV.initialize(0.0);
  rhoWW.initialize(0.0);
  rhoUV.initialize(0.0);
  rhoUW.initialize(0.0);
  rhoVW.initialize(0.0);
  rhoU.initialize(0.0);
  rhoV.initialize(0.0);
  rhoW.initialize(0.0);

  Uintah::parallel_for( range2, [&](int i, int j, int k){
    rhoUU(i,j,k) = rho(i,j,k)*CCuVel(i,j,k)*CCuVel(i,j,k);
    rhoVV(i,j,k) = rho(i,j,k)*CCvVel(i,j,k)*CCvVel(i,j,k);
    rhoWW(i,j,k) = rho(i,j,k)*CCwVel(i,j,k)*CCwVel(i,j,k);
    rhoUV(i,j,k) = rho(i,j,k)*CCuVel(i,j,k)*CCvVel(i,j,k);
    rhoUW(i,j,k) = rho(i,j,k)*CCuVel(i,j,k)*CCwVel(i,j,k);
    rhoVW(i,j,k) = rho(i,j,k)*CCvVel(i,j,k)*CCwVel(i,j,k);
    rhoU(i,j,k) = rho(i,j,k)*CCuVel(i,j,k);
    rhoV(i,j,k) = rho(i,j,k)*CCvVel(i,j,k);
    rhoW(i,j,k) = rho(i,j,k)*CCwVel(i,j,k);
  });
}

} //namespace Uintah
