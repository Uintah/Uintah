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

  m_volFraction_name = "volFraction";
  m_cc_u_vel_name = parse_ups_for_role( CCUVELOCITY, db, "CCUVelocity" );//;m_u_vel_name + "_cc";
  m_cc_v_vel_name = parse_ups_for_role( CCVVELOCITY, db, "CCVVelocity" );//m_v_vel_name + "_cc";
  m_cc_w_vel_name = parse_ups_for_role( CCWVELOCITY, db, "CCWVelocity" );;//m_w_vel_name + "_cc";

  if (m_u_vel_name == "uVelocitySPBC") { // this is production code
    m_create_labels_IsI_t_viscosity = false;
  }

  std::string m_Type_filter_name;
  db->findBlock("filter")->getAttribute("type",m_Type_filter_name);
  m_IsI_name = "strainMagnitudeLabel";
  //m_ref_density_name = "denRefArray"; // name used in production code
  //m_cell_type_name = "cellType";
  Type_filter = ArchesCore::get_filter_from_string( m_Type_filter_name );
  m_Filter.get_w(Type_filter);
}

//--------------------------------------------------------------------------------------------------
void
DSFT::create_local_labels(){

  if (m_create_labels_IsI_t_viscosity) {
    register_new_variable<CCVariable<double> >(m_IsI_name);
  }
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
  register_new_variable<SFCXVariable<double> >( "Filterrhou");
  register_new_variable<SFCYVariable<double> >( "Filterrhov");
  register_new_variable<SFCZVariable<double> >( "Filterrhow");
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


  register_variable( "Filterrho", ArchesFieldContainer::COMPUTES ,  variable_registry,  m_task_name, packed_tasks);
  register_variable( "Filterrhou", ArchesFieldContainer::COMPUTES ,  variable_registry,  m_task_name, packed_tasks);
  register_variable( "Filterrhov", ArchesFieldContainer::COMPUTES ,  variable_registry,  m_task_name, packed_tasks);
  register_variable( "Filterrhow", ArchesFieldContainer::COMPUTES ,  variable_registry,  m_task_name, packed_tasks);
}

//--------------------------------------------------------------------------------------------------
void
DSFT::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& filterRho = tsk_info->get_uintah_field_add< CCVariable<double> >("Filterrho");
  SFCXVariable<double>& filterRhoU = tsk_info->get_uintah_field_add< SFCXVariable<double> >("Filterrhou");
  SFCYVariable<double>& filterRhoV = tsk_info->get_uintah_field_add< SFCYVariable<double> >("Filterrhov");
  SFCZVariable<double>& filterRhoW = tsk_info->get_uintah_field_add< SFCZVariable<double> >("Filterrhow");
  filterRho.initialize(0.0);
  filterRhoU.initialize(0.0);
  filterRhoV.initialize(0.0);
  filterRhoW.initialize(0.0);


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
  int nGrho = nG +1;
  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, nG , ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_density_name, ArchesFieldContainer::REQUIRES, nGrho, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, nGrho, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_cc_u_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, ArchesFieldContainer::REQUIRES, nG, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  //register_variable( m_ref_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  //register_variable( m_cell_type_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  //register_variable( "rhoBC", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  register_variable( m_IsI_name, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "Beta11", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "Beta12", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "Beta13", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "Beta22", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "Beta23", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "Beta33", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable( "s11", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "s12", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "s13", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "s22", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "s23", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  register_variable( "s33", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);

  register_variable( "Filterrho", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , m_task_name, packed_tasks);
  register_variable( "Filterrhou", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , m_task_name, packed_tasks);
  register_variable( "Filterrhov", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , m_task_name, packed_tasks);
  register_variable( "Filterrhow", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , m_task_name, packed_tasks);
  register_variable( "rhoUU", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , m_task_name, packed_tasks);
  register_variable( "rhoVV", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep  , m_task_name, packed_tasks);
  register_variable( "rhoWW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );
  register_variable( "rhoUV", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );
  register_variable( "rhoUW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );
  register_variable( "rhoVW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );
  register_variable( "rhoU", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );
  register_variable( "rhoV", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );
  register_variable( "rhoW", ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks );

}

//--------------------------------------------------------------------------------------------------
void
DSFT::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(m_u_vel_name));
  constSFCYVariable<double>& vVel = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(m_v_vel_name));
  constSFCZVariable<double>& wVel = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(m_w_vel_name));
  constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));
  //constCCVariable<double>& ref_rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_ref_density_name));
  //constCCVariable<int>& cell_type = *(tsk_info->get_const_uintah_field<constCCVariable<int> >(m_cell_type_name));

  constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);
  constCCVariable<double>& CCuVel = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_cc_u_vel_name));
  constCCVariable<double>& CCvVel = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_cc_v_vel_name));
  constCCVariable<double>& CCwVel = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_cc_w_vel_name));

  const Vector Dx = patch->dCell(); //
  int nG1 = 0;
  int nG2 = 0;
  // Compute IsI and Beta

  int nGhosts2 = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)
  int nGhosts1 = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)
  int nGrho = 2;

  if ( tsk_info->packed_tasks() ){
    nGhosts2 = 2;
    nGhosts1 = 1;
    nG1 = nGhosts1;
    nG2 = nGhosts2;
    nGrho = 4;
  }

  IntVector low_filter = patch->getCellLowIndex() + IntVector(-nG1,-nG1,-nG1);
  IntVector high_filter = patch->getCellHighIndex() + IntVector(nG1,nG1,nG1);
  IntVector low_filter2 = patch->getCellLowIndex() + IntVector(-nG2,-nG2,-nG2);
  IntVector high_filter2 = patch->getCellHighIndex() + IntVector(nG2,nG2,nG2);
  Uintah::BlockRange range2(low_filter2, high_filter2 );
  Uintah::BlockRange range1(low_filter, high_filter );
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex( ));


  CCVariable<double>& IsI = tsk_info->get_uintah_field_add< CCVariable<double> >(m_IsI_name,nGhosts2 );
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

  IsI.initialize(0.0);
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

  Uintah::ArchesCore::computeIsInsij get_IsIsij( IsI, s11, s22, s33, s12, s13, s23,
                                                 uVel, vVel, wVel,
                                                 CCuVel, CCvVel, CCwVel, Dx);
  Uintah::parallel_for(range2,get_IsIsij);

  Uintah::parallel_for( range2, [&](int i, int j, int k){
    Beta11(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s11(i,j,k);
    Beta22(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s22(i,j,k);
    Beta33(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s33(i,j,k);
    Beta12(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s12(i,j,k);
    Beta13(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s13(i,j,k);
    Beta23(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s23(i,j,k);
  });

  ArchesCore::BCFilter bcfilter;
  bcfilter.apply_zero_neumann(patch,Beta11,vol_fraction);
  bcfilter.apply_zero_neumann(patch,Beta22,vol_fraction);
  bcfilter.apply_zero_neumann(patch,Beta33,vol_fraction);
  bcfilter.apply_zero_neumann(patch,Beta12,vol_fraction);
  bcfilter.apply_zero_neumann(patch,Beta13,vol_fraction);
  bcfilter.apply_zero_neumann(patch,Beta23,vol_fraction);
  // Filter rho
  CCVariable<double>& filterRho = tsk_info->get_uintah_field_add< CCVariable<double> >("Filterrho", nGhosts2);
  filterRho.initialize(0.0);
  // this need to be fixed
  //filterRho.copy(ref_rho);

  CCVariable<double>& rhoBC = tsk_info->get_uintah_field_add< CCVariable<double> >("rhoBC",nGrho);

  rhoBC.copy(rho);
  bcfilter.apply_BC_rho(patch,rhoBC,rho,vol_fraction);
  m_Filter.applyFilter(rhoBC,filterRho,range2,vol_fraction);

  //m_Filter.applyFilter(rho,filterRho,range1,vol_fraction);

  // filter rho*ux...
  SFCXVariable<double>& filterRhoU = tsk_info->get_uintah_field_add< SFCXVariable<double> >("Filterrhou", nGhosts2);
  SFCYVariable<double>& filterRhoV = tsk_info->get_uintah_field_add< SFCYVariable<double> >("Filterrhov", nGhosts2);
  SFCZVariable<double>& filterRhoW = tsk_info->get_uintah_field_add< SFCZVariable<double> >("Filterrhow", nGhosts2);
  filterRhoU.initialize(0.0);
  filterRhoV.initialize(0.0);
  filterRhoW.initialize(0.0);

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;

  IntVector low;
  if ( xminus ){
    low  = patch->getCellLowIndex()+IntVector(1,0,0) + IntVector(-nG2,-nG2,-nG2);
  }else{
    low  = patch->getCellLowIndex()+ IntVector(-nG2,-nG2,-nG2);
  }

  Uintah::BlockRange range_u(low, high_filter2);
  m_Filter.applyFilter(uVel,filterRhoU,rho,vol_fraction,range_u);
  if ( yminus ){
    low = patch->getCellLowIndex()+IntVector(0,1,0) + IntVector(-nG2,-nG2,-nG2);
  } else {
    low = patch->getCellLowIndex()+ IntVector(-nG2,-nG2,-nG2);
  }

  Uintah::BlockRange range_v(low, high_filter2);
  m_Filter.applyFilter(vVel,filterRhoV,rho,vol_fraction,range_v);

  if ( zminus ){
    low = patch->getCellLowIndex()+IntVector(0,0,1)+ IntVector(-nG2,-nG2,-nG2);
  } else {
    low = patch->getCellLowIndex()+ IntVector(-nG2,-nG2,-nG2);
  }
  Uintah::BlockRange range_w(low, high_filter2);
  m_Filter.applyFilter(wVel,filterRhoW,rho,vol_fraction,range_w);


  bcfilter.apply_BC_rhou(patch,filterRhoU,uVel,rho,vol_fraction);
  bcfilter.apply_BC_rhou(patch,filterRhoV,vVel,rho,vol_fraction);
  bcfilter.apply_BC_rhou(patch,filterRhoW,wVel,rho,vol_fraction);
  bcfilter.apply_BC_filter_rho(patch,filterRho,rhoBC,vol_fraction);
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
  bcfilter.apply_zero_neumann(patch,rhoUU,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoVV,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoWW,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoUV,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoUW,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoVW,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoV,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoU,vol_fraction);
  bcfilter.apply_zero_neumann(patch,rhoW,vol_fraction);
}

} //namespace Uintah
