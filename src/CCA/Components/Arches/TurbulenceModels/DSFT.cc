#include <CCA/Components/Arches/TurbulenceModels/DSFT.h>
#include <math.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
DSFT::DSFT( std::string task_name, int matl_index, const std::string turb_model_name ) :
TaskInterface( task_name, matl_index ), m_turb_model_name(turb_model_name) {

}

//--------------------------------------------------------------------------------------------------
DSFT::~DSFT(){
}

//--------------------------------------------------------------------------------------------------
void
DSFT::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;
  // u, v , w velocities
  m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
  m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
  m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );
  m_cc_u_vel_name = parse_ups_for_role( CCUVELOCITY_ROLE, db, m_u_vel_name + "_cc" );
  m_cc_v_vel_name = parse_ups_for_role( CCVVELOCITY_ROLE, db, m_v_vel_name + "_cc" );
  m_cc_w_vel_name = parse_ups_for_role( CCWVELOCITY_ROLE, db, m_w_vel_name + "_cc" );

  m_density_name = parse_ups_for_role( DENSITY_ROLE, db, "density" );

  m_rhou_vel_name = default_uMom_name;
  m_rhov_vel_name = default_vMom_name;
  m_rhow_vel_name = default_wMom_name;


  m_volFraction_name = "volFraction";

  std::string m_Type_filter_name;
  db->findBlock("filter")->getAttribute("type",m_Type_filter_name);

  std::stringstream composite_name;
  composite_name << "strainMagnitude_" << m_turb_model_name;
  m_IsI_name = composite_name.str();

  //** HACK **//
  if (m_u_vel_name == "uVelocitySPBC") { // this is production code
    m_create_labels_IsI_t_viscosity = false;
    m_IsI_name = "strainMagnitudeLabel";
  }

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

  register_new_variable<CCVariable<double> >( "Filterrho" );
  register_new_variable<SFCXVariable<double> >( "Filterrhou" );
  register_new_variable<SFCYVariable<double> >( "Filterrhov" );
  register_new_variable<SFCZVariable<double> >( "Filterrhow" );
  register_new_variable<CCVariable<double> >( "rhoUU" );
  register_new_variable<CCVariable<double> >( "rhoVV" );
  register_new_variable<CCVariable<double> >( "rhoWW" );
  register_new_variable<CCVariable<double> >( "rhoUV" );
  register_new_variable<CCVariable<double> >( "rhoUW" );
  register_new_variable<CCVariable<double> >( "rhoVW" );
  register_new_variable<CCVariable<double> >( "rhoU" );
  register_new_variable<CCVariable<double> >( "rhoV" );
  register_new_variable<CCVariable<double> >( "rhoW" );
  register_new_variable<CCVariable<double> >( "rhoBC" );

}

//--------------------------------------------------------------------------------------------------
void
DSFT::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry , const bool packed_tasks){


  typedef ArchesFieldContainer AFC;
  int nG = 1;
  int nGrho = nG + 1;

  register_variable( m_u_vel_name, AFC::REQUIRES, nG, AFC::NEWDW, variable_registry);
  register_variable( m_v_vel_name, AFC::REQUIRES, nG, AFC::NEWDW, variable_registry);
  register_variable( m_w_vel_name, AFC::REQUIRES, nG, AFC::NEWDW, variable_registry);
  register_variable( m_density_name, AFC::REQUIRES, nGrho, AFC::NEWDW, variable_registry);
  register_variable( m_volFraction_name, AFC::REQUIRES, nGrho, AFC::NEWDW, variable_registry);

  register_variable( m_cc_u_vel_name, AFC::REQUIRES, nG, AFC::NEWDW, variable_registry);
  register_variable( m_cc_v_vel_name, AFC::REQUIRES, nG, AFC::NEWDW, variable_registry);
  register_variable( m_cc_w_vel_name, AFC::REQUIRES, nG, AFC::NEWDW, variable_registry);

  register_variable( "rhoBC",    AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_IsI_name, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Beta11",   AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Beta12",   AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Beta13",   AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Beta22",   AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Beta23",   AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Beta33",   AFC::COMPUTES, variable_registry, m_task_name );

  register_variable( "s11", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "s12", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "s13", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "s22", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "s23", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "s33", AFC::COMPUTES, variable_registry, m_task_name );

  register_variable( "Filterrho",  AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Filterrhou", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Filterrhov", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "Filterrhow", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoUU",      AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoVV",      AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoWW",      AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoUV",      AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoUW",      AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoVW",      AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoU",       AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoV",       AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "rhoW",       AFC::COMPUTES, variable_registry, m_task_name );
}

//--------------------------------------------------------------------------------------------------
void
DSFT::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& filterRho = tsk_info->get_field< CCVariable<double> >("Filterrho");
  SFCXVariable<double>& filterRhoU = tsk_info->get_field< SFCXVariable<double> >("Filterrhou");
  SFCYVariable<double>& filterRhoV = tsk_info->get_field< SFCYVariable<double> >("Filterrhov");
  SFCZVariable<double>& filterRhoW = tsk_info->get_field< SFCZVariable<double> >("Filterrhow");
  filterRho.initialize(0.0);
  filterRhoU.initialize(0.0);
  filterRhoV.initialize(0.0);
  filterRhoW.initialize(0.0);

  computeModel(patch, tsk_info);

}

//--------------------------------------------------------------------------------------------------
void
DSFT::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
                              const int time_substep,
                              const bool packed_tasks){
  int nG = 1;
  if (packed_tasks ){
   nG = 3;
  }
  int nGrho = nG + 1;

  typedef ArchesFieldContainer AFC;

  register_variable( m_u_vel_name, AFC::REQUIRES, nG, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_v_vel_name, AFC::REQUIRES, nG, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_w_vel_name, AFC::REQUIRES, nG, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_density_name, AFC::REQUIRES, nGrho, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_volFraction_name, AFC::REQUIRES, nGrho, AFC::LATEST, variable_registry, time_substep );

  register_variable( m_cc_u_vel_name, AFC::REQUIRES, nG, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, AFC::REQUIRES, nG, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, AFC::REQUIRES, nG, AFC::LATEST, variable_registry, time_substep);

  register_variable( "rhoBC",    AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( m_IsI_name, AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "Beta11",   AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "Beta12",   AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "Beta13",   AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "Beta22",   AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "Beta23",   AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "Beta33",   AFC::COMPUTES, variable_registry, time_substep , m_task_name );

  register_variable( "s11", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "s12", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "s13", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "s22", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "s23", AFC::COMPUTES, variable_registry, time_substep , m_task_name );
  register_variable( "s33", AFC::COMPUTES, variable_registry, time_substep , m_task_name );

  register_variable( "Filterrho",  AFC::COMPUTES,  variable_registry, time_substep, m_task_name );
  register_variable( "Filterrhou", AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "Filterrhov", AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "Filterrhow", AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoUU",      AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoVV",      AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoWW",      AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoUV",      AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoUW",      AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoVW",      AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoU",       AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoV",       AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( "rhoW",       AFC::COMPUTES, variable_registry, time_substep, m_task_name );

}

//--------------------------------------------------------------------------------------------------
void
DSFT::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  this->computeModel(patch, tsk_info);
}

//--------------------------------------------------------------------------------------------------
void
DSFT::computeModel( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = tsk_info->get_field<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_field<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_field<constSFCZVariable<double> >(m_w_vel_name);
  constCCVariable<double>& rho = tsk_info->get_field<constCCVariable<double> >(m_density_name);
  constCCVariable<double>& vol_fraction = tsk_info->get_field<constCCVariable<double> >(m_volFraction_name);
  constCCVariable<double>& CCuVel = tsk_info->get_field<constCCVariable<double> >(m_cc_u_vel_name);
  constCCVariable<double>& CCvVel = tsk_info->get_field<constCCVariable<double> >(m_cc_v_vel_name);
  constCCVariable<double>& CCwVel = tsk_info->get_field<constCCVariable<double> >(m_cc_w_vel_name);

  const Vector Dx = patch->dCell();
  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  CCVariable<double>& IsI = tsk_info->get_field< CCVariable<double> >( m_IsI_name );
  CCVariable<double>& s11 = tsk_info->get_field< CCVariable<double> >( "s11" );
  CCVariable<double>& s12 = tsk_info->get_field< CCVariable<double> >( "s12" );
  CCVariable<double>& s13 = tsk_info->get_field< CCVariable<double> >( "s13" );
  CCVariable<double>& s22 = tsk_info->get_field< CCVariable<double> >( "s22" );
  CCVariable<double>& s23 = tsk_info->get_field< CCVariable<double> >( "s23" );
  CCVariable<double>& s33 = tsk_info->get_field< CCVariable<double> >( "s33" );

  CCVariable<double>& Beta11 = tsk_info->get_field< CCVariable<double> >( "Beta11" );
  CCVariable<double>& Beta12 = tsk_info->get_field< CCVariable<double> >( "Beta12" );
  CCVariable<double>& Beta13 = tsk_info->get_field< CCVariable<double> >( "Beta13" );
  CCVariable<double>& Beta22 = tsk_info->get_field< CCVariable<double> >( "Beta22" );
  CCVariable<double>& Beta23 = tsk_info->get_field< CCVariable<double> >( "Beta23" );
  CCVariable<double>& Beta33 = tsk_info->get_field< CCVariable<double> >( "Beta33" );

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

  Uintah::parallel_for( range, [&](int i, int j, int k){

    double uep = 0.0;
    double uwp = 0.0;
    double vep = 0.0;
    double vwp = 0.0;
    double wep = 0.0;
    double wwp = 0.0;
    double unp = 0.0;
    double usp = 0.0;
    double vnp = 0.0;
    double vsp = 0.0;
    double wnp = 0.0;
    double wsp = 0.0;
    double utp = 0.0;
    double ubp = 0.0;
    double vtp = 0.0;
    double vbp = 0.0;
    double wtp = 0.0;
    double wbp = 0.0;

    // x-dir
    {
      STENCIL3_1D(0);
      uep = uVel(IJK_P_);
      uwp = uVel(IJK_);

      vep = 0.50 * CCvVel(IJK_P_);
      vwp = 0.50 * CCvVel(IJK_M_);

      wep = 0.50 * CCwVel(IJK_P_);
      wwp = 0.50 * CCwVel(IJK_M_);
    }

    // y-dir
    {
      STENCIL3_1D(1);
      unp = 0.50 * CCuVel(IJK_P_);
      usp = 0.50 * CCuVel(IJK_M_);

      vnp = vVel(IJK_P_);
      vsp = vVel(IJK_);

      wnp = 0.50 * CCwVel(IJK_P_);
      wsp = 0.50 * CCwVel(IJK_M_);
    }

    // z-dir
    {
      STENCIL3_1D(2);
      utp = 0.50 * CCuVel(IJK_P_);
      ubp = 0.50 * CCuVel(IJK_M_);

      vtp = 0.50 * CCvVel(IJK_P_);
      vbp = 0.50 * CCvVel(IJK_M_);

      wtp = wVel(IJK_P_);
      wbp = wVel(IJK_);
    }

    s11(i,j,k) = (uep-uwp)/Dx.x();
    s22(i,j,k) = (vnp-vsp)/Dx.y();
    s33(i,j,k) = (wtp-wbp)/Dx.z();
    s12(i,j,k) = 0.50 * ((unp-usp)/Dx.y() + (vep-vwp)/Dx.x());
    s13(i,j,k) = 0.50 * ((utp-ubp)/Dx.z() + (wep-wwp)/Dx.x());
    s23(i,j,k) = 0.50 * ((vtp-vbp)/Dx.z() + (wnp-wsp)/Dx.y());

    IsI(i,j,k) = 2.0 * ( s11(i,j,k)*s11(i,j,k) + s22(i,j,k)*s22(i,j,k) + s33(i,j,k)*s33(i,j,k)
               + 2.0 * ( s12(i,j,k)*s12(i,j,k) + s13(i,j,k)*s13(i,j,k) + s23(i,j,k)*s23(i,j,k) ) );

    IsI(i,j,k) = sqrt( IsI(i,j,k) );

  });

  Uintah::parallel_for( range, [&](int i, int j, int k){
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
  CCVariable<double>& filterRho = tsk_info->get_field< CCVariable<double> >("Filterrho");
  filterRho.initialize(0.0);


  // this need to be fixed (??)
  //filterRho.copy(ref_rho);

  CCVariable<double>& rhoBC = tsk_info->get_field< CCVariable<double> >("rhoBC");

  rhoBC.initialize(0.0);
  Uintah::parallel_for(range, [&](int i, int j, int k){
      rhoBC(i,j,k) = rho(i,j,k);
  });
  bcfilter.apply_BC_rho( patch, rhoBC, rho, vol_fraction);
  m_Filter.applyFilter( rho, filterRho, range, vol_fraction);

  //m_Filter.applyFilter(rho,filterRho,range,vol_fraction);

  // filter rho*ux...
  SFCXVariable<double>& filterRhoU = tsk_info->get_field< SFCXVariable<double> >("Filterrhou");
  SFCYVariable<double>& filterRhoV = tsk_info->get_field< SFCYVariable<double> >("Filterrhov");
  SFCZVariable<double>& filterRhoW = tsk_info->get_field< SFCZVariable<double> >("Filterrhow");
  filterRhoU.initialize(0.0);
  filterRhoV.initialize(0.0);
  filterRhoW.initialize(0.0);

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;

  IntVector low;
  if ( xminus ){
    low  = patch->getCellLowIndex()+IntVector(1,0,0);
  }else{
    low  = patch->getCellLowIndex();
  }

  Uintah::BlockRange range_u(low, patch->getCellHighIndex() );
  m_Filter.applyFilter(uVel,filterRhoU,rho,vol_fraction,range_u);
  if ( yminus ){
    low = patch->getCellLowIndex()+IntVector(0,1,0);
  } else {
    low = patch->getCellLowIndex();
  }

  Uintah::BlockRange range_v(low, patch->getCellHighIndex() );
  m_Filter.applyFilter(vVel,filterRhoV,rho,vol_fraction,range_v);

  if ( zminus ){
    low = patch->getCellLowIndex()+IntVector(0,0,1);
  } else {
    low = patch->getCellLowIndex();
  }
  Uintah::BlockRange range_w(low, patch->getCellHighIndex() );
  m_Filter.applyFilter(wVel,filterRhoW,rho,vol_fraction,range_w);

  bcfilter.apply_BC_rhou(patch,filterRhoU,uVel,rho,vol_fraction);
  bcfilter.apply_BC_rhou(patch,filterRhoV,vVel,rho,vol_fraction);
  bcfilter.apply_BC_rhou(patch,filterRhoW,wVel,rho,vol_fraction);
  bcfilter.apply_BC_filter_rho(patch,filterRho,rhoBC,vol_fraction);

  // Compute rhouiuj at cc
  CCVariable<double>& rhoUU = tsk_info->get_field< CCVariable<double> >("rhoUU");
  CCVariable<double>& rhoVV = tsk_info->get_field< CCVariable<double> >("rhoVV");
  CCVariable<double>& rhoWW = tsk_info->get_field< CCVariable<double> >("rhoWW");
  CCVariable<double>& rhoUV = tsk_info->get_field< CCVariable<double> >("rhoUV");
  CCVariable<double>& rhoUW = tsk_info->get_field< CCVariable<double> >("rhoUW");
  CCVariable<double>& rhoVW = tsk_info->get_field< CCVariable<double> >("rhoVW");
  CCVariable<double>& rhoU = tsk_info->get_field< CCVariable<double> >("rhoU");
  CCVariable<double>& rhoV = tsk_info->get_field< CCVariable<double> >("rhoV");
  CCVariable<double>& rhoW = tsk_info->get_field< CCVariable<double> >("rhoW");

  rhoUU.initialize(0.0);
  rhoVV.initialize(0.0);
  rhoWW.initialize(0.0);
  rhoUV.initialize(0.0);
  rhoUW.initialize(0.0);
  rhoVW.initialize(0.0);
  rhoU.initialize(0.0);
  rhoV.initialize(0.0);
  rhoW.initialize(0.0);

  Uintah::parallel_for( range, [&](int i, int j, int k){
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
