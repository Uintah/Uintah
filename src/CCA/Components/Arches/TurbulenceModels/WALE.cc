#include <CCA/Components/Arches/TurbulenceModels/WALE.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

typedef ArchesFieldContainer AFC;

//---------------------------------------------------------------------------------
WALE::WALE( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//---------------------------------------------------------------------------------
WALE::~WALE()
{}

//---------------------------------------------------------------------------------
void
WALE::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  Nghost_cells = 1;

  m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );

  m_cc_u_vel_name = m_u_vel_name + "_cc";
  m_cc_v_vel_name = m_v_vel_name + "_cc";
  m_cc_w_vel_name = m_w_vel_name + "_cc";

  db->getWithDefault("Cs", m_Cs, .5);

  if (db->findBlock("use_my_name_viscosity")){
    db->findBlock("use_my_name_viscosity")->getAttribute("label",m_t_vis_name);
  } else{
    m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY, db );
  }

  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
    params_root->findBlock("PhysicalConstants")->require("viscosity",
                                                          m_molecular_visc);
    if( m_molecular_visc == 0 ) {
      std::stringstream msg;
      msg << "ERROR: Constant WALE: problemSetup(): Zero viscosity specified \n"
          << "       in <PhysicalConstants> section of input file." << std::endl;
      throw InvalidValue(msg.str(),__FILE__,__LINE__);
    }
  } else {
    std::stringstream msg;
    msg << "ERROR: Constant WALE: problemSetup(): Missing <PhysicalConstants> \n"
        << "       section in input file!" << std::endl;
    throw InvalidValue(msg.str(),__FILE__,__LINE__);
  }
}

//---------------------------------------------------------------------------------
void
WALE::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_t_vis_name);

}

//---------------------------------------------------------------------------------
void
WALE::register_initialize( std::vector<AFC::VariableInformation>&
                                       variable_registry , const bool packed_tasks){

  register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

}

//---------------------------------------------------------------------------------
void
WALE::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  mu_sgc.initialize(0.0);

}

//---------------------------------------------------------------------------------
void
WALE::register_timestep_init( std::vector<AFC::VariableInformation>&
                              variable_registry , const bool packed_tasks){

  register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

}

//---------------------------------------------------------------------------------
void
WALE::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));

}

//---------------------------------------------------------------------------------
void
WALE::register_timestep_eval( std::vector<AFC::VariableInformation>&
                              variable_registry, const int time_substep , const bool packed_tasks){

  register_variable( m_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);

  register_variable( m_cc_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);

  register_variable( m_t_vis_name, AFC::MODIFIES ,  variable_registry, time_substep );

}

//---------------------------------------------------------------------------------
void
WALE::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& uVel = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(m_w_vel_name);

  constCCVariable<double>& CCuVel = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_cc_u_vel_name);
  constCCVariable<double>& CCvVel = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_cc_v_vel_name);
  constCCVariable<double>& CCwVel = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_cc_w_vel_name);

  CCVariable<double>& mu_sgc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_t_vis_name);

  const Vector Dx = patch->dCell();
  const double delta = pow(Dx.x()*Dx.y()*Dx.z(),1./3.);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

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

    const double s11 = (uep-uwp)/Dx.x();
    const double s22 = (vnp-vsp)/Dx.y();
    const double s33 = (wtp-wbp)/Dx.z();
    const double s12 = 0.50 * ((unp-usp)/Dx.y() + (vep-vwp)/Dx.x());
    const double s13 = 0.50 * ((utp-ubp)/Dx.z() + (wep-wwp)/Dx.x());
    const double s23 = 0.50 * ((vtp-vbp)/Dx.z() + (wnp-wsp)/Dx.y());

    const double du11 = (uep-uwp)/Dx.x(); // dudx
    const double du12 = (unp-usp)/Dx.y(); //dudy
    const double du21 = (vep-vwp)/Dx.x(); //dvdx
    const double du13 = (utp-ubp)/Dx.z(); //dudz
    const double du31 = (wep-wwp)/Dx.x(); //dwdx
    const double du22 = (vnp-vsp)/Dx.y(); //dvdy
    const double du23 = (vtp-vbp)/Dx.z(); //dvdz
    const double du32 = (wnp-wsp)/Dx.y(); //dwdy
    const double du33 = (wtp-wbp)/Dx.z(); //dwdz

    const double s11d = du11*du11 + du12*du21 + du13*du31  ;
    const double s12d = 0.5*(du11*du12 + du12*du22 + du13*du32 + du21*du11 + du22*du21 + du23*du31);
    const double s13d = 0.5*(du11*du13 + du12*du23 + du13*du33 + du31*du11 + du32*du21 + du33*du31);
    const double s22d = du21*du12 + du22*du22 + du23*du32  ;
    const double s23d = 0.5*(du21*du13 + du22*du23 + du23*du33 + du31*du12 + du32*du22 + du33*du32);
    const double s33d = du31*du13 + du32*du23 + du33*du33  ;

    const double SijdSijd = (s11d*s11d + s22d*s22d + s33d*s33d
                 + 2.0*s12d*s12d + 2.0*s13d*s13d + 2.0*s23d*s23d);

    const double SijSij =( s11*s11 + s22*s22 + s33*s33
                 + 2.0*s12*s12 + 2.0*s13*s13 + 2.0*s23*s23);

    const double fvis = pow(SijdSijd,1.5)/(pow(SijSij,2.5) + pow(SijdSijd,5./4.));

    mu_sgc(i,j,k) = pow(m_Cs*delta,2.0)*fvis + m_molecular_visc; // I need to times density
  });

}
} //namespace Uintah
