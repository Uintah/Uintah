#include <CCA/Components/Arches/TurbulenceModels/SGSsigma.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <stdio.h>
#include <math.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

// The code is to use the singular value to build a subgrid-scale model for computing the subgrid viscosity, also called the sigma model -- Franck Nicoud-2011

using namespace Uintah;
using namespace std;

//--------------------------------------------------------------------------------------------------
SGSsigma::SGSsigma( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){

   //hard coded velocity names:
  //m_u_vel_name = "uVelocitySPBC";
  //m_v_vel_name = "vVelocitySPBC";
  //m_w_vel_name = "wVelocitySPBC";
  //m_sigOper="sigmaOperator";

}

//--------------------------------------------------------------------------------------------------
SGSsigma::~SGSsigma(){
}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::problemSetup( ProblemSpecP& db ){

  m_PI = acos(-1.0);
  using namespace Uintah::ArchesCore;

  Nghost_cells = 1;

  m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );
  m_density_name     = parse_ups_for_role( DENSITY, db, "density" );

  m_cc_u_vel_name = parse_ups_for_role( CCUVELOCITY, db, "CCUVelocity" );//;m_u_vel_name + "_cc";
  m_cc_v_vel_name = parse_ups_for_role( CCVVELOCITY, db, "CCVVelocity" );//m_v_vel_name + "_cc";
  m_cc_w_vel_name = parse_ups_for_role( CCWVELOCITY, db, "CCWVelocity" );;//m_w_vel_name + "_cc";

  m_IsI_name = "strainMagnitudeLabel";
  m_turb_viscosity_name = "turb_viscosity";
  m_volFraction_name = "volFraction";

  db->getWithDefault("Cs", m_Cs, 1.5 );

  if (db->findBlock("use_my_name_viscosity")){
    db->findBlock("use_my_name_viscosity")->getAttribute("label",m_t_vis_name);
  } else{
    m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY, db, "viscosityCTS" );
  }

  if (m_u_vel_name == "uVelocitySPBC") { // this is production code
    m_create_labels_IsI_t_viscosity = false;
  }

  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
    params_root->findBlock("PhysicalConstants")->require("viscosity",
                                                          m_molecular_visc);
    if( m_molecular_visc == 0 ) {
      std::stringstream msg;
       msg << "ERROR: Constant Sigma: problemSetup(): Zero viscosity specified \n"
          << "       in <PhysicalConstants> section of input file." << std::endl;
      throw InvalidValue(msg.str(),__FILE__,__LINE__);
    }
  } else {
    std::stringstream msg;
    msg << "ERROR: Constant Sigma: problemSetup(): Missing <PhysicalConstants> \n"
        << "       section in input file!" << std::endl;
    throw InvalidValue(msg.str(),__FILE__,__LINE__);
  }

  //if (params_root->findBlock("PhysicalConstants")) {
    //ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    //db_phys->require("viscosity", m_visc);
    //if( m_visc == 0 ) {
      //throw InvalidValue("ERROR: SGSsigma: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    //}
  //} else {
    //throw InvalidValue("ERROR: SGSsigma: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
  //}

}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::create_local_labels(){

  if (m_create_labels_IsI_t_viscosity) {
    register_new_variable<CCVariable<double> >(m_IsI_name);
    register_new_variable<CCVariable<double> >( m_t_vis_name);
    register_new_variable<CCVariable<double> >( m_turb_viscosity_name);
  }

 //register_new_variable<CCVariable<double> >( m_sigOper );

}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks ){

  register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );
  register_variable( m_turb_viscosity_name, AFC::COMPUTES, variable_registry );
  //register_variable( m_sigOper, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info){

  CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
  CCVariable<double>& mu_turb = *(tsk_info->get_uintah_field<CCVariable<double> >(m_turb_viscosity_name));
  mu_sgc.initialize(0.0);
  mu_turb.initialize(0.0);

  //CCVariable<double>& sigOper =  tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigOper);
  //Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  //Uintah::parallel_for( range, [&](int i, int j, int k){
                       //sigOper(i,j,k)=0.0;
  //});

}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::register_timestep_init(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks ){

  //register_variable( m_sigOper, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info){

  //CCVariable<double>& sigOper =  tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigOper);
  //Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  //Uintah::parallel_for( range, [&](int i, int j, int k){

    //sigOper(i,j,k)=0.0;

  //});
}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES,Nghost_cells , ArchesFieldContainer::NEWDW, variable_registry);
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES,Nghost_cells , ArchesFieldContainer::NEWDW, variable_registry);
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES,Nghost_cells , ArchesFieldContainer::NEWDW, variable_registry);
  //register_variable( "CsLabel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry);
  register_variable( m_cc_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_density_name, AFC::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_volFraction_name, AFC::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  //register_variable( "CCVelocity",      ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry);
  //register_variable( "density",         ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
  //register_variable( "volFraction",     ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( m_IsI_name, ArchesFieldContainer::COMPUTES ,  variable_registry, time_substep , m_task_name, packed_tasks);
  if (m_create_labels_IsI_t_viscosity) {
    register_variable( m_t_vis_name, AFC::COMPUTES ,  variable_registry, time_substep );
    register_variable( m_turb_viscosity_name, AFC::COMPUTES ,  variable_registry, time_substep );
  } else {
    register_variable( m_t_vis_name, AFC::MODIFIES ,  variable_registry, time_substep );
    register_variable( m_turb_viscosity_name, AFC::MODIFIES ,  variable_registry, time_substep );
  }
  //register_variable( "viscosityCTS",    ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  //register_variable( "turb_viscosity",  ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  //register_variable( m_sigOper,         ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
SGSsigma::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info){

  Vector Dx=patch->dCell();
  double dx=Dx.x(); double dy=Dx.y(); double dz=Dx.z();
  double filter = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
  double filter2 = filter*filter;

  constSFCXVariable<double>& uVel = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(m_w_vel_name);
  //constCCVariable<double>& Cs_dynamic = tsk_info->get_const_uintah_field_add<constCCVariable<double> >("CsLabel");
  constCCVariable<double>& CCuVel = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_cc_u_vel_name);
  constCCVariable<double>& CCvVel = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_cc_v_vel_name);
  constCCVariable<double>& CCwVel = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_cc_w_vel_name);

  //constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >("volFraction");
  //constCCVariable<double>& Density_sigma = tsk_info->get_const_uintah_field_add<constCCVariable<double> >("density");
  //constCCVariable<Vector>& CCVelocity =    tsk_info->get_const_uintah_field_add<constCCVariable<Vector> >("CCVelocity");

  CCVariable<double>& mu_sgc = tsk_info->get_uintah_field_add<CCVariable<double> >(m_t_vis_name);
  CCVariable<double>& mu_turb = *(tsk_info->get_uintah_field<CCVariable<double> >(m_turb_viscosity_name));
  CCVariable<double>& IsI = tsk_info->get_uintah_field_add< CCVariable<double> >(m_IsI_name);
  constCCVariable<double>& Density_sigma = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));
  constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);
  //CCVariable<double>& viscosity_new = tsk_info->get_uintah_field_add<CCVariable<double> >("viscosityCTS");
  //CCVariable<double>&  TurbViscosity_new = tsk_info->get_uintah_field_add<CCVariable<double> >("turb_viscosity");
  //CCVariable<double>& sigOper = tsk_info->get_uintah_field_add<CCVariable<double> >(m_sigOper);

  double VelgUx, VelgUy, VelgUz, VelgVx, VelgVy, VelgVz, VelgWx, VelgWy, VelgWz;
  double G11,G12,G13,G21,G22,G23,G31,G32,G33;
  double trG,trG2;
  double I1,I2,I3;
  double alpha1,alpha2,alpha3;
  double sigma1,sigma2,sigma3;
  //double mu=0.0;
  double sigOper=0.0;
  //mu=m_visc;

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  Uintah::parallel_for( range, [&](int i, int j, int k){
    // face velocity of fourth order
    //VelgUx=(-2.0*uVel(i+2,j,k)      +12.0*uVel(i+1,j,k)        -6.0*uVel(i,j,k)            - 4.0*uVel(i-1,j,k))/dx/12.0;
    //VelgUy=(-CCVelocity(i,j+2,k)[0] +8.0*CCVelocity(i,j+1,k)[0]-8.0*CCVelocity(i,j-1,k)[0] + CCVelocity(i,j-2,k)[0])/dy/12.0;
    //VelgUz=(-CCVelocity(i,j,k+2)[0] +8.0*CCVelocity(i,j,k+1)[0]-8.0*CCVelocity(i,j,k-1)[0] + CCVelocity(i,j,k-2)[0])/dz/12.0;

    //VelgVx=(-CCVelocity(i+2,j,k)[1] +8.0*CCVelocity(i+1,j,k)[1]-8.0*CCVelocity(i-1,j,k)[1] + CCVelocity(i-2,j,k)[1])/dx/12.0;
    //VelgVy=(-2.0*vVel(i,j+2,k)      +12.0*vVel(i,j+1,k)        -6.0*vVel(i,j,k)            -4.0*vVel(i,j-1,k))/dy/12.0;
    //VelgVz=(-CCVelocity(i,j,k+2)[1] +8.0*CCVelocity(i,j,k+1)[1]-8.0*CCVelocity(i,j,k-1)[1] + CCVelocity(i,j,k-2)[1])/dz/12.0;

    //VelgWx=(-CCVelocity(i+2,j,k)[2] +8.0*CCVelocity(i+1,j,k)[2]-8.0*CCVelocity(i-1,j,k)[2] + CCVelocity(i-2,j,k)[2])/dx/12.0;
    //VelgWy=(-CCVelocity(i,j+2,k)[2] +8.0*CCVelocity(i,j+1,k)[2]-8.0*CCVelocity(i,j-1,k)[2] + CCVelocity(i,j-2,k)[2])/dy/12.0;
    //VelgWz=(-2.0*wVel(i,j,k+2)      +12.0*wVel(i,j,k+1)        -6.0*wVel(i,j,k)            -4.0*wVel(i,j,k-1))/dz/12.0;

    // face velocity of second order-most reliable
    //VelgUx=(uVel(i+1,j,k) - uVel(i,j,k))/dx;
    //VelgUy=(CCVelocity(i,j+1,k)[0] - CCVelocity(i,j-1,k)[0])/dy/2.0;
    //VelgUz=(CCVelocity(i,j,k+1)[0] - CCVelocity(i,j,k-1)[0])/dz/2.0;

    //VelgVx=(CCVelocity(i+1,j,k)[1] - CCVelocity(i-1,j,k)[1])/dx/2.0;
    //VelgVy=(vVel(i,j+1,k) - vVel(i,j,k))/dy;
    //VelgVz=(CCVelocity(i,j,k+1)[1] - CCVelocity(i,j,k-1)[1])/dz/2.0;

    //VelgWx=(CCVelocity(i+1,j,k)[2] - CCVelocity(i-1,j,k)[2])/dx/2.0;
    //VelgWy=(CCVelocity(i,j+1,k)[2] - CCVelocity(i,j-1,k)[2])/dy/2.0;
    //VelgWz=(wVel(i,j,k+1)-wVel(i,j,k))/dz;

    // Consider the kokkos interface and old interface
    VelgUx=(uVel(i+1,j,k) - uVel(i,j,k))/dx;
    VelgUy=(CCuVel(i,j+1,k) - CCuVel(i,j-1,k))/dy/2.0;
    VelgUz=(CCuVel(i,j,k+1) - CCuVel(i,j,k-1))/dz/2.0;

    VelgVx=(CCvVel(i+1,j,k) - CCvVel(i-1,j,k))/dx/2.0;
    VelgVy=(vVel(i,j+1,k) - vVel(i,j,k))/dy;
    VelgVz=(CCvVel(i,j,k+1) - CCvVel(i,j,k-1))/dz/2.0;

    VelgWx=(CCwVel(i+1,j,k) - CCwVel(i-1,j,k))/dx/2.0;
    VelgWy=(CCwVel(i,j+1,k) - CCwVel(i,j-1,k))/dy/2.0;
    VelgWz=(wVel(i,j,k+1)-wVel(i,j,k))/dz;


    G11=VelgUx*VelgUx+VelgVx*VelgVx+VelgWx*VelgWx;
    G12=VelgUx*VelgUy+VelgVx*VelgVy+VelgWx*VelgWy;
    G13=VelgUx*VelgUz+VelgVx*VelgVz+VelgWx*VelgWz;

    G21=G12;
    G22=VelgUy*VelgUy+VelgVy*VelgVy+VelgWy*VelgWy;
    G23=VelgUy*VelgUz+VelgVy*VelgVz+VelgWy*VelgWz;

    G31=G13;
    G32=G23;
    G33=VelgUz*VelgUz+VelgVz*VelgVz+VelgWz*VelgWz;
    trG=G11+G22+G33;

    //-----since G is symmetric
    trG2=G11*G11+G12*G12+G13*G13+G21*G21+G22*G22+G23*G23+G31*G31+G32*G32+G33*G33;

    I1=G11+G22+G33;
    I2=0.5*(trG*trG-trG2);
    I3=G11*G22*G33+G12*G23*G31+G21*G32*G13-G13*G22*G31-G23*G32*G11-G12*G21*G33;

    alpha1=I1*I1/9-I2/3;
    alpha2=I1*I1*I1/27-I1*I2/6+I3/2;
    alpha3=acos(alpha2/pow(alpha1,1.5))/3;

    sigma1=pow(I1/3+2*sqrt(alpha1)*cos(alpha3),0.5);
    sigma2=pow(I1/3-2*sqrt(alpha1)*cos(m_PI/3+alpha3),0.5);
    sigma3=pow(I1/3-2*sqrt(alpha1)*cos(m_PI/3-alpha3),0.5);

    sigOper=std::max( 1e-30, sigma3*(sigma1-sigma2)*(sigma2-sigma3)/(sigma1*sigma1) );

    const double s11= VelgUx;
    const double s22= VelgVy;
    const double s33= VelgWz;
    const double s12 = 0.50 * ( VelgUy + VelgVx );
    const double s13 = 0.50 * ( VelgUz + VelgWx );
    const double s23 = 0.50 * ( VelgVz + VelgWy );
    const double SijSij =( s11*s11 + s22*s22 + s33*s33
                 + 2.0*s12*s12 + 2.0*s13*s13 + 2.0*s23*s23);
    IsI(i,j,k) = std::sqrt(2.0*SijSij);

 // viscosity_new(i,j,k) =  ( Cs_dynamic(i,j,k) * filter2 * sigOper(i,j,k)* Density_sigma(i,j,k)+mu ) * vol_fraction(i,j,k);

    mu_sgc(i,j,k) =  ( m_Cs * m_Cs * filter2 * sigOper  * Density_sigma(i,j,k)+m_molecular_visc )
                              * vol_fraction(i,j,k) ;
    mu_turb(i,j,k) = (mu_sgc(i,j,k) - m_molecular_visc)*vol_fraction(i,j,k); //

    //TurbViscosity_new(i,j,k) = viscosity_new(i,j,k) - mu;

  /* if(i==2&&j==20&&k==20)
   { std::cout<<"scientific:\n"<<std::scientific;
    std::cout<<"Uvel gradient"<< VelgUx<<" "<< VelgUy<<" "<<VelgUz<<"\n"<<"Vvel gradient"<< VelgVx<<" "<< VelgVy<<" "<<VelgVz<<"\n"<<"Vvel gradient"<< VelgWx<<" "<< VelgWy<<" "<<VelgWz<<"\n"<<std::endl;
     std::cout<<"G11="<< G11<<" G12="<< G12<<" G13="<<G13<<"\n"<<" G21="<< G21<<" G22="<< G22<<" G23="<<G23<<"\n"<<" G31="<< G31<<" G32="<<G32 <<" G33="<<G33<<"\n"<<std::endl;
     std::cout<<" I1="<< I1  <<" I2= "<< I2 <<" I3="<< I3<<"\n" <<" alpha1="<< alpha1  <<" alpha2="<< alpha2<<" alpha3="<<alpha3<<"\n"<<" sigma1="<< sigma1 <<" sigma2= "<<sigma2<<" sigma3= "<<sigma3<<"\n"<<std::endl;
     std::cout<<" sigmaOperator="<<sigOper(i,j,k)<<" Cs_dynamic="<<Cs_dynamic(i,j,k)<<" filter="<<filter2<<" Density_sigma="<<Density_sigma(i,j,k)<<" vol_fraction="<<vol_fraction(i,j,k)<<" Viscosity_new="<<viscosity_new(i,j,k)<<std::endl;
   }
   */

  });

  ArchesCore::BCFilter bcfilter;
  bcfilter.apply_zero_neumann(patch,mu_sgc,vol_fraction);
  bcfilter.apply_zero_neumann(patch,mu_turb,vol_fraction);
}
