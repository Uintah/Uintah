#include <CCA/Components/Arches/TurbulenceModels/WALE.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

namespace Uintah{

typedef ArchesFieldContainer AFC;

//---------------------------------------------------------------------------------
WALE::WALE( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//---------------------------------------------------------------------------------
WALE::~WALE()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WALE::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WALE::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &WALE::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &WALE::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &WALE::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WALE::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &WALE::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &WALE::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &WALE::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WALE::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace WALE::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//---------------------------------------------------------------------------------
void
WALE::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  Nghost_cells = 1;

  m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
  m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
  m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );
  m_density_name = parse_ups_for_role( DENSITY_ROLE, db, "density" );

  m_cc_u_vel_name = parse_ups_for_role( CCUVELOCITY_ROLE, db, m_u_vel_name + "_cc"  );
  m_cc_v_vel_name = parse_ups_for_role( CCVVELOCITY_ROLE, db, m_v_vel_name + "_cc"  );
  m_cc_w_vel_name = parse_ups_for_role( CCWVELOCITY_ROLE, db, m_w_vel_name + "_cc"  );

  m_total_vis_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db, ArchesCore::default_viscosity_name );

  std::stringstream composite_name;
  composite_name << "strainMagnitudeLabel_" << m_task_name;
  m_IsI_name = composite_name.str();

  m_turb_viscosity_name = "turb_viscosity";
  m_volFraction_name = "volFraction";

  db->getWithDefault("Cs", m_Cs, .5);

  //m_total_vis_name = m_task_name;

  // Use the production velocity name to signal if this is being used in the ExplicitSolver.
  if (m_u_vel_name == "uVelocitySPBC") {
    m_create_labels_IsI_t_viscosity = false;
    m_total_vis_name = "viscosityCTS";
    m_IsI_name = "strainMagnitudeLabel";
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

  if (m_create_labels_IsI_t_viscosity) {
    register_new_variable<CCVariable<double> >(m_IsI_name);
    register_new_variable<CCVariable<double> >( m_total_vis_name);
    register_new_variable<CCVariable<double> >( m_turb_viscosity_name);
  }

}

//---------------------------------------------------------------------------------
void
WALE::register_initialize( std::vector<AFC::VariableInformation>&
                           variable_registry , const bool packed_tasks ){

  register_variable( m_total_vis_name, AFC::COMPUTES, variable_registry );
  register_variable( m_turb_viscosity_name, AFC::COMPUTES, variable_registry );
  register_variable( m_IsI_name, AFC::COMPUTES ,  variable_registry );

}

//---------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void WALE::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto mu_sgc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_total_vis_name);
  auto mu_turb = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_turb_viscosity_name);
  auto IsI = tsk_info->get_field< CCVariable<double>, double, MemSpace>(m_IsI_name);

  parallel_initialize(execObj,0.0, mu_sgc, mu_turb, IsI);

}

//---------------------------------------------------------------------------------
void
WALE::register_timestep_eval( std::vector<AFC::VariableInformation>&
                              variable_registry, const int time_substep, const bool packed_tasks){

  register_variable( m_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);

  register_variable( m_cc_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_density_name, AFC::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_volFraction_name, AFC::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_IsI_name, AFC::COMPUTES ,  variable_registry, time_substep );
  if (m_create_labels_IsI_t_viscosity) {
    register_variable( m_total_vis_name, AFC::COMPUTES ,  variable_registry, time_substep );
    register_variable( m_turb_viscosity_name, AFC::COMPUTES ,  variable_registry, time_substep );
  } else {
    register_variable( m_total_vis_name, AFC::MODIFIES ,  variable_registry, time_substep );
    register_variable( m_turb_viscosity_name, AFC::MODIFIES ,  variable_registry, time_substep );
  }
}

//---------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void WALE::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto uVel = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(m_u_vel_name);
  auto vVel = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(m_v_vel_name);
  auto wVel = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(m_w_vel_name);

  auto CCuVel = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_cc_u_vel_name);
  auto CCvVel = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_cc_v_vel_name);
  auto CCwVel = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_cc_w_vel_name);

  auto mu_sgc = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_total_vis_name);
  auto mu_turb = tsk_info->get_field<CCVariable<double> ,double, MemSpace>(m_turb_viscosity_name);
  auto IsI = tsk_info->get_field< CCVariable<double>, double, MemSpace>(m_IsI_name);
  auto rho = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_density_name);
  auto vol_fraction = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_volFraction_name);

  parallel_initialize(execObj, 0.0, IsI, mu_sgc);
  const Vector Dx = patch->dCell();
  const double delta = pow(Dx.x()*Dx.y()*Dx.z(),1./3.);

//  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  const double SMALL = 1e-16;
  const double local_Cs = m_Cs;
  const double local_molecular_visc = m_molecular_visc;
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

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


    const double du11 = (uep-uwp)/Dx.x(); // dudx
    const double du12 = (unp-usp)/Dx.y(); //dudy
    const double du21 = (vep-vwp)/Dx.x(); //dvdx
    const double du13 = (utp-ubp)/Dx.z(); //dudz
    const double du31 = (wep-wwp)/Dx.x(); //dwdx
    const double du22 = (vnp-vsp)/Dx.y(); //dvdy
    const double du23 = (vtp-vbp)/Dx.z(); //dvdz
    const double du32 = (wnp-wsp)/Dx.y(); //dwdy
    const double du33 = (wtp-wbp)/Dx.z(); //dwdz

    const double s11 = du11;// (uep-uwp)/Dx.x();
    const double s22 = du22;// (vnp-vsp)/Dx.y();
    const double s33 = du33;// (wtp-wbp)/Dx.z();
    //const double s12 = 0.50 * ((unp-usp)/Dx.y() + (vep-vwp)/Dx.x());
    const double s12 = 0.50 * ( du12 + du21 );
    //const double s13 = 0.50 * ((utp-ubp)/Dx.z() + (wep-wwp)/Dx.x());
    const double s13 = 0.50 * ( du13 + du31 );
    //const double s23 = 0.50 * ((vtp-vbp)/Dx.z() + (wnp-wsp)/Dx.y());
    const double s23 = 0.50 * ( du23 + du32 );

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

    const double fvis = pow(SijdSijd,1.5)/(pow(SijSij,2.5) + pow(SijdSijd,5./4.)+SMALL);

    mu_sgc(i,j,k) = pow(local_Cs*delta,2.0)*fvis*rho(i,j,k)*vol_fraction(i,j,k) + local_molecular_visc;
    IsI(i,j,k) = sqrt(2.0*SijSij)*vol_fraction(i,j,k) ;
    mu_turb(i,j,k) = mu_sgc(i,j,k) - local_molecular_visc; //
  });
  Uintah::ArchesCore::BCFilter bcfilter;
  bcfilter.apply_zero_neumann(execObj,patch,mu_sgc,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,mu_turb,vol_fraction);

}
} //namespace Uintah
