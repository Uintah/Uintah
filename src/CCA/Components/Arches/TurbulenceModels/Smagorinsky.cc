#include <CCA/Components/Arches/TurbulenceModels/Smagorinsky.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
Smagorinsky::Smagorinsky( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
Smagorinsky::~Smagorinsky()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Smagorinsky::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Smagorinsky::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &Smagorinsky::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Smagorinsky::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Smagorinsky::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Smagorinsky::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &Smagorinsky::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Smagorinsky::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Smagorinsky::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace Smagorinsky::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &Smagorinsky::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Smagorinsky::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Smagorinsky::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace Smagorinsky::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
Smagorinsky::problemSetup( ProblemSpecP& db ){

  using namespace Uintah::ArchesCore;

  m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, "uVelocity" );
  m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, "vVelocity" );
  m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, "wVelocity" );

  m_cc_u_vel_name = parse_ups_for_role( CCUVELOCITY_ROLE, db, m_u_vel_name + "_cc" );
  m_cc_v_vel_name = parse_ups_for_role( CCVVELOCITY_ROLE, db, m_v_vel_name + "_cc" );
  m_cc_w_vel_name = parse_ups_for_role( CCWVELOCITY_ROLE, db, m_w_vel_name + "_cc" );

  m_density_name = parse_ups_for_role( DENSITY_ROLE, db, "density" );

  Nghost_cells = 1;

  db->getWithDefault("Cs", m_Cs, 1.3);

  m_total_vis_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db, default_viscosity_name );

  // ** HACK **
  if ( m_total_vis_name == "viscosityCTS" ){ m_using_production = true; }

  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
    params_root->findBlock("PhysicalConstants")->require("viscosity", m_molecular_visc);
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

//--------------------------------------------------------------------------------------------------
void
Smagorinsky::create_local_labels(){

  if ( !m_using_production ){
    register_new_variable<CCVariable<double> >( m_total_vis_name);
  }

}

//--------------------------------------------------------------------------------------------------
void
Smagorinsky::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks )
{

  register_variable( m_total_vis_name, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void Smagorinsky::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& mu_sgc = tsk_info->get_field<CCVariable<double> >(m_total_vis_name);
  mu_sgc.initialize(0.0);

}

//--------------------------------------------------------------------------------------------------
void
Smagorinsky::register_timestep_init(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks )
{

  if ( !m_using_production ){
    register_variable( m_total_vis_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
Smagorinsky::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  if ( !m_using_production ){
    CCVariable<double>& mu_sgc = tsk_info->get_field<CCVariable<double> >(m_total_vis_name);
    mu_sgc.initialize(0.0);
  }

}

//--------------------------------------------------------------------------------------------------
void
Smagorinsky::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                     variable_registry, const int time_substep  , const bool packed_tasks ){

  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  register_variable( m_cc_u_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_v_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);
  register_variable( m_cc_w_vel_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  register_variable( m_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep);

  register_variable( m_total_vis_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void Smagorinsky::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  constSFCXVariable<double>& uVel = tsk_info->get_field<constSFCXVariable<double> >(m_u_vel_name);
  constSFCYVariable<double>& vVel = tsk_info->get_field<constSFCYVariable<double> >(m_v_vel_name);
  constSFCZVariable<double>& wVel = tsk_info->get_field<constSFCZVariable<double> >(m_w_vel_name);

  constCCVariable<double>& CCuVel = tsk_info->get_field<constCCVariable<double> >(m_cc_u_vel_name);
  constCCVariable<double>& CCvVel = tsk_info->get_field<constCCVariable<double> >(m_cc_v_vel_name);
  constCCVariable<double>& CCwVel = tsk_info->get_field<constCCVariable<double> >(m_cc_w_vel_name);

  constCCVariable<double>& density = tsk_info->get_field<constCCVariable<double> >(m_density_name);

  CCVariable<double>& mu_sgc = tsk_info->get_field<CCVariable<double> >(m_total_vis_name);

  const Vector Dx = patch->dCell();
  const double delta = pow(Dx.x()*Dx.y()*Dx.z(),1./3.);
  double IsI = 0.0;

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
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

    IsI = 2.0 * ( pow(s11,2.0) + pow(s22,2.0) + pow(s33,2.0)
              + 2.0 * ( pow(s12,2) + pow(s13,2) + pow(s23,2) ) );

    IsI = std::sqrt( IsI );
    mu_sgc(i,j,k) = pow(m_Cs*delta,2.0)*IsI*density(i,j,k) + m_molecular_visc; // I need to times density

  });

}
} //namespace Uintah
