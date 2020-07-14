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
TaskAssignedExecutionSpace DSFT::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DSFT::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &DSFT::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DSFT::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DSFT::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DSFT::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DSFT::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &DSFT::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &DSFT::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DSFT::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DSFT::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
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
template <typename ExecSpace, typename MemSpace>
void DSFT::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto filterRho = tsk_info->get_field< CCVariable<double>, double, MemSpace >("Filterrho");
  auto filterRhoU = tsk_info->get_field< SFCXVariable<double>, double, MemSpace >("Filterrhou");
  auto filterRhoV = tsk_info->get_field< SFCYVariable<double>, double, MemSpace >("Filterrhov");
  auto filterRhoW = tsk_info->get_field< SFCZVariable<double>, double, MemSpace >("Filterrhow");

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
    filterRho(i,j,k) = 0.0;
    filterRhoU(i,j,k) = 0.0;
    filterRhoV(i,j,k) = 0.0;
    filterRhoW(i,j,k) = 0.0;
  });

  computeModel(patch, tsk_info, execObj);

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
template <typename ExecSpace, typename MemSpace>
void DSFT::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  this->computeModel(patch, tsk_info, execObj);
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DSFT::computeModel( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto uVel = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace >(m_u_vel_name);
  auto vVel = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace >(m_v_vel_name);
  auto wVel = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace >(m_w_vel_name);
  auto rho = tsk_info->get_field<constCCVariable<double>, const double, MemSpace >(m_density_name);
  auto vol_fraction = tsk_info->get_field<constCCVariable<double>, const double, MemSpace >(m_volFraction_name);
  auto CCuVel = tsk_info->get_field<constCCVariable<double>, const double, MemSpace >(m_cc_u_vel_name);
  auto CCvVel = tsk_info->get_field<constCCVariable<double>, const double, MemSpace >(m_cc_v_vel_name);
  auto CCwVel = tsk_info->get_field<constCCVariable<double>, const double, MemSpace >(m_cc_w_vel_name);

  const Vector Dx = patch->dCell();
  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::BlockRange initrange( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  auto IsI = tsk_info->get_field< CCVariable<double>, double, MemSpace >( m_IsI_name );
  auto s11 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "s11" );
  auto s12 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "s12" );
  auto s13 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "s13" );
  auto s22 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "s22" );
  auto s23 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "s23" );
  auto s33 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "s33" );

  auto Beta11 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "Beta11" );
  auto Beta12 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "Beta12" );
  auto Beta13 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "Beta13" );
  auto Beta22 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "Beta22" );
  auto Beta23 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "Beta23" );
  auto Beta33 = tsk_info->get_field< CCVariable<double>, double, MemSpace >( "Beta33" );

  auto filterRho = tsk_info->get_field< CCVariable<double>, double, MemSpace >("Filterrho");

  auto rhoBC = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoBC");

  auto filterRhoU = tsk_info->get_field< SFCXVariable<double>, double, MemSpace >("Filterrhou");
  auto filterRhoV = tsk_info->get_field< SFCYVariable<double>, double, MemSpace >("Filterrhov");
  auto filterRhoW = tsk_info->get_field< SFCZVariable<double>, double, MemSpace >("Filterrhow");

  auto rhoUU = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoUU");
  auto rhoVV = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoVV");
  auto rhoWW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoWW");
  auto rhoUV = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoUV");
  auto rhoUW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoUW");
  auto rhoVW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoVW");
  auto rhoU = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoU");
  auto rhoV = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoV");
  auto rhoW = tsk_info->get_field< CCVariable<double>, double, MemSpace >("rhoW");

  //init all at once
  Uintah::parallel_for(execObj, initrange, KOKKOS_LAMBDA(int i, int j, int k){
    IsI(i, j, k) = 0.0;
	s11(i, j, k) = 0.0;
	s12(i, j, k) = 0.0;
	s13(i, j, k) = 0.0;
	s22(i, j, k) = 0.0;
	s23(i, j, k) = 0.0;
	s33(i, j, k) = 0.0;

	Beta11(i, j, k) = 0.0;
	Beta12(i, j, k) = 0.0;
	Beta13(i, j, k) = 0.0;
	Beta22(i, j, k) = 0.0;
	Beta23(i, j, k) = 0.0;
	Beta33(i, j, k) = 0.0;

	filterRho(i, j, k) = 0.0;

	rhoBC(i, j, k) = rho(i,j,k);

	filterRhoU(i, j, k) = 0.0;
	filterRhoV(i, j, k) = 0.0;
	filterRhoW(i, j, k) = 0.0;

    rhoUU(i, j, k) = 0.0;
    rhoVV(i, j, k) = 0.0;
    rhoWW(i, j, k) = 0.0;
    rhoUV(i, j, k) = 0.0;
    rhoUW(i, j, k) = 0.0;
    rhoVW(i, j, k) = 0.0;
    rhoU(i, j, k) = 0.0;
    rhoV(i, j, k) = 0.0;
    rhoW(i, j, k) = 0.0;

  });

  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){

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

  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
    Beta11(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s11(i,j,k);
    Beta22(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s22(i,j,k);
    Beta33(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s33(i,j,k);
    Beta12(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s12(i,j,k);
    Beta13(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s13(i,j,k);
    Beta23(i,j,k) = rho(i,j,k)*IsI(i,j,k)*s23(i,j,k);
  });


  ArchesCore::BCFilter bcfilter;
  bcfilter.apply_zero_neumann(execObj,patch,Beta11,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,Beta22,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,Beta33,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,Beta12,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,Beta13,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,Beta23,vol_fraction);

  // Filter rho

  // this need to be fixed (??)
  //filterRho.copy(ref_rho);

  bcfilter.apply_BC_rho( patch, rhoBC, rho, vol_fraction, execObj);
  m_Filter.applyFilter( rho, filterRho, range, vol_fraction, execObj);

  //m_Filter.applyFilter(rho,filterRho,range,vol_fraction);

  // filter rho*ux...

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
  m_Filter.applyFilter<constSFCXVariable<double>>(uVel,filterRhoU,rho,vol_fraction,range_u, execObj);
  if ( yminus ){
    low = patch->getCellLowIndex()+IntVector(0,1,0);
  } else {
    low = patch->getCellLowIndex();
  }

  Uintah::BlockRange range_v(low, patch->getCellHighIndex() );
  m_Filter.applyFilter<constSFCYVariable<double>>(vVel,filterRhoV,rho,vol_fraction,range_v, execObj);

  if ( zminus ){
    low = patch->getCellLowIndex()+IntVector(0,0,1);
  } else {
    low = patch->getCellLowIndex();
  }
  Uintah::BlockRange range_w(low, patch->getCellHighIndex() );
  m_Filter.applyFilter<constSFCZVariable<double>>(wVel,filterRhoW,rho,vol_fraction,range_w, execObj);

  bcfilter.apply_BC_rhou<SFCXVariable<double>>(patch,filterRhoU,uVel,rho,vol_fraction, execObj);
  bcfilter.apply_BC_rhou<SFCYVariable<double>>(patch,filterRhoV,vVel,rho,vol_fraction, execObj);
  bcfilter.apply_BC_rhou<SFCZVariable<double>>(patch,filterRhoW,wVel,rho,vol_fraction, execObj);
  bcfilter.apply_BC_filter_rho(patch,filterRho,rhoBC,vol_fraction, execObj);

  // Compute rhouiuj at cc

  Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
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

  bcfilter.apply_zero_neumann(execObj,patch,rhoUU,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoVV,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoWW,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoUV,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoUW,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoVW,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoV,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoU,vol_fraction);
  bcfilter.apply_zero_neumann(execObj,patch,rhoW,vol_fraction);
}

} //namespace Uintah
