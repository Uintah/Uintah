#include <CCA/Components/Arches/ParticleModels/RateDeposition.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
Uintah::RateDeposition::RateDeposition( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {}

//--------------------------------------------------------------------------------------------------
RateDeposition::~RateDeposition(){}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RateDeposition::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RateDeposition::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &RateDeposition::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &RateDeposition::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &RateDeposition::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RateDeposition::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &RateDeposition::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &RateDeposition::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &RateDeposition::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RateDeposition::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &RateDeposition::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &RateDeposition::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &RateDeposition::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RateDeposition::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
RateDeposition::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();
  CoalHelper& coal_helper = CoalHelper::self();

   T_fluid = coal_helper.get_coal_db().T_fluid;
   FactA = coal_helper.get_coal_db().visc_pre_exponential_factor;
   lnFactA = std::log(FactA);
   FactB = coal_helper.get_coal_db().visc_activation_energy;
  if (FactA==-999 || FactB==-999){
    throw ProblemSetupException("Error: RateDeposition requires specification of ash viscosity parameters.", __FILE__, __LINE__);
  }

  _Tmelt = coal_helper.get_coal_db().T_hemisphere;

  _ParticleTemperature_base_name  = ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_TEMPERATURE);
  _MaxParticleTemperature_base_name= ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_MAXTEMPERATURE);

  _ProbParticleX_base_name = "ProbParticleX";
  _ProbParticleY_base_name = "ProbParticleY";
  _ProbParticleZ_base_name = "ProbParticleZ";

  _ProbDepositionX_base_name = "ProbDepositionX";
  _ProbDepositionY_base_name = "ProbDepositionY";
  _ProbDepositionZ_base_name = "ProbDepositionZ";

  _RateDepositionX_base_name= "RateDepositionX";
  _RateDepositionY_base_name= "RateDepositionY";
  _RateDepositionZ_base_name= "RateDepositionZ";

  _RateImpactX_base_name= "RateImpactLossX";
  _RateImpactY_base_name= "RateImpactLossY";
  _RateImpactZ_base_name= "RateImpactLossZ";

  _ProbSurfaceX_name = "ProbSurfaceX";
  _ProbSurfaceY_name = "ProbSurfaceY";
  _ProbSurfaceZ_name = "ProbSurfaceZ";

  _WallTemperature_name = "Temperature";

  _xvel_base_name  = ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_XVEL);
  _yvel_base_name  = ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_YVEL);
  _zvel_base_name  = ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_ZVEL);

  _weight_base_name  = "w";
  _rho_base_name  = ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_DENSITY);
  _diameter_base_name  = ArchesCore::parse_for_particle_role_to_label(db,ArchesCore::P_SIZE);

  _FluxPx_base_name  = "FluxPx";
  _FluxPy_base_name  = "FluxPy";
  _FluxPz_base_name  = "FluxPz";
  _pi_div_six = acos(-1.0)/6.0;

}

//--------------------------------------------------------------------------------------------------
void
RateDeposition::create_local_labels(){
  for (int i =0; i< _Nenv ; i++){
    const std::string ProbParticleX_name = get_env_name(i, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(i, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(i, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(i, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(i, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(i, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(i, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(i, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(i, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(i, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(i, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(i, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(i, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(i, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(i, _RateImpactZ_base_name);

    register_new_variable< SFCXVariable<double> >(RateDepositionX_name );
    register_new_variable< SFCYVariable<double> >(RateDepositionY_name );
    register_new_variable< SFCZVariable<double> >(RateDepositionZ_name );

    register_new_variable< SFCXVariable<double> >(RateImpactX_name );
    register_new_variable< SFCYVariable<double> >(RateImpactY_name );
    register_new_variable< SFCZVariable<double> >(RateImpactZ_name );

    register_new_variable< SFCXVariable<double> >(ProbParticleX_name );
    register_new_variable< SFCYVariable<double> >(ProbParticleY_name );
    register_new_variable< SFCZVariable<double> >(ProbParticleZ_name );

    register_new_variable< SFCXVariable<double> >(FluxPx_name );
    register_new_variable< SFCYVariable<double> >(FluxPy_name );
    register_new_variable< SFCZVariable<double> >(FluxPz_name );

    register_new_variable< SFCXVariable<double> >(ProbDepositionX_name );
    register_new_variable< SFCYVariable<double> >(ProbDepositionY_name );
    register_new_variable< SFCZVariable<double> >(ProbDepositionZ_name );
  }
  register_new_variable< SFCXVariable<double> >(_ProbSurfaceX_name );
  register_new_variable< SFCYVariable<double> >(_ProbSurfaceY_name );
  register_new_variable< SFCZVariable<double> >(_ProbSurfaceZ_name );

}

//--------------------------------------------------------------------------------------------------
void
RateDeposition::register_initialize( std::vector<AFC_VI>& variable_registry , const bool packed_tasks){

  for ( int i=0; i< _Nenv;i++){
    const std::string ProbParticleX_name = get_env_name(i, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(i, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(i, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(i, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(i, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(i, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(i, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(i, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(i, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(i, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(i, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(i, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(i, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(i, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(i, _RateImpactZ_base_name);


    register_variable(  RateDepositionX_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateDepositionY_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateDepositionZ_name   , AFC::COMPUTES , variable_registry );

    register_variable(  RateImpactX_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateImpactY_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateImpactZ_name   , AFC::COMPUTES , variable_registry );


    register_variable(  FluxPx_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  FluxPy_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  FluxPz_name    ,  AFC::COMPUTES , variable_registry );

    register_variable(  ProbParticleX_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbParticleY_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbParticleZ_name    ,  AFC::COMPUTES , variable_registry );

    register_variable(  ProbDepositionX_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbDepositionY_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbDepositionZ_name    ,  AFC::COMPUTES , variable_registry );
  }

  register_variable(  _ProbSurfaceX_name     , AFC::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceY_name     , AFC::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceZ_name     , AFC::COMPUTES ,  variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void RateDeposition::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( int e=0; e< _Nenv;e++){
    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(e, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(e, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(e, _RateImpactZ_base_name);

    SFCXVariable<double>& FluxPx   =         tsk_info->get_field<SFCXVariable<double> >(FluxPx_name);
    SFCYVariable<double>& FluxPy   =         tsk_info->get_field<SFCYVariable<double> >(FluxPy_name);
    SFCZVariable<double>& FluxPz   =         tsk_info->get_field<SFCZVariable<double> >(FluxPz_name);

    SFCXVariable<double>& ProbParticleX  =   tsk_info->get_field<SFCXVariable<double> >(ProbParticleX_name);
    SFCYVariable<double>& ProbParticleY  =   tsk_info->get_field<SFCYVariable<double> >(ProbParticleY_name);
    SFCZVariable<double>& ProbParticleZ  =   tsk_info->get_field<SFCZVariable<double> >(ProbParticleZ_name);

    SFCXVariable<double>& ProbDepositionX   =  tsk_info->get_field<SFCXVariable<double> >(ProbDepositionX_name);
    SFCYVariable<double>& ProbDepositionY   =  tsk_info->get_field<SFCYVariable<double> >(ProbDepositionY_name);
    SFCZVariable<double>& ProbDepositionZ   =  tsk_info->get_field<SFCZVariable<double> >(ProbDepositionZ_name);

    SFCXVariable<double>& RateDepositionX   =  tsk_info->get_field<SFCXVariable<double> >( RateDepositionX_name);
    SFCYVariable<double>& RateDepositionY   =  tsk_info->get_field<SFCYVariable<double> >( RateDepositionY_name);
    SFCZVariable<double>& RateDepositionZ   =  tsk_info->get_field<SFCZVariable<double> >( RateDepositionZ_name);

    SFCXVariable<double>& RateImpactX = tsk_info->get_field<SFCXVariable<double> >( RateImpactX_name);
    SFCYVariable<double>& RateImpactY = tsk_info->get_field<SFCYVariable<double> >( RateImpactY_name);
    SFCZVariable<double>& RateImpactZ = tsk_info->get_field<SFCZVariable<double> >( RateImpactZ_name);

    RateDepositionX.initialize(0.0);
    RateDepositionY.initialize(0.0);
    RateDepositionZ.initialize(0.0);

    RateImpactX.initialize(0.0);
    RateImpactY.initialize(0.0);
    RateImpactZ.initialize(0.0);

    ProbParticleX.initialize(0.0);
    ProbParticleY.initialize(0.0);
    ProbParticleZ.initialize(0.0);

    ProbDepositionX.initialize( 0.0);
    ProbDepositionY.initialize( 0.0);
    ProbDepositionZ.initialize( 0.0);

    FluxPx.initialize(0.0) ;
    FluxPy.initialize(0.0) ;
    FluxPz.initialize(0.0) ;

  }
  SFCXVariable<double>& ProbSurfaceX =  tsk_info->get_field<SFCXVariable<double> >(_ProbSurfaceX_name);
  SFCYVariable<double>& ProbSurfaceY =  tsk_info->get_field<SFCYVariable<double> >(_ProbSurfaceY_name);
  SFCZVariable<double>& ProbSurfaceZ =  tsk_info->get_field<SFCZVariable<double> >(_ProbSurfaceZ_name);
  ProbSurfaceX.initialize(0.0) ;
  ProbSurfaceY.initialize(0.0) ;
  ProbSurfaceZ.initialize(0.0) ;
}

//--------------------------------------------------------------------------------------------------
void
RateDeposition::register_timestep_init( std::vector<AFC_VI>& variable_registry , const bool packed_tasks){
  for ( int i=0; i< _Nenv;i++){
    const std::string ProbParticleX_name = get_env_name(i, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(i, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(i, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(i, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(i, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(i, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(i, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(i, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(i, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(i, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(i, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(i, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(i, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(i, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(i, _RateImpactZ_base_name);

    register_variable(  RateDepositionX_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateDepositionY_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateDepositionZ_name   , AFC::COMPUTES , variable_registry );

    register_variable(  RateImpactX_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateImpactY_name   , AFC::COMPUTES , variable_registry );
    register_variable(  RateImpactZ_name   , AFC::COMPUTES , variable_registry );

    register_variable(  FluxPx_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  FluxPy_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  FluxPz_name    ,  AFC::COMPUTES , variable_registry );

    register_variable(  ProbParticleX_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbParticleY_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbParticleZ_name    ,  AFC::COMPUTES , variable_registry );

    register_variable(  ProbDepositionX_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbDepositionY_name    ,  AFC::COMPUTES , variable_registry );
    register_variable(  ProbDepositionZ_name    ,  AFC::COMPUTES , variable_registry );
  }

  register_variable(  _ProbSurfaceX_name     , AFC::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceY_name     , AFC::COMPUTES ,  variable_registry );
  register_variable(  _ProbSurfaceZ_name     , AFC::COMPUTES ,  variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
RateDeposition::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( int e=0; e< _Nenv;e++){
    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(e, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(e, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(e, _RateImpactZ_base_name);

    SFCXVariable<double>& FluxPx = tsk_info->get_field<SFCXVariable<double> >(FluxPx_name);
    SFCYVariable<double>& FluxPy = tsk_info->get_field<SFCYVariable<double> >(FluxPy_name);
    SFCZVariable<double>& FluxPz = tsk_info->get_field<SFCZVariable<double> >(FluxPz_name);

    SFCXVariable<double>& ProbParticleX = tsk_info->get_field<SFCXVariable<double> >(ProbParticleX_name);
    SFCYVariable<double>& ProbParticleY = tsk_info->get_field<SFCYVariable<double> >(ProbParticleY_name);
    SFCZVariable<double>& ProbParticleZ = tsk_info->get_field<SFCZVariable<double> >(ProbParticleZ_name);

    SFCXVariable<double>& ProbDepositionX = tsk_info->get_field<SFCXVariable<double> >(ProbDepositionX_name);
    SFCYVariable<double>& ProbDepositionY = tsk_info->get_field<SFCYVariable<double> >(ProbDepositionY_name);
    SFCZVariable<double>& ProbDepositionZ = tsk_info->get_field<SFCZVariable<double> >(ProbDepositionZ_name);

    SFCXVariable<double>& RateDepositionX = tsk_info->get_field<SFCXVariable<double> >( RateDepositionX_name);
    SFCYVariable<double>& RateDepositionY = tsk_info->get_field<SFCYVariable<double> >( RateDepositionY_name);
    SFCZVariable<double>& RateDepositionZ = tsk_info->get_field<SFCZVariable<double> >( RateDepositionZ_name);
    RateDepositionX.initialize(0.0);
    RateDepositionY.initialize(0.0);
    RateDepositionZ.initialize(0.0);

    SFCXVariable<double>& RateImpactX = tsk_info->get_field<SFCXVariable<double> >( RateImpactX_name);
    SFCYVariable<double>& RateImpactY = tsk_info->get_field<SFCYVariable<double> >( RateImpactY_name);
    SFCZVariable<double>& RateImpactZ = tsk_info->get_field<SFCZVariable<double> >( RateImpactZ_name);
    RateImpactX.initialize(0.0);
    RateImpactY.initialize(0.0);
    RateImpactZ.initialize(0.0);

    ProbParticleX.initialize(0.0);
    ProbParticleY.initialize(0.0);
    ProbParticleZ.initialize(0.0);

    ProbDepositionX.initialize( 0.0);
    ProbDepositionY.initialize( 0.0);
    ProbDepositionZ.initialize( 0.0);

    FluxPx.initialize(0.0) ;
    FluxPy.initialize(0.0) ;
    FluxPz.initialize(0.0) ;

  }

  SFCXVariable<double>& ProbSurfaceX = tsk_info->get_field<SFCXVariable<double> >(_ProbSurfaceX_name);
  SFCYVariable<double>& ProbSurfaceY = tsk_info->get_field<SFCYVariable<double> >(_ProbSurfaceY_name);
  SFCZVariable<double>& ProbSurfaceZ = tsk_info->get_field<SFCZVariable<double> >(_ProbSurfaceZ_name);
  ProbSurfaceX.initialize(0.0) ;
  ProbSurfaceY.initialize(0.0) ;
  ProbSurfaceZ.initialize(0.0) ;

}

//--------------------------------------------------------------------------------------------------
void
RateDeposition::register_timestep_eval( std::vector<AFC_VI>& variable_registry, const int time_substep , const bool packed_tasks){

  for(int e= 0; e< _Nenv; e++){

    const std::string MaxParticleTemperature_name = ArchesCore::append_env(_MaxParticleTemperature_base_name ,e);
    const std::string ParticleTemperature_name = ArchesCore::append_env(_ParticleTemperature_base_name ,e);
    const std::string weight_name = ArchesCore::append_env(_weight_base_name ,e);
    const std::string rho_name = ArchesCore::append_env(_rho_base_name ,e);
    const std::string diameter_name = ArchesCore::append_env(_diameter_base_name ,e);

    const std::string  xvel_name = ArchesCore::append_env(_xvel_base_name ,e);
    const std::string  yvel_name = ArchesCore::append_env(_yvel_base_name ,e);
    const std::string  zvel_name = ArchesCore::append_env(_zvel_base_name ,e);

    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(e, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(e, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(e, _RateImpactZ_base_name);


    register_variable( RateDepositionX_name, AFC::MODIFIES, variable_registry );
    register_variable( RateDepositionY_name, AFC::MODIFIES, variable_registry );
    register_variable( RateDepositionZ_name, AFC::MODIFIES, variable_registry );

    register_variable( RateImpactX_name, AFC::MODIFIES, variable_registry );
    register_variable( RateImpactY_name, AFC::MODIFIES, variable_registry );
    register_variable( RateImpactZ_name, AFC::MODIFIES, variable_registry );

    register_variable( MaxParticleTemperature_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
    register_variable( ParticleTemperature_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
    register_variable( weight_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
    register_variable( rho_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
    register_variable( diameter_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );


    register_variable( xvel_name, AFC::REQUIRES, 1 ,AFC::NEWDW, variable_registry );
    register_variable( yvel_name, AFC::REQUIRES, 1 ,AFC::NEWDW, variable_registry );
    register_variable( zvel_name, AFC::REQUIRES, 1 ,AFC::NEWDW, variable_registry );

    register_variable( ProbParticleX_name, AFC::MODIFIES, variable_registry );
    register_variable( ProbParticleY_name, AFC::MODIFIES, variable_registry );
    register_variable( ProbParticleZ_name, AFC::MODIFIES, variable_registry );

    register_variable( FluxPx_name,  AFC::MODIFIES, variable_registry );
    register_variable( FluxPy_name,  AFC::MODIFIES, variable_registry );
    register_variable( FluxPz_name,  AFC::MODIFIES, variable_registry );

    register_variable( ProbDepositionX_name, AFC::MODIFIES, variable_registry );
    register_variable( ProbDepositionY_name, AFC::MODIFIES, variable_registry );
    register_variable( ProbDepositionZ_name, AFC::MODIFIES, variable_registry );
  }

  register_variable( _ProbSurfaceX_name, AFC::MODIFIES, variable_registry );
  register_variable( _ProbSurfaceY_name, AFC::MODIFIES, variable_registry );
  register_variable( _ProbSurfaceZ_name, AFC::MODIFIES, variable_registry );

  register_variable( "surf_out_normX", AFC::REQUIRES, 1, AFC::OLDDW, variable_registry , time_substep );
  register_variable( "surf_out_normY", AFC::REQUIRES, 1, AFC::OLDDW, variable_registry , time_substep );
  register_variable( "surf_out_normZ", AFC::REQUIRES, 1, AFC::OLDDW, variable_registry , time_substep );
  register_variable( "surf_in_normX", AFC::REQUIRES, 1, AFC::OLDDW, variable_registry , time_substep );
  register_variable( "surf_in_normY", AFC::REQUIRES, 1, AFC::OLDDW, variable_registry , time_substep );
  register_variable( "surf_in_normZ", AFC::REQUIRES, 1, AFC::OLDDW, variable_registry , time_substep );
  register_variable( "temperature", AFC::REQUIRES, 1, AFC::LATEST, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void RateDeposition::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  //double CaO=_CaO;double MgO=_MgO; double AlO=_AlO;double SiO=_SiO; //const double alpha=0;
  // const double B0=0; const doulbe B1=0; const double B3=0;
  //double CaOmolar=0.0;               double AlOmolar=0.0;
  //CaOmolar=CaO/(CaO+MgO+AlO+SiO);            AlOmolar=AlO/(CaO+MgO+AlO+SiO);
  //const double alpha=CaOmolar/(AlOmolar+CaOmolar);
  // const double B0=13.8+39.9355*alpha-44.049*alpha*alpha;
  // const double B1=30.481-117.1505*alpha+129.9978*alpha*alpha;
  // const double B2=-40.9429+234.0486*alpha-300.04*alpha*alpha;
  // const double B3= 60.7619-153.9276*alpha+211.1616*alpha*alpha;
  //const double Bactivational=B0+B1*SiO+B2*SiO*SiO+B3*SiO*SiO*SiO;
  //const double Aprepontional=exp(-(0.2693*Bactivational+11.6725));  //const double Bactivational= 47800;

  // computed probability variables:
  SFCXVariable<double>& ProbSurfaceX = tsk_info->get_field<SFCXVariable<double> >(_ProbSurfaceX_name);
  SFCYVariable<double>& ProbSurfaceY = tsk_info->get_field<SFCYVariable<double> >(_ProbSurfaceY_name);
  SFCZVariable<double>& ProbSurfaceZ = tsk_info->get_field<SFCZVariable<double> >(_ProbSurfaceZ_name);

  // constant surface normals
  constSFCXVariable<double>& Norm_in_X = tsk_info->get_field<constSFCXVariable<double> >("surf_in_normX");
  constSFCYVariable<double>& Norm_in_Y = tsk_info->get_field<constSFCYVariable<double> >("surf_in_normY");
  constSFCZVariable<double>& Norm_in_Z = tsk_info->get_field<constSFCZVariable<double> >("surf_in_normZ");
  constSFCXVariable<double>& Norm_out_X = tsk_info->get_field<constSFCXVariable<double> >("surf_out_normX");
  constSFCYVariable<double>& Norm_out_Y = tsk_info->get_field<constSFCYVariable<double> >("surf_out_normY");
  constSFCZVariable<double>& Norm_out_Z = tsk_info->get_field<constSFCZVariable<double> >("surf_out_normZ");

  // constant gas temperature
  constCCVariable<double>& WallTemperature = tsk_info->get_field<constCCVariable<double> >("temperature");

  //Compute the probability of sticking for each face using the wall temperature.
  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

    const double MaxT_temp= 2000.0;

    // Shifts to the WALL cell
    const int shifti = ( Norm_out_X(i,j,k) > 0 ) ? 1 : 0;
    const int shiftj = ( Norm_out_Y(i,j,k) > 0 ) ? 1 : 0;
    const int shiftk = ( Norm_out_Z(i,j,k) > 0 ) ? 1 : 0;

    ProbSurfaceX(i,j,k) = ( Norm_out_X(i,j,k) != 0.0 ) ?
                          compute_prob_stick_fact( lnFactA, FactB, WallTemperature(i-shifti,j,k), MaxT_temp ) :
                          0.0;

    ProbSurfaceY(i,j,k) = ( Norm_out_Y(i,j,k) != 0.0 ) ?
                          compute_prob_stick_fact( lnFactA, FactB, WallTemperature(i,j-shiftj,k), MaxT_temp ) :
                          0.0;

    ProbSurfaceZ(i,j,k) = ( Norm_out_Z(i,j,k) != 0.0 ) ?
                          compute_prob_stick_fact( lnFactA, FactB, WallTemperature(i,j,k-shiftk), MaxT_temp ) :
                          0.0;

  });

  for(int e=0; e<_Nenv; e++){

    const std::string ParticleTemperature_name = ArchesCore::append_env(_ParticleTemperature_base_name ,e);
    const std::string MaxParticleTemperature_name = ArchesCore::append_env(_MaxParticleTemperature_base_name ,e);
    const std::string weight_name = ArchesCore::append_env(_weight_base_name ,e);
    const std::string rho_name = ArchesCore::append_env(_rho_base_name ,e);
    const std::string diameter_name = ArchesCore::append_env(_diameter_base_name ,e);

    const std::string xvel_name = ArchesCore::append_env(_xvel_base_name ,e);
    const std::string yvel_name = ArchesCore::append_env(_yvel_base_name ,e);
    const std::string zvel_name = ArchesCore::append_env(_zvel_base_name ,e);

    const std::string ProbParticleX_name = get_env_name(e, _ProbParticleX_base_name);
    const std::string ProbParticleY_name = get_env_name(e, _ProbParticleY_base_name);
    const std::string ProbParticleZ_name = get_env_name(e, _ProbParticleZ_base_name);

    const std::string ProbDepositionX_name = get_env_name(e, _ProbDepositionX_base_name);
    const std::string ProbDepositionY_name = get_env_name(e, _ProbDepositionY_base_name);
    const std::string ProbDepositionZ_name = get_env_name(e, _ProbDepositionZ_base_name);

    const std::string FluxPx_name = get_env_name(e, _FluxPx_base_name);
    const std::string FluxPy_name = get_env_name(e, _FluxPy_base_name);
    const std::string FluxPz_name = get_env_name(e, _FluxPz_base_name);

    const std::string RateDepositionX_name = get_env_name(e, _RateDepositionX_base_name);
    const std::string RateDepositionY_name = get_env_name(e, _RateDepositionY_base_name);
    const std::string RateDepositionZ_name = get_env_name(e, _RateDepositionZ_base_name);

    const std::string RateImpactX_name = get_env_name(e, _RateImpactX_base_name);
    const std::string RateImpactY_name = get_env_name(e, _RateImpactY_base_name);
    const std::string RateImpactZ_name = get_env_name(e, _RateImpactZ_base_name);


    SFCXVariable<double>& FluxPx = tsk_info->get_field<SFCXVariable<double> >(FluxPx_name);
    SFCYVariable<double>& FluxPy = tsk_info->get_field<SFCYVariable<double> >(FluxPy_name);
    SFCZVariable<double>& FluxPz = tsk_info->get_field<SFCZVariable<double> >(FluxPz_name);

    SFCXVariable<double>& ProbParticleX = tsk_info->get_field<SFCXVariable<double> >(ProbParticleX_name);
    SFCYVariable<double>& ProbParticleY = tsk_info->get_field<SFCYVariable<double> >(ProbParticleY_name);
    SFCZVariable<double>& ProbParticleZ = tsk_info->get_field<SFCZVariable<double> >(ProbParticleZ_name);

    SFCXVariable<double>& ProbDepositionX = tsk_info->get_field<SFCXVariable<double> >(ProbDepositionX_name);
    SFCYVariable<double>& ProbDepositionY = tsk_info->get_field<SFCYVariable<double> >(ProbDepositionY_name);
    SFCZVariable<double>& ProbDepositionZ = tsk_info->get_field<SFCZVariable<double> >(ProbDepositionZ_name);

    SFCXVariable<double>& RateDepositionX = tsk_info->get_field<SFCXVariable<double> >( RateDepositionX_name);
    SFCYVariable<double>& RateDepositionY = tsk_info->get_field<SFCYVariable<double> >( RateDepositionY_name);
    SFCZVariable<double>& RateDepositionZ = tsk_info->get_field<SFCZVariable<double> >( RateDepositionZ_name);

    SFCXVariable<double>& RateImpactX = tsk_info->get_field<SFCXVariable<double> >( RateImpactX_name);
    SFCYVariable<double>& RateImpactY = tsk_info->get_field<SFCYVariable<double> >( RateImpactY_name);
    SFCZVariable<double>& RateImpactZ = tsk_info->get_field<SFCZVariable<double> >( RateImpactZ_name);

    constCCVariable<double>&  MaxParticleTemperature = tsk_info->get_field<constCCVariable<double> >( MaxParticleTemperature_name);
    constCCVariable<double>&  ParticleTemperature = tsk_info->get_field<constCCVariable<double> >( ParticleTemperature_name);
    constCCVariable<double>&  weight = tsk_info->get_field<constCCVariable<double> >( weight_name);
    constCCVariable<double>&  rho = tsk_info->get_field<constCCVariable<double> >( rho_name);
    constCCVariable<double>&  diameter = tsk_info->get_field<constCCVariable<double> >( diameter_name);

    constCCVariable<double>& xvel = tsk_info->get_field<constCCVariable<double> >( xvel_name);
    constCCVariable<double>& yvel = tsk_info->get_field<constCCVariable<double> >( yvel_name);
    constCCVariable<double>& zvel = tsk_info->get_field<constCCVariable<double> >( zvel_name);

    //Compute the probability of sticking for each particle using particle temperature.
    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getExtraCellHighIndex() );

    Uintah::parallel_for( range, [&](int i, int j, int k){

      // Shifts to the FLOW cell
      const int shifti = ( Norm_out_X(i,j,k) > 0.0 ) ? 0 : 1;
      const int shiftj = ( Norm_out_Y(i,j,k) > 0.0 ) ? 0 : 1;
      const int shiftk = ( Norm_out_Z(i,j,k) > 0.0 ) ? 0 : 1;


      //--------------------compute the particle flux --------------------------------------------------------------
      // X direction
      {
        ProbParticleX(i,j,k) = ( Norm_out_X(i,j,k) != 0.0 ) ?
                               compute_prob_stick_fact( lnFactA, FactB, ParticleTemperature(i-shifti,j,k), MaxParticleTemperature(i-shifti,j,k) ) :
                               0.0;
        ProbDepositionX(i,j,k)= std::min(1.0, 0.5*(ProbParticleX(i,j,k)+sqrt(ProbParticleX(i,j,k)*ProbParticleX(i,j,k) +4*(1-ProbParticleX(i,j,k))*ProbSurfaceX(i,j,k))));

        FluxPx(i,j,k) = ( Norm_out_X(i,j,k) != 0.0 ) ?
                        rho(i-shifti,j,k) * xvel(i-shifti,j,k) * weight(i-shifti,j,k) * _pi_div_six * (diameter(i-shifti,j,k) * diameter(i-shifti,j,k) * diameter(i-shifti,j,k)) :
                        0.0;

        RateDepositionX(i,j,k)= FluxPx(i,j,k) * Norm_in_X(i,j,k) > 0.0 ?
                                FluxPx(i,j,k) * ProbDepositionX(i,j,k) :
                                0.0;

        RateImpactX(i,j,k) = FluxPx(i,j,k) * Norm_in_X(i,j,k) > 0.0 ?
                              FluxPx(i,j,k) * (1.-ProbDepositionX(i,j,k)) :
                              0.0;

      }

      // Y direction
      {
        ProbParticleY(i,j,k) = ( Norm_out_Y(i,j,k) != 0.0 ) ?
                               compute_prob_stick_fact( lnFactA, FactB, ParticleTemperature(i,j-shiftj,k), MaxParticleTemperature(i,j-shiftj,k) ) :
                               0.0;
        ProbDepositionY(i,j,k)= std::min(1.0, 0.5*(ProbParticleY(i,j,k)+sqrt(ProbParticleY(i,j,k)*ProbParticleY(i,j,k) +4*(1-ProbParticleY(i,j,k))*ProbSurfaceY(i,j,k))));

        FluxPy(i,j,k) = ( Norm_out_Y(i,j,k) != 0.0 ) ?
                        rho(i,j-shiftj,k)*yvel(i,j-shiftj,k)*weight(i,j-shiftj,k)*_pi_div_six*(diameter(i,j-shiftj,k)*diameter(i,j-shiftj,k)*diameter(i,j-shiftj,k)) :
                        0.0;

        RateDepositionY(i,j,k)= FluxPy(i,j,k) * Norm_in_Y(i,j,k) > 0.0 ?
                                FluxPy(i,j,k) * ProbDepositionY(i,j,k) :
                                0.0;
        RateImpactY(i,j,k) = FluxPy(i,j,k) * Norm_in_Y(i,j,k) > 0.0 ?
                              FluxPy(i,j,k) * (1.-ProbDepositionY(i,j,k)) :
                              0.0;



      }

      // Z direction
      {
        ProbParticleZ(i,j,k) = ( Norm_out_Z(i,j,k) != 0.0 ) ?
                               compute_prob_stick_fact( lnFactA, FactB, ParticleTemperature(i,j,k-shiftk), MaxParticleTemperature(i,j,k-shiftk) ) :
                               0.0;
        ProbDepositionZ(i,j,k)= std::min(1.0, 0.5*(ProbParticleZ(i,j,k)+sqrt(ProbParticleZ(i,j,k)*ProbParticleZ(i,j,k) +4*(1-ProbParticleZ(i,j,k))*ProbSurfaceZ(i,j,k))));

        FluxPz(i,j,k) = ( Norm_out_Z(i,j,k) != 0.0 ) ?
                        rho(i,j,k-shiftk)*zvel(i,j,k-shiftk)*weight(i,j,k-shiftk)*_pi_div_six*(diameter(i,j,k-shiftk)*diameter(i,j,k-shiftk)*diameter(i,j,k-shiftk)) :
                        0.0;

        RateDepositionZ(i,j,k)= FluxPz(i,j,k) * Norm_in_Z(i,j,k) > 0.0 ?
                                FluxPz(i,j,k) * ProbDepositionZ(i,j,k) :
                                0.0;

        RateImpactZ(i,j,k)= FluxPz(i,j,k) * Norm_in_Z(i,j,k) > 0.0 ?
                             FluxPz(i,j,k) * (1.-ProbDepositionZ(i,j,k)) :
                             0.0;

      }

    }); // end cell loop

  } // end for environment
}
