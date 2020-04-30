#include <CCA/Components/Arches/PropertyModelsV2/DensityPredictor.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
DensityPredictor::DensityPredictor( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
  m_use_exact_guess = false;
}

//--------------------------------------------------------------------------------------------------
DensityPredictor::~DensityPredictor(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityPredictor::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityPredictor::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &DensityPredictor::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &DensityPredictor::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &DensityPredictor::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityPredictor::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &DensityPredictor::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &DensityPredictor::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &DensityPredictor::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityPredictor::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &DensityPredictor::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &DensityPredictor::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &DensityPredictor::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace DensityPredictor::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
DensityPredictor::problemSetup( ProblemSpecP& db ){

  if (db->findBlock("use_exact_guess")){
    m_use_exact_guess = true;
    ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ColdFlow");
    if ( db_prop == nullptr ){
      throw InvalidValue("Error: For the density predictor, you must be using cold flow model when computing the exact rho/rhof relationship.", __FILE__, __LINE__);
    }
    db_prop->findBlock("stream_0")->getAttribute("density",m_rho0);
    db_prop->findBlock("stream_1")->getAttribute("density",m_rho1);
    m_f_name = "NA";
    db_prop->findBlock("mixture_fraction")->getAttribute("label",m_f_name);
    if ( m_f_name == "NA" ){
      throw InvalidValue("Error: Mixture fraction name not recognized: "+m_f_name,__FILE__, __LINE__);
    }
  }

  ProblemSpecP press_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("PressureSolver");
  if (press_db->findBlock("src")){
    std::string srcname;
    for (ProblemSpecP src_db = press_db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      m_mass_sources.push_back( srcname );
    }
  }
}

void
DensityPredictor::create_local_labels(){
  register_new_variable<CCVariable<double> >( "new_densityGuess" );
}

//--------------------------------------------------------------------------------------------------
void
DensityPredictor::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( "new_densityGuess", ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DensityPredictor::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& rho = tsk_info->get_field<CCVariable<double> >("new_densityGuess");
  rho.initialize( 0.0 );

}

//--------------------------------------------------------------------------------------------------
void
DensityPredictor::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

  register_variable( "new_densityGuess", ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
DensityPredictor::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& rho = tsk_info->get_field<CCVariable<double> >("new_densityGuess");
  rho.initialize( 0.0 );

}

//--------------------------------------------------------------------------------------------------
void
DensityPredictor::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( "new_densityGuess"  , ArchesFieldContainer::MODIFIES,  variable_registry, time_substep );
  register_variable( "densityGuess"  , ArchesFieldContainer::MODIFIES,  variable_registry, time_substep );
  register_variable( "densityCP"     , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  register_variable( "volFraction"   , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "uVelocitySPBC" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "vVelocitySPBC" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "wVelocitySPBC" , ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::LATEST , variable_registry , time_substep );
  register_variable( "sm_cont" , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
  if ( !m_use_exact_guess ){
    //typedef std::vector<std::string> SVec;
    //for (SVec::iterator i = m_mass_sources.begin(); i != m_mass_sources.end(); i++ ){
      //register_variable( *i , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );
    //}
  }
  if ( m_use_exact_guess )
    register_variable( m_f_name     , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void DensityPredictor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& rho_guess = tsk_info->get_field<CCVariable<double> >( "new_densityGuess");
  CCVariable<double>& rho_guess_a = tsk_info->get_field<CCVariable<double> >( "densityGuess");
  constCCVariable<double>& rho = tsk_info->get_field<constCCVariable<double> >( "densityCP" );
  constCCVariable<double>& eps = tsk_info->get_field<constCCVariable<double > >( "volFraction" );

  constSFCXVariable<double>& u = tsk_info->get_field<constSFCXVariable<double> >( "uVelocitySPBC" );
  constSFCYVariable<double>& v = tsk_info->get_field<constSFCYVariable<double> >( "vVelocitySPBC" );
  constSFCZVariable<double>& w = tsk_info->get_field<constSFCZVariable<double> >( "wVelocitySPBC" );

  //---work---
  const double dt = tsk_info->get_dt();

  if ( m_use_exact_guess ){

    constCCVariable<double>& f = tsk_info->get_field<constCCVariable<double> >( m_f_name );

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){

      rho_guess(i,j,k) = ( m_rho1 - rho(i,j,k) * f(i,j,k) * ( m_rho1/m_rho0 - 1.)) * eps(i,j,k);

    });

  } else {

    Vector Dx = patch->dCell();
    const double Aew = Dx.y() * Dx.z();
    const double Ans = Dx.z() * Dx.x();
    const double Atb = Dx.x() * Dx.y();
    const double vol = Dx.x() * Dx.y() * Dx.z();

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){

      const double rho_u_e = (rho(i,j,k) + rho(i+1,j,k))/2. * u(i+1,j,k);
      const double rho_u_w = (rho(i,j,k) + rho(i-1,j,k))/2. * u(i,j,k);
      const double rho_v_n = (rho(i,j,k) + rho(i,j+1,k))/2. * v(i,j+1,k);
      const double rho_v_s = (rho(i,j,k) + rho(i,j-1,k))/2. * v(i,j,k);
      const double rho_w_t = (rho(i,j,k) + rho(i,j,k+1))/2. * w(i,j,k+1);
      const double rho_w_b = (rho(i,j,k) + rho(i,j,k-1))/2. * w(i,j,k);

      rho_guess(i,j,k) = ( rho(i,j,k) - dt * ( (rho_u_e - rho_u_w) * Aew
                                             + (rho_v_n - rho_v_s) * Ans
                                             + (rho_w_t - rho_w_b) * Atb ) / vol ) * eps(i,j,k);

    });

    //adding extra mass sources
    for (auto i = m_mass_sources.begin(); i != m_mass_sources.end(); i++ ){

      constCCVariable<double>& src = tsk_info->get_field<constCCVariable<double> >( *i );
      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        rho_guess(i,j,k) = rho_guess(i,j,k) + dt * src(i,j,k);
      });
    }
  }

  //this kludge is needed until the old version goes away...
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    rho_guess_a(i,j,k) = rho_guess(i,j,k);
  });
}
} //namespace Uintah
