#include <CCA/Components/Arches/ParticleModels/CoalDensity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalDensity::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalDensity::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &CoalDensity::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &CoalDensity::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &CoalDensity::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalDensity::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &CoalDensity::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &CoalDensity::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &CoalDensity::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalDensity::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalDensity::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
CoalDensity::problemSetup( ProblemSpecP& db ){

  db->getWithDefault("model_type",_model_type,"constant_volume_dqmom");

  const ProblemSpecP db_root = db->getRootNode();
  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){

    ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

    db_coal_props->require("density",_rhop_o);
    db_coal_props->require("diameter_distribution", _sizes);

    if ( db_coal_props->findBlock("ultimate_analysis")){

      //<!-- as received mass fractions C+H+O+N+S+char+ash+moisture=1 -->
      ProblemSpecP db_ua = db_coal_props->findBlock("ultimate_analysis");
      CoalAnalysis coal;

      db_ua->require("C",coal.C);
      db_ua->require("H",coal.H);
      db_ua->require("O",coal.O);
      db_ua->require("N",coal.N);
      db_ua->require("S",coal.S);
      db_ua->require("H2O",coal.H2O);
      db_ua->require("ASH",coal.ASH);
      db_ua->require("CHAR",coal.CHAR);

      double coal_daf = coal.C + coal.H + coal.O + coal.N + coal.S; //dry ash free coal
      double coal_dry = coal.C + coal.H + coal.O + coal.N + coal.S + coal.ASH + coal.CHAR; //moisture free coal
      _raw_coal_mf = coal_daf / coal_dry;
      _char_mf = coal.CHAR / coal_dry;
      _ash_mf = coal.ASH / coal_dry;

      _init_char.clear();
      _init_rawcoal.clear();
      _init_ash.clear();
      _denom.clear();

//      _Nenv = _sizes.size();

      for ( unsigned int i = 0; i < _sizes.size(); i++ ){

        double mass_dry = (_pi/6.0) * pow(_sizes[i],3) * _rhop_o;     // kg/particle
        _init_ash.push_back(mass_dry  * _ash_mf);                      // kg_ash/particle (initial)
        _init_char.push_back(mass_dry * _char_mf);                     // kg_char/particle (initial)
        _init_rawcoal.push_back(mass_dry * _raw_coal_mf);              // kg_ash/particle (initial)
        _denom.push_back( _init_ash[i] +
                          _init_char[i] +
                          _init_rawcoal[i] );

      }

    } else {
      throw ProblemSetupException("Error: No <ultimate_analysis> found in input file.", __FILE__, __LINE__);
    }

    _rawcoal_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
    _char_base_name    = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
    _diameter_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);

  } else {
    throw ProblemSetupException("Error: <CoalProperties> required in UPS file to compute a coal density.", __FILE__, __LINE__);
  }


}

//--------------------------------------------------------------------------------------------------
void
CoalDensity::create_local_labels(){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string rho_name = get_env_name( i, m_task_name );
    register_new_variable<CCVariable<double> >( rho_name );

  }
}

//--------------------------------------------------------------------------------------------------
void
CoalDensity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string rho_name  = get_env_name( i, m_task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name , ArchesFieldContainer::REQUIRES , 0                    , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( rc_name   , ArchesFieldContainer::REQUIRES , 0                    , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( rho_name  , ArchesFieldContainer::COMPUTES , variable_registry );

    if ( _model_type != "constant_volume_dqmom" ) {
      const std::string diameter_name = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name , ArchesFieldContainer::REQUIRES , 0              , ArchesFieldContainer::NEWDW , variable_registry );
    }

  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CoalDensity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){

    const std::string rho_name  = get_env_name( ienv, m_task_name );
    const std::string char_name = get_env_name( ienv, _char_base_name );
    const std::string rc_name   = get_env_name( ienv, _rawcoal_base_name );

    CCVariable<double>&      rho   = tsk_info->get_field<CCVariable<double> >( rho_name );
    constCCVariable<double>& cchar = tsk_info->get_field<constCCVariable<double> >( char_name );
    constCCVariable<double>& rc    = tsk_info->get_field<constCCVariable<double> >( rc_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      rho(i,j,k) = 0.0;
    });

    if ( _model_type == "constant_volume_dqmom") {

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double ratio = ( cchar(i,j,k) + rc(i,j,k) + _init_ash[ienv] ) / _denom[ienv];

        //These if's are not optimal for Kokkos, but they don't change the answers from
        //the spatialOps implementation. Perhaps use min/max?
        if ( ratio > 1.0 ) {
          rho(i,j,k) = _rhop_o;
        } else if ( ratio < _init_ash[ienv]/_denom[ienv]){
          rho(i,j,k) = _init_ash[ienv]/_denom[ienv] * _rhop_o;
        } else {
          rho(i,j,k) = ratio*_rhop_o;
        }
      });
    } else if (_model_type == "cqmom") {
      const std::string diameter_name  = get_env_name( ienv, _diameter_base_name );
      constCCVariable<double>& dp = tsk_info->get_field<constCCVariable<double> >( diameter_name );

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
	      const double massDry = _pi/6. * std::pow(dp(i,j,k), 3.) * _rhop_o;
        const double initAsh = _ash_mf * massDry;
        const double denom   = initAsh + _char_mf * massDry + _raw_coal_mf * massDry;
        const double ratio   = (denom > 0.0) ? (cchar(i,j,k) + rc(i,j,k) + initAsh)/denom : 1.01;

        //These if's are not optimal for Kokkos, but they don't change the answers from
        //the spatialOps implementation. Perhaps use min/max?
        if ( ratio > 1.0 ) {
          rho(i,j,k) = _rhop_o;
        } else if ( ratio < initAsh/denom){
          rho(i,j,k) = initAsh/denom*_rhop_o;
        } else {
          rho(i,j,k) = ratio*_rhop_o;
        }

      });
    } else {
      const std::string diameter_name  = get_env_name( ienv, _diameter_base_name );
      constCCVariable<double>& dp = tsk_info->get_field<constCCVariable<double> >( diameter_name );

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
	    const double volume = _pi/6. * dp(i,j,k)*dp(i,j,k)*dp(i,j,k);
	    rho(i,j,k) = ( cchar(i,j,k) + rc(i,j,k) + _init_ash[ienv] ) / volume ;
      });
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
CoalDensity::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string rho_name  = get_env_name( i, m_task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
    register_variable( rc_name  , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
    register_variable( rho_name , ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

    if ( _model_type != "constant_volume_dqmom" ) {
      const std::string diameter_name = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
    }
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CoalDensity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){

    const std::string rho_name  = get_env_name( ienv, m_task_name );
    const std::string char_name = get_env_name( ienv, _char_base_name );
    const std::string rc_name   = get_env_name( ienv, _rawcoal_base_name );

    CCVariable<double>&      rho   = tsk_info->get_field<CCVariable<double> >( rho_name );
    constCCVariable<double>& cchar = tsk_info->get_field<constCCVariable<double> >( char_name );
    constCCVariable<double>& rc    = tsk_info->get_field<constCCVariable<double> >( rc_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      rho(i,j,k) = 0.0;
    });

    if ( _model_type == "constant_volume_dqmom") {
      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        const double ratio = ( cchar(i,j,k) + rc(i,j,k) + _init_ash[ienv] ) / _denom[ienv];

        //These if's are not optimal for Kokkos, but they don't change the answers from
        //the spatialOps implementation. Perhaps use min/max?
        if ( ratio > 1.0 ) {
          rho(i,j,k) = _rhop_o;
        } else if ( ratio <_init_ash[ienv]/_denom[ienv]){
          rho(i,j,k) = _init_ash[ienv]/_denom[ienv] * _rhop_o;
        } else {
          rho(i,j,k) = ratio*_rhop_o;
        }

      });
    } else if (_model_type == "cqmom"){
      const std::string diameter_name  = get_env_name( ienv, _diameter_base_name );
      constCCVariable<double>& dp = tsk_info->get_field<constCCVariable<double> >( diameter_name );

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
	      const double massDry = _pi/6. * std::pow(dp(i,j,k), 3.) * _rhop_o;
        const double initAsh = _ash_mf * massDry;
        const double denom   = initAsh + _char_mf * massDry + _raw_coal_mf * massDry;
        const double ratio   = (denom > 0.0) ? (cchar(i,j,k) + rc(i,j,k) + initAsh)/denom : 1.01;
        //These if's are not optimal for Kokkos, but they don't change the answers from
        //the spatialOps implementation. Perhaps use min/max?
        if ( ratio > 1.0 ) {
          rho(i,j,k) = _rhop_o;
        } else if ( ratio < initAsh/denom){
          rho(i,j,k) = initAsh/denom*_rhop_o;
        } else {
          rho(i,j,k) = ratio*_rhop_o;
        }
      });
    } else {
      const std::string diameter_name  = get_env_name( ienv, _diameter_base_name );
      constCCVariable<double>& dp = tsk_info->get_field<constCCVariable<double> >( diameter_name );

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
	    const double volume = _pi/6. * dp(i,j,k)*dp(i,j,k)*dp(i,j,k);
	    rho(i,j,k) = ( cchar(i,j,k) + rc(i,j,k) + _init_ash[ienv] ) / volume ;
      });
    }
  }
}
} //namespace Uintah
