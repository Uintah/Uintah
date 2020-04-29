#include <CCA/Components/Arches/ParticleModels/CoalTemperature.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalTemperature::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalTemperature::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &CoalTemperature::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &CoalTemperature::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &CoalTemperature::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalTemperature::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &CoalTemperature::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &CoalTemperature::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &CoalTemperature::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalTemperature::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &CoalTemperature::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &CoalTemperature::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &CoalTemperature::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace CoalTemperature::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
CoalTemperature::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();
  db->getWithDefault("const_size",_const_size,true);

  // Make this model aware of the RK stepping so that dT/dt is computed correctly at the end of
  // each stage. Ideally, this would be done in some central location, which is in the works.
  std::string order;
  db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->findBlock("ExplicitIntegrator")->getAttribute("order", order);

  if ( order == "first" ){
    _time_factor.resize(1);
    _time_factor[0] = 1.;
  } else if ( order == "second" ){
    _time_factor.resize(2);
    _time_factor[0] = 1.;
    _time_factor[1] = 1.;
  } else {
    _time_factor.resize(3);
    _time_factor[0] = 1.;
    _time_factor[1] = 0.5;
    _time_factor[2] = 1.;
  }

  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){

    ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

    db_coal_props->require("density",_rhop_o);
    db_coal_props->require("diameter_distribution", _sizes);
    db_coal_props->require("temperature", _initial_temperature);
    db_coal_props->require("ash_enthalpy", _Ha0);
    db_coal_props->require("char_enthalpy", _Hh0);
    db_coal_props->require("raw_coal_enthalpy", _Hc0);

    if ( db_coal_props->findBlock("ultimate_analysis")){

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
      double raw_coal_mf = coal_daf / coal_dry;
      double char_mf = coal.CHAR / coal_dry;
      _ash_mf = coal.ASH / coal_dry;

      _MW_avg = (coal.C/coal_daf)/12.01 + (coal.H/coal_daf)/1.008 + (coal.O/coal_daf)/16.0 + (coal.N/coal_daf)/14.01 + (coal.S/coal_daf)/32.06;
      _MW_avg = 1.0/_MW_avg;
      _RdC = _Rgas/12.01;
      _RdMW = _Rgas/_MW_avg;

      _init_char.clear();
      _init_rawcoal.clear();
      _init_ash.clear();
      _denom.clear();

//      _Nenv = _sizes.size();

      for ( unsigned int i = 0; i < _sizes.size(); i++ ){

        double mass_dry = (_pi/6.0) * std::pow(_sizes[i],3.0) * _rhop_o;     // kg/particle
        _init_ash.push_back(mass_dry  * _ash_mf);                     // kg_ash/particle (initial)
        _init_char.push_back(mass_dry * char_mf);                     // kg_char/particle (initial)
        _init_rawcoal.push_back(mass_dry * raw_coal_mf);              // kg_ash/particle (initial)
        _denom.push_back( _init_ash[i] +
                          _init_char[i] +
                          _init_rawcoal[i] );

      }
    } else {
      throw ProblemSetupException("Error: No <ultimate_analysis> found in input file.", __FILE__, __LINE__);
    }

    _diameter_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
    _rawcoal_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
    _char_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
    _enthalpy_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ENTHALPY);
    _dTdt_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DTDT);
    _gas_temperature_name = "temperature";
    _vol_fraction_name = "volFraction";

  } else {
    throw ProblemSetupException("Error: <Coal> is missing the <Properties> section.", __FILE__, __LINE__);
  }


}

//--------------------------------------------------------------------------------------------------
void
CoalTemperature::create_local_labels(){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name = get_env_name( i, m_task_name );
    register_new_variable<CCVariable<double> >( temperature_name );
    const std::string dTdt_name = get_env_name( i, _dTdt_base_name );
    register_new_variable<CCVariable<double> >( dTdt_name );

  }
}

//--------------------------------------------------------------------------------------------------
void
CoalTemperature::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name  = get_env_name( i, m_task_name );
    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( enthalpy_name , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( rc_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( temperature_name  , ArchesFieldContainer::COMPUTES , variable_registry );
    register_variable( dTdt_name  , ArchesFieldContainer::COMPUTES , variable_registry );

    if ( !_const_size ) {
      const std::string diameter_name   = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    }

  }

  const std::string gas_temperature_name  = _gas_temperature_name;
  register_variable( gas_temperature_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW , variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CoalTemperature::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){

    const std::string temperature_name = get_env_name( ienv, m_task_name );
    const std::string dTdt_name = get_env_name( ienv, _dTdt_base_name );
    CCVariable<double>& temperature = tsk_info->get_field<CCVariable<double> >( temperature_name );
    CCVariable<double>& dTdt = tsk_info->get_field<CCVariable<double> >( dTdt_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      temperature(i,j,k) = _initial_temperature;
      dTdt(i,j,k) = 0.0;
    });

  }
}

//--------------------------------------------------------------------------------------------------
void
CoalTemperature::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name  = get_env_name( i, m_task_name );
    const std::string temperatureold_name  = get_env_name( i, m_task_name );

    register_variable( temperature_name , ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( temperatureold_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry );

  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
CoalTemperature::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){

    const std::string temperature_name  = get_env_name( ienv, m_task_name );
    const std::string temperatureold_name  = get_env_name( ienv, m_task_name );

    CCVariable<double>& temperature   = tsk_info->get_field<CCVariable<double> >( temperature_name );
    constCCVariable<double>& temperature_old   = tsk_info->get_field<constCCVariable<double> >( temperatureold_name );
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      temperature(i,j,k) = temperature_old(i,j,k);
    });

  }
}

//--------------------------------------------------------------------------------------------------
void
CoalTemperature::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string temperature_name  = get_env_name( i, m_task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( temperature_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
    register_variable( char_name, ArchesFieldContainer        ::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( enthalpy_name, ArchesFieldContainer    ::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( rc_name , ArchesFieldContainer         ::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( temperature_name, ArchesFieldContainer::MODIFIES, variable_registry );
    register_variable( dTdt_name , ArchesFieldContainer::COMPUTES, variable_registry );

    if ( !_const_size ) {
      const std::string diameter_name   = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name, ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::NEWDW, variable_registry );
    }

  }
  const std::string gas_temperature_name   = _gas_temperature_name;
  register_variable( gas_temperature_name , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
  register_variable( _vol_fraction_name, ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW, variable_registry );
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void CoalTemperature::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  const std::string gas_temperature_name   = _gas_temperature_name;
  constCCVariable<double>& gas_temperature = tsk_info->get_field<constCCVariable<double> >(gas_temperature_name);
  constCCVariable<double>& vol_frac        = tsk_info->get_field<constCCVariable<double> >(_vol_fraction_name);

  const double dt = tsk_info->get_dt() * _time_factor[tsk_info->get_time_substep()];

  for ( int ix = 0; ix < _Nenv; ix++ ){

    const std::string temperature_name = get_env_name( ix, m_task_name );
    const std::string dTdt_name        = get_env_name( ix, _dTdt_base_name );
    const std::string char_name        = get_env_name( ix, _char_base_name );
    const std::string enthalpy_name    = get_env_name( ix, _enthalpy_base_name );
    const std::string rc_name          = get_env_name( ix, _rawcoal_base_name );

    CCVariable<double>& temperature         = tsk_info->get_field<CCVariable<double> >(temperature_name);
    CCVariable<double>& dTdt                = tsk_info->get_field<CCVariable<double> >(dTdt_name);
    constCCVariable<double>& rcmass         = tsk_info->get_field<constCCVariable<double> >(rc_name);
    constCCVariable<double>& charmass       = tsk_info->get_field<constCCVariable<double> >(char_name);
    constCCVariable<double>& enthalpy       = tsk_info->get_field<constCCVariable<double> >(enthalpy_name);
    constCCVariable<double>& temperatureold = tsk_info->get_field<constCCVariable<double> >(temperature_name);

    constCCVariable<double>* vdiameter = nullptr;
    if ( !_const_size ) {
      const std::string diameter_name = get_env_name( ix, _diameter_base_name );
      constCCVariable<double>& this_diameter = tsk_info->get_field<constCCVariable<double> >(diameter_name);
      vdiameter = &this_diameter;
    }
    constCCVariable<double>& diameter = *(vdiameter);

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for( range, [&](int i, int j, int k){

      int icount   = 0;
      double delta = 1.0;
      double tol   = 1.0;
      double hint  = 0.0;
      double Ha    = 0.0;
      double Hc    = 0.0;
      double H1    = 0.0;
      double f1    = 0.0;
      double H2    = 0.0;
      double dT    = 0.0;

      double pT       = temperature(i,j,k);
      double gT       = gas_temperature(i,j,k);
      double pT_olddw = temperatureold(i,j,k);
      double oldpT    = temperature(i,j,k);
      double RC       = rcmass(i,j,k);
      double CH       = charmass(i,j,k);
      double pE       = enthalpy(i,j,k);

      double massDry=0.0;
      double initAsh=0.0;
      double dp=0.0;

      if ( vol_frac(i,j,k) < 0.5 ){

        temperature(i,j,k)=gT; // gas temperature
        dTdt(i,j,k)=(pT-pT_olddw)/dt;

      } else {

        int max_iter=15;
        int iter =0;

        if ( !_const_size ) {

          dp = diameter(i,j,k);
          massDry = _pi/6.0 * std::pow( dp, 3.0 ) * _rhop_o;
          initAsh = massDry * _ash_mf;

        } else {

          initAsh = _init_ash[ix];

        }

        if ( initAsh > 0.0 ) {

          for ( ; iter < max_iter; iter++) {
            icount++;
            oldpT = pT;

            // compute enthalpy given Tguess
            hint = -156.076 + 380/(-1 + exp(380 / pT)) + 3600/(-1 + exp(1800 / pT));
            Ha = -202849.0 + _Ha0 + pT * (593. + pT * 0.293);
            Hc = _Hc0 + hint * _RdMW;
            H1 = Hc * (RC + CH) + Ha * initAsh;
            f1 = pE - H1;

            // compute enthalpy given Tguess + delta
            pT = pT + delta;
            hint = -156.076 + 380/(-1 + exp(380 / pT)) + 3600/(-1 + exp(1800 / pT));
            Ha = -202849.0 + _Ha0 + pT * (593. + pT * 0.293);
            Hc = _Hc0 + hint * _RdMW;
            H2 = Hc * (RC + CH) + Ha * initAsh;

            // correct temperature
            dT = f1 * delta / (H1-H2) + delta;
            pT = pT - dT;    //to add an coefficient for steadness

            // check to see if tolernace has been met
            tol = std::abs(oldpT - pT);

            if (tol < 0.01 )
              break;

          }

          if (iter ==max_iter-1 || pT <273.0 || pT > 3500.0 ){

            double pT_low=273;
            hint = -156.076 + 380/(-1 + exp(380 / pT_low)) + 3600/(-1 + exp(1800 / pT_low));
            Ha = -202849.0 + _Ha0 + pT_low * (593. + pT_low * 0.293);
            Hc = _Hc0 + hint * _RdMW;
            double H_low = Hc * (RC + CH) + Ha * initAsh;
            double pT_high=3500;
            hint = -156.076 + 380/(-1 + exp(380 / pT_high)) + 3600/(-1 + exp(1800 / pT_high));
            Ha = -202849.0 + _Ha0 + pT_high * (593. + pT_high * 0.293);
            Hc = _Hc0 + hint * _RdMW;
            double H_high = Hc * (RC + CH) + Ha * initAsh;

            if (pE < H_low || pT < 273.0){

              pT = 273.0;

            } else if (pE > H_high || pT > 3500.0) {

              pT = 3500.0;

            }
          }
        } else {

          pT = _initial_temperature; //prevent nans when dp & ash = 0.0 in cqmom

        }

        temperature(i,j,k)=pT;
        dTdt(i,j,k)=(pT-pT_olddw)/dt;

      }
    });
  }
}
} //namespace Uintah
