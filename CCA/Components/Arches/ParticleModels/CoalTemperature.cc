#include <CCA/Components/Arches/ParticleModels/CoalTemperature.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;
using namespace std;

CoalTemperature::CoalTemperature( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {

  _pi = acos(-1.0);
  _Rgas = 8314.3; // J/K/kmol
}

CoalTemperature::~CoalTemperature(){
}

void
CoalTemperature::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();
  db->getWithDefault("const_size",_const_size,true);
  
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

      _MW_avg = (coal.C/coal_daf)/12.01 + (coal.H/coal_daf)/1.008 + (coal.O/coal_daf)/16 + (coal.N/coal_daf)/14.01 + (coal.S/coal_daf)/32.06;
      _MW_avg = 1/_MW_avg;
      _RdC = _Rgas/12.01;
      _RdMW = _Rgas/_MW_avg;

      _init_char.clear();
      _init_rawcoal.clear();
      _init_ash.clear();
      _denom.clear();

//      _Nenv = _sizes.size();

      for ( unsigned int i = 0; i < _sizes.size(); i++ ){

        double mass_dry = (_pi/6.0) * pow(_sizes[i],3) * _rhop_o;     // kg/particle
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

    _diameter_base_name = ParticleTools::parse_for_role_to_label(db, "size");
    _rawcoal_base_name = ParticleTools::parse_for_role_to_label(db, "raw_coal");
    _char_base_name = ParticleTools::parse_for_role_to_label(db, "char");
    _enthalpy_base_name = ParticleTools::parse_for_role_to_label(db, "enthalpy");
    _dTdt_base_name = ParticleTools::parse_for_role_to_label(db, "dTdt");
    _gas_temperature_name = "temperature";
    _vol_fraction_name = "volFraction";

  } else {
    throw ProblemSetupException("Error: <Coal> is missing the <Properties> section.", __FILE__, __LINE__);
  }


}

void
CoalTemperature::create_local_labels(){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name = get_env_name( i, _task_name );
    register_new_variable<CCVariable<double> >( temperature_name );
    const std::string dTdt_name = get_env_name( i, _dTdt_base_name );
    register_new_variable<CCVariable<double> >( dTdt_name );

  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
CoalTemperature::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name  = get_env_name( i, _task_name );
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

void
CoalTemperature::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
    SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name = get_env_name( i, _task_name );
    SVolFP temperature = tsk_info->get_so_field<SVolF>( temperature_name );
    const std::string dTdt_name = get_env_name( i, _dTdt_base_name );
    SVolFP dTdt = tsk_info->get_so_field<SVolF>( dTdt_name );

    *temperature <<= _initial_temperature;
    *dTdt <<= 0.0;

  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
CoalTemperature::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string temperatureold_name  = get_env_name( i, _task_name );

    register_variable( temperature_name , ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( temperatureold_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry );

  }

}

void
CoalTemperature::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
    SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string temperatureold_name  = get_env_name( i, _task_name );

    SVolFP temperature   = tsk_info->get_so_field<SVolF>( temperature_name );
    SVolFP temperatureold   = tsk_info->get_const_so_field<SVolF>( temperatureold_name );

    *temperature <<= *temperatureold;
  }
}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
CoalTemperature::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry );
    register_variable( temperature_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
    register_variable( enthalpy_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry );
    register_variable( rc_name  , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry );
    register_variable( temperature_name , ArchesFieldContainer::MODIFIES, variable_registry );
    register_variable( dTdt_name , ArchesFieldContainer::COMPUTES, variable_registry );
    
    if ( !_const_size ) {
      const std::string diameter_name   = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name, ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
    }

  }
  const std::string gas_temperature_name   = _gas_temperature_name;
  register_variable( gas_temperature_name   , ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST , variable_registry );
  register_variable( _vol_fraction_name, ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::LATEST, variable_registry );
}

void
CoalTemperature::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
    SpatialOps::OperatorDatabase& opr ){
  const std::string gas_temperature_name   = _gas_temperature_name;
  constCCVariable<double>* vgas_temperature = tsk_info->get_const_uintah_field<constCCVariable<double> >(gas_temperature_name);
  constCCVariable<double>& gas_temperature = *vgas_temperature;
  constCCVariable<double>* vvol_frac = tsk_info->get_const_uintah_field<constCCVariable<double> >(_vol_fraction_name);
  constCCVariable<double>& vol_frac = *vvol_frac;
  for ( int i = 0; i < _Nenv; i++ ){
    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );
    const double dt = tsk_info->get_dt(); // this is from the old dw.. so we have [T^t-T^(t-1)]/[t-(t-1)]

    CCVariable<double>* vtemperature = tsk_info->get_uintah_field<CCVariable<double> >(temperature_name);
    CCVariable<double>* vdTdt = tsk_info->get_uintah_field<CCVariable<double> >(dTdt_name);
    constCCVariable<double>* vrcmass = tsk_info->get_const_uintah_field<constCCVariable<double> >(rc_name);
    constCCVariable<double>* vchar = tsk_info->get_const_uintah_field<constCCVariable<double> >(char_name);
    constCCVariable<double>* venthalpy = tsk_info->get_const_uintah_field<constCCVariable<double> >(enthalpy_name);
    constCCVariable<double>* vtemperatureold = tsk_info->get_const_uintah_field<constCCVariable<double> >(temperature_name);
    CCVariable<double>& temperature = *vtemperature;
    CCVariable<double>& dTdt = *vdTdt;
    constCCVariable<double>& temperatureold = *vtemperatureold;
    constCCVariable<double>& rcmass = *vrcmass;
    constCCVariable<double>& charmass = *vchar;
    constCCVariable<double>& enthalpy = *venthalpy;

    constCCVariable<double>* vdiameter;
    if ( !_const_size ) {
      const std::string diameter_name = get_env_name( i, _diameter_base_name );
      vdiameter = tsk_info->get_const_uintah_field<constCCVariable<double> >(diameter_name);
    }
    constCCVariable<double>& diameter = *vdiameter;
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      int icount = 0;
      double delta = 1;

      double tol = 1;
      double hint = 0.0;
      double Ha = 0.0;
      double Hc = 0.0;
      //double Hh = 0.0; unused
      double H = 0.0;
      double f1 = 0.0;
      double f2 = 0.0;
      double dT = 0.0;

      double pT = temperature[c];
      double gT = gas_temperature[c];
      double pT_olddw = temperatureold[c];
      double oldpT = temperature[c];
      double RC = rcmass[c];
      double CH = charmass[c];
      double pE = enthalpy[c];
      double vf = vol_frac[c];

      double massDry;
      double initAsh;
      double dp;
      
      if (vf < 1.0e-10 ){
        temperature[c]=gT; // gas temperature
        dTdt[c]=(pT-pT_olddw)/dt;
      } else {
        int max_iter=15;
        int iter =0;
        
        if ( !_const_size ) {
          dp = diameter[c];
          massDry = _pi/6.0 * pow( dp, 3.0 ) * _rhop_o;
          initAsh = massDry * _ash_mf;
        } else {
          initAsh = _init_ash[i];
        }
        
        if ( initAsh > 0.0 ) {
          for ( ; iter < max_iter; iter++) {
            icount++;
            oldpT = pT;
            // compute enthalpy given Tguess
            hint = -156.076 + 380/(-1 + exp(380 / pT)) + 3600/(-1 + exp(1800 / pT));
            Ha = -202849.0 + _Ha0 + pT * (593. + pT * 0.293);
            Hc = _Hc0 + hint * _RdMW;
            H = Hc * (RC + CH) + Ha * initAsh;
            f1 = pE - H;
            // compute enthalpy given Tguess + delta
            pT = pT + delta;
            hint = -156.076 + 380/(-1 + exp(380 / pT)) + 3600/(-1 + exp(1800 / pT));
            Ha = -202849.0 + _Ha0 + pT * (593. + pT * 0.293);
            Hc = _Hc0 + hint * _RdMW;
            H = Hc * (RC + CH) + Ha * initAsh;
            f2 = pE - H;
            // correct temperature
            dT = f1 * delta / (f2-f1) + delta;
            pT = pT - dT;    //to add an coefficient for steadness
            // check to see if tolernace has been met
            tol = abs(oldpT - pT);

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
        
        temperature[c]=pT;
        dTdt[c]=(pT-pT_olddw)/dt;
      }
    }
  }
}
