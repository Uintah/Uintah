#include <CCA/Components/Arches/PropertyModelsV2/CoalTemperatureNebo.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyHelper.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;
using namespace std;

CoalTemperatureNebo::CoalTemperatureNebo( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 

  _pi = acos(-1.0);
  _Rgas = 8314.3; // J/K/kmol
}

CoalTemperatureNebo::~CoalTemperatureNebo(){ 
}

void 
CoalTemperatureNebo::problemSetup( ProblemSpecP& db ){ 


  const ProblemSpecP db_root = db->getRootNode(); 
  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties") ){ 

    ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties");

    db_coal_props->require("density",_rhop_o);
    db_coal_props->require("diameter_distribution", _sizes); 
    db_coal_props->require("temperature", _initial_temperature); 
    db_coal_props->require("ash_enthalpy", _Ha0); 
    db_coal_props->require("char_enthalpy", _Hh0); 
    db_coal_props->require("raw_coal_enthalpy", _Hc0); 

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
      double raw_coal_mf = coal_daf / coal_dry; 
      double char_mf = coal.CHAR / coal_dry; 
      double ash_mf = coal.ASH / coal_dry; 

      _MW_avg = (coal.C/coal_daf)/12.0107 + (coal.H/coal_daf)/1.00794 + (coal.O/coal_daf)/16 + (coal.N/coal_daf)/14.0067 + (coal.S/coal_daf)/32.065;
      _MW_avg = 1/_MW_avg;
      _RdC = _Rgas/12.0107;
      _RdMW = _Rgas/_MW_avg; 

      _init_char.clear(); 
      _init_rawcoal.clear(); 
      _init_ash.clear(); 
      _denom.clear(); 

      _Nenv = _sizes.size(); 

      for ( unsigned int i = 0; i < _sizes.size(); i++ ){ 

        double mass_dry = (_pi/6.0) * pow(_sizes[i],3) * _rhop_o;     // kg/particle
        _init_ash.push_back(mass_dry  * ash_mf);                      // kg_ash/particle (initial)  
        _init_char.push_back(mass_dry * char_mf);                     // kg_char/particle (initial)
        _init_rawcoal.push_back(mass_dry * raw_coal_mf);              // kg_ash/particle (initial)
        _denom.push_back( _init_ash[i] + 
                          _init_char[i] + 
                          _init_rawcoal[i] );

      }
    } else { 
      throw ProblemSetupException("Error: No <ultimate_analysis> found in input file.", __FILE__, __LINE__); 
    }

    _rawcoal_base_name = PropertyHelper::parse_for_role_to_label(db, "raw_coal"); 
    _char_base_name = PropertyHelper::parse_for_role_to_label(db, "char"); 
    _enthalpy_base_name = PropertyHelper::parse_for_role_to_label(db, "enthalpy"); 
    _dTdt_base_name = PropertyHelper::parse_for_role_to_label(db, "dTdt"); 

  } else { 
    throw ProblemSetupException("Error: <Coal> is missing the <Properties> section.", __FILE__, __LINE__);
  }


}

void 
CoalTemperatureNebo::create_local_labels(){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string temperature_name = get_env_name( i, _task_name );
    register_new_variable( temperature_name, CC_DOUBLE ); 
    const std::string dTdt_name = get_env_name( i, _dTdt_base_name );
    register_new_variable( dTdt_name, CC_DOUBLE ); 

  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
CoalTemperatureNebo::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );


    register_variable( char_name , CC_DOUBLE , REQUIRES , 0 , NEWDW , variable_registry );
    register_variable( enthalpy_name , CC_DOUBLE , REQUIRES , 0 , NEWDW , variable_registry );
    register_variable( rc_name   , CC_DOUBLE , REQUIRES , 0 , NEWDW , variable_registry );
    register_variable( temperature_name  , CC_DOUBLE , COMPUTES , variable_registry );
    register_variable( dTdt_name  , CC_DOUBLE , COMPUTES , variable_registry );

  }

}

void 
CoalTemperatureNebo::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
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
CoalTemperatureNebo::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string temperatureold_name  = get_env_name( i, _task_name );

    register_variable( temperature_name , CC_DOUBLE, COMPUTES, variable_registry );
    register_variable( temperatureold_name , CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry );

  }

}

void 
CoalTemperatureNebo::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
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
CoalTemperatureNebo::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name, CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry );
    register_variable( temperature_name, CC_DOUBLE, REQUIRES, 0, OLDDW, variable_registry );
    register_variable( enthalpy_name, CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry );
    register_variable( rc_name  , CC_DOUBLE, REQUIRES, 0, LATEST, variable_registry );
    register_variable( temperature_name , CC_DOUBLE, MODIFIES, variable_registry );
    register_variable( dTdt_name , CC_DOUBLE, COMPUTES, variable_registry );

  }

}

void 
CoalTemperatureNebo::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
    SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  for ( int i = 0; i < _Nenv; i++ ){ 
    const std::string temperature_name  = get_env_name( i, _task_name );
    const std::string dTdt_name  = get_env_name( i, _dTdt_base_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string enthalpy_name = get_env_name( i, _enthalpy_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );
    const double dt = tsk_info->get_dt();

    SVolFP temperature   = tsk_info->get_so_field<SVolF>( temperature_name );
    SVolFP dTdt   = tsk_info->get_so_field<SVolF>( dTdt_name );
    SVolFP temperatureolddw   = tsk_info->get_const_so_field<SVolF>( temperature_name );
    SVolFP enthalpy = tsk_info->get_const_so_field<SVolF>( enthalpy_name );
    SVolFP cchar = tsk_info->get_const_so_field<SVolF>( char_name );
    SVolFP rc    = tsk_info->get_const_so_field<SVolF>( rc_name );

    // Newton's method
    SpatialOps::SpatFldPtr<SVolF> hint = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> Ha = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> Hc = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> Hh = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> H = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> tol = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> f1 = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> f2 = SpatialFieldStore::get<SVolF>(*temperature); 
    SpatialOps::SpatFldPtr<SVolF> dT = SpatialFieldStore::get<SVolF>(*temperature);//temperature change for each step 
    SpatialOps::SpatFldPtr<SVolF> oldtemperature = SpatialFieldStore::get<SVolF>(*temperature);//see the change in T

    int icount = 0;
    double delta = 1;
    double tolMax = 1;
    for ( int iter = 0; iter < 15; iter++) {
      icount++;
      *oldtemperature <<= *temperature;
      // compute enthalpy given Tguess
      *hint <<= -156.076 + 380/(-1 + exp(380 / *temperature)) + 3600/(-1 + exp(1800 / *temperature));
      *Ha <<= -202849.0 + _Ha0 + *temperature * (593. + *temperature * 0.293); 
      *Hc <<= _Hc0 + *hint * _RdMW;
      *Hh <<= _Hh0 + *hint * _RdC;
      *H <<= *Hc * (*rc + min(0.0,*cchar)) + *Hh * max(0.0,*cchar) + *Ha * _init_ash[i];
      *f1 <<= *enthalpy - *H;    
      // compute enthalpy given Tguess + delta
      *temperature <<= *temperature + delta;
      *hint <<= -156.076 + 380/(-1 + exp(380 / *temperature)) + 3600/(-1 + exp(1800 / *temperature));
      *Ha <<= -202849.0 + _Ha0 + *temperature * (593. + *temperature * 0.293); 
      *Hc <<= _Hc0 + *hint * _RdMW;
      *Hh <<= _Hh0 + *hint * _RdC;
      *H <<= *Hc * (*rc + min(0.0,*cchar)) + *Hh * max(0.0,*cchar) + *Ha * _init_ash[i];
      *f2 <<= *enthalpy - *H;
      // correct temperature
      *dT <<= *f1 * delta / (*f2-*f1) + delta; 
      *temperature <<= *temperature - *dT;    //to add an coefficient for steadness
      // check to see if tolernace has been met
      tolMax = field_max_interior(*oldtemperature - *temperature);
      if (abs(tolMax) < 0.01 ) 
        break;
    }
    // if the temperature calculation is above or below reasonable values we will
    // assume the weights were too small and reset to the initial temperature
    *temperature <<= cond( *temperature > 3000.0, _initial_temperature )
                         ( *temperature < 290.0, _initial_temperature )
                         ( *temperature );
    *dTdt <<= (*temperature-*temperatureolddw)/dt;
  }
}

