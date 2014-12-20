#include <CCA/Components/Arches/PropertyModelsV2/CoalDensity.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

CoalDensity::CoalDensity( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 

  _pi = acos(-1.0);
}

CoalDensity::~CoalDensity(){ 
}

void 
CoalDensity::problemSetup( ProblemSpecP& db ){ 


  const ProblemSpecP db_root = db->getRootNode(); 
  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("CoalProperties") ){ 

    ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("CoalProperties");

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
      double raw_coal_mf = coal_daf / coal_dry; 
      double char_mf = coal.CHAR / coal_dry; 
      double ash_mf = coal.ASH / coal_dry; 

      _init_char.clear(); 
      _init_rawcoal.clear(); 
      _init_ash.clear(); 

      _Nenv = _sizes.size(); 
  
      for ( int i = 0; i < _sizes.size(); i++ ){ 

        double mass_dry = (_pi/6.0) * pow(_sizes[i],3) * _rhop_o;     // kg/particle
        _init_ash.push_back(mass_dry  * ash_mf);                      // kg_ash/particle (initial)  
        _init_char.push_back(mass_dry * char_mf);                     // kg_char/particle (initial)
        _init_rawcoal.push_back(mass_dry * raw_coal_mf);              // kg_ash/particle (initial)

      }
      
    } else { 

    }

    if ( db_coal_props->findBlock("raw_coal") ){ 

      _rawcoal_base_name = "NOTSPECIFIED";
      db_coal_props->findBlock("raw_coal")->getAttribute("label",_rawcoal_base_name); 

    } else { 
      throw ProblemSetupException("Error: No <rawcoal> node (and label attribute) found in <CoalProperties>.", __FILE__, __LINE__);
    }
    if ( db_coal_props->findBlock("char") ){ 

      _char_base_name = "NOTSPECIFIED";
      db_coal_props->findBlock("char")->getAttribute("label",_char_base_name); 

    } else { 
      throw ProblemSetupException("Error: No <char> node (and label attribute) found in <CoalProperties>.", __FILE__, __LINE__);
    }


  } else { 
    throw ProblemSetupException("Error: <CoalProperties> required in UPS file to compute a coal density.", __FILE__, __LINE__);
  }


}

void 
CoalDensity::create_local_labels(){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name = get_env_name( i, _density_base_name );
    register_new_variable( rho_name, CC_DOUBLE ); 

  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
CoalDensity::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name = get_env_name( i, _density_base_name );
    const std::string char_name = get_env_name( i, _char_base_name ); 
    const std::string rc_name = get_env_name( i, _rawcoal_base_name ); 

    register_variable( char_name, CC_DOUBLE, REQUIRES, 0, NEWDW, variable_registry ); 
    register_variable( rc_name, CC_DOUBLE, REQUIRES, 0, NEWDW, variable_registry ); 
    register_variable( rho_name, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

}

void 
CoalDensity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name = get_env_name( i, _density_base_name );
    const std::string char_name = get_env_name( i, _char_base_name ); 
    const std::string rc_name = get_env_name( i, _rawcoal_base_name ); 

    SVolFP rho = tsk_info->get_so_field<SVolF>( rho_name );   
    SVolFP cchar = tsk_info->get_so_field<SVolF>( char_name ); 
    SVolFP rc = tsk_info->get_so_field<SVolF>( rc_name ); 

    SpatialOps::SpatFldPtr<SVolF> ratio = SpatialFieldStore::get<SVolF>(*rho); 

    double denom = ( _init_ash[i] + _init_rawcoal[i] + _init_char[i] ); 

    *ratio <<= ( *cchar + *rc + _init_ash[i] ) / denom; 

    *rho <<= cond( *ratio > 1.0, _rhop_o )
                 ( *ratio < 0.0, 0.0 )
                 ( *ratio * _rhop_o ); 


  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
CoalDensity::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 
}

void 
CoalDensity::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 
}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
CoalDensity::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name = get_env_name( i, _density_base_name );
    const std::string char_name = get_env_name( i, _char_base_name ); 
    const std::string rc_name = get_env_name( i, _rawcoal_base_name ); 

    register_variable( char_name, CC_DOUBLE, REQUIRES, 0, NEWDW, variable_registry ); 
    register_variable( rc_name, CC_DOUBLE, REQUIRES, 0, NEWDW, variable_registry ); 
    register_variable( rho_name, CC_DOUBLE, COMPUTES, variable_registry ); 

  }

}

void 
CoalDensity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name  = get_env_name( i, _density_base_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    SVolFP rho = tsk_info->get_so_field<SVolF>( rho_name );   
    SVolFP cchar = tsk_info->get_so_field<SVolF>( char_name ); 
    SVolFP rc = tsk_info->get_so_field<SVolF>( rc_name ); 

    SpatialOps::SpatFldPtr<SVolF> ratio = SpatialFieldStore::get<SVolF>(*rho); 

    double denom = ( _init_ash[i] + _init_rawcoal[i] + _init_char[i] ); 

    *ratio <<= ( *cchar + *rc + _init_ash[i] ) / denom; 

    *rho <<= cond( *ratio > 1.0, _rhop_o )
                 ( *ratio < 0.0, 0.0 )
                 ( *ratio * _rhop_o ); 

  }
}
