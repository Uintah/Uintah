#include <CCA/Components/Arches/ParticleModels/CoalDensity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

CoalDensity::CoalDensity( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {

  _pi = acos(-1.0);
}

CoalDensity::~CoalDensity(){
}

void
CoalDensity::problemSetup( ProblemSpecP& db ){

  db->getWithDefault("const_size",_const_size,true);
  
  const ProblemSpecP db_root = db->getRootNode();
  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties") ){

    ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties");

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

    _rawcoal_base_name = ParticleTools::parse_for_role_to_label(db, "raw_coal");
    _char_base_name    = ParticleTools::parse_for_role_to_label(db, "char");
    _diameter_base_name = ParticleTools::parse_for_role_to_label(db, "size");

  } else {
    throw ProblemSetupException("Error: <CoalProperties> required in UPS file to compute a coal density.", __FILE__, __LINE__);
  }


}

void
CoalDensity::create_local_labels(){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string rho_name = get_env_name( i, _task_name );
    register_new_variable<CCVariable<double> >( rho_name );

  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
CoalDensity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string rho_name  = get_env_name( i, _task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name , ArchesFieldContainer::REQUIRES , 0                    , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( rc_name   , ArchesFieldContainer::REQUIRES , 0                    , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( rho_name  , ArchesFieldContainer::COMPUTES , variable_registry );
    
    if ( !_const_size ) {
      const std::string diameter_name = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name , ArchesFieldContainer::REQUIRES , 0              , ArchesFieldContainer::NEWDW , variable_registry );
    }

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

    const std::string rho_name  = get_env_name( i, _task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    SVolFP rho   = tsk_info->get_so_field<SVolF>( rho_name );
    SVolFP cchar = tsk_info->get_const_so_field<SVolF>( char_name );
    SVolFP rc    = tsk_info->get_const_so_field<SVolF>( rc_name );

    SpatialOps::SpatFldPtr<SVolF> ratio = SpatialFieldStore::get<SVolF>(*rho);

    if ( _const_size ) {
      *ratio <<= ( *cchar + *rc + _init_ash[i] ) / _denom[i];

      *rho <<= cond( *ratio > 1.0, _rhop_o )
                   ( *ratio < _init_ash[i]/_denom[i], _init_ash[i]/_denom[i] * _rhop_o )
                   ( *ratio * _rhop_o );
    } else {
      const std::string diameter_name  = get_env_name( i, _diameter_base_name );
      SVolFP dp = tsk_info->get_const_so_field<SVolF>( diameter_name );
      
      SpatialOps::SpatFldPtr<SVolF> initAsh = SpatialFieldStore::get<SVolF>( *rho );
      SpatialOps::SpatFldPtr<SVolF> massDry = SpatialFieldStore::get<SVolF>( *rho );
      SpatialOps::SpatFldPtr<SVolF> denom = SpatialFieldStore::get<SVolF>( *rho );
      
      *massDry <<= _pi/6.0 * pow( *dp, 3.0 ) * _rhop_o;
      *initAsh <<= _ash_mf * *massDry;
      *denom <<= *initAsh + _char_mf * *massDry + _raw_coal_mf * *massDry;
      
      *ratio <<= ( *cchar + *rc + *initAsh) / *denom;
      
      *rho <<= cond( *ratio > 1.0, _rhop_o )
                   ( *ratio < *initAsh/ *denom, *initAsh / *denom * _rhop_o )
                   ( *ratio * _rhop_o );
    }

  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
CoalDensity::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
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
CoalDensity::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string rho_name  = get_env_name( i, _task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    register_variable( char_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST,
      variable_registry );
    register_variable( rc_name  , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST,
      variable_registry );
    register_variable( rho_name , ArchesFieldContainer::COMPUTES, variable_registry );
    
    if ( !_const_size ) {
      const std::string diameter_name = get_env_name( i, _diameter_base_name );
      register_variable( diameter_name , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST , variable_registry );
    }

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

    const std::string rho_name  = get_env_name( i, _task_name );
    const std::string char_name = get_env_name( i, _char_base_name );
    const std::string rc_name   = get_env_name( i, _rawcoal_base_name );

    SVolFP rho   = tsk_info->get_so_field<SVolF>( rho_name );
    SVolFP cchar = tsk_info->get_const_so_field<SVolF>( char_name );
    SVolFP rc    = tsk_info->get_const_so_field<SVolF>( rc_name );

    SpatialOps::SpatFldPtr<SVolF> ratio = SpatialFieldStore::get<SVolF>(*rho);

    if ( _const_size ) {
      *ratio <<= ( *cchar + *rc + _init_ash[i] ) / _denom[i];
      
      *rho <<= cond( *ratio > 1.0, _rhop_o )
                   ( *ratio < _init_ash[i]/_denom[i], _init_ash[i]/_denom[i] * _rhop_o )
                   ( *ratio * _rhop_o );
    } else {
      const std::string diameter_name  = get_env_name( i, _diameter_base_name );
      SVolFP dp = tsk_info->get_const_so_field<SVolF>( diameter_name );
      
      SpatialOps::SpatFldPtr<SVolF> initAsh = SpatialFieldStore::get<SVolF>( *rho );
      SpatialOps::SpatFldPtr<SVolF> massDry = SpatialFieldStore::get<SVolF>( *rho );
      SpatialOps::SpatFldPtr<SVolF> denom = SpatialFieldStore::get<SVolF>( *rho );
      
      *massDry <<= _pi/6.0 * pow( *dp, 3.0 ) * _rhop_o;
      *initAsh <<= _ash_mf * *massDry;
      *denom <<= *initAsh + _char_mf * *massDry + _raw_coal_mf * *massDry;
      
      *ratio <<= ( *cchar + *rc + *initAsh) / *denom;
      
      *rho <<= cond( *ratio > 1.0, _rhop_o )
                   ( *ratio < *initAsh/ *denom, *initAsh / *denom * _rhop_o )
                   ( *ratio * _rhop_o );
    }

 //   SVolF::iterator it = ratio->interior_begin();
 //   for (; it != ratio->interior_end(); it++){
 //     std::cout << "ratio= " << *it << std::endl;
 //   }

  }
}
