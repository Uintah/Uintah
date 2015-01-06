#include <CCA/Components/Arches/PropertyModelsV2/ConstEnvProperty.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyHelper.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

ConstEnvProperty::ConstEnvProperty( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

ConstEnvProperty::~ConstEnvProperty(){ 
}

void 
ConstEnvProperty::problemSetup( ProblemSpecP& db ){ 

  db->require("constants", _const); 
  _Nenv = _const.size(); 

}

void 
ConstEnvProperty::create_local_labels(){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name = get_env_name( i, _task_name );
    register_new_variable( rho_name, CC_DOUBLE ); 

  }
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
ConstEnvProperty::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name  = get_env_name( i, _task_name );
    register_variable( rho_name  , CC_DOUBLE , COMPUTES , variable_registry );

  }
}

void 
ConstEnvProperty::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name  = get_env_name( i, _task_name );
    SVolFP rho   = tsk_info->get_so_field<SVolF>( rho_name );

    *rho <<= _const[i];

  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
ConstEnvProperty::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name  = get_env_name( i, _task_name );
    register_variable( rho_name  , CC_DOUBLE , COMPUTES , variable_registry );
    register_variable( rho_name  , CC_DOUBLE , REQUIRES , 0, OLDDW, variable_registry );

  }
}

void 
ConstEnvProperty::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 
  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  for ( int i = 0; i < _Nenv; i++ ){ 

    const std::string rho_name  = get_env_name( i, _task_name );

    SVolFP rho     = tsk_info->get_so_field<SVolF>( rho_name );
    SVolFP old_rho = tsk_info->get_const_so_field<SVolF>( rho_name );

    *rho <<= *old_rho; 

  }
}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
ConstEnvProperty::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

}

void 
ConstEnvProperty::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                SpatialOps::OperatorDatabase& opr ){ 

}
