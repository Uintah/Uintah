#include <CCA/Components/Arches/ParticleModels/CoalMassClip.h>
#include <CCA/Components/Arches/ParticleModels/ParticleHelper.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

CoalMassClip::CoalMassClip( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {
}

CoalMassClip::~CoalMassClip(){
}

void
CoalMassClip::problemSetup( ProblemSpecP& db ){

  _raw_coal_base = ParticleHelper::parse_for_role_to_label(db, "raw_coal");
  _char_base     = ParticleHelper::parse_for_role_to_label(db, "char");

}

void
CoalMassClip::create_local_labels(){
}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
CoalMassClip::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
CoalMassClip::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
CoalMassClip::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
CoalMassClip::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){
}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
CoalMassClip::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string char_name  = ParticleHelper::append_env( _char_base, i );
    const std::string rc_name    = ParticleHelper::append_env( _raw_coal_base, i );

    register_variable_new( char_name , ArchesFieldContainer::MODIFIES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable_new( rc_name   , ArchesFieldContainer::MODIFIES , 0 , ArchesFieldContainer::NEWDW , variable_registry );

  }

}

void
CoalMassClip::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string char_name = ParticleHelper::append_env( _char_base, i );
    const std::string rc_name   = ParticleHelper::append_env( _raw_coal_base, i );

    SVolFP coal_char   = tsk_info->get_so_field<SVolF>( char_name );
    SVolFP raw_coal    = tsk_info->get_so_field<SVolF>( rc_name );

    CoalHelper& coal_helper = CoalHelper::self();
    CoalHelper::CoalDBInfo& coal_db = coal_helper.get_coal_db();

    *raw_coal  <<= min( max( 0.0, *raw_coal ), coal_db.init_rawcoal[i] );
    *coal_char <<= max(*coal_char, -1.0* *raw_coal);

  }
}
