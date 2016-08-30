#include <CCA/Components/Arches/ParticleModels/CoalMassClip.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

using namespace Uintah;

CoalMassClip::CoalMassClip( std::string task_name, int matl_index, const int N ) :
TaskInterface( task_name, matl_index ), _Nenv(N) {
}

CoalMassClip::~CoalMassClip(){
}

void
CoalMassClip::problemSetup( ProblemSpecP& db ){

  _raw_coal_base = ParticleTools::parse_for_role_to_label(db, "raw_coal");
  _char_base     = ParticleTools::parse_for_role_to_label(db, "char");

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
CoalMassClip::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
CoalMassClip::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
CoalMassClip::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}
//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
CoalMassClip::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string char_name  = ParticleTools::append_env( _char_base, i );
    const std::string rc_name    = ParticleTools::append_env( _raw_coal_base, i );

    register_variable( char_name , ArchesFieldContainer::MODIFIES , 0 , ArchesFieldContainer::NEWDW , variable_registry );
    register_variable( rc_name   , ArchesFieldContainer::MODIFIES , 0 , ArchesFieldContainer::NEWDW , variable_registry );

  }

}

void
CoalMassClip::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for ( int ei = 0; ei < _Nenv; ei++ ){

    const std::string char_name = ParticleTools::append_env( _char_base, ei );
    const std::string rc_name   = ParticleTools::append_env( _raw_coal_base, ei );

    CCVariable<double>& coal_char   = *(tsk_info->get_uintah_field<CCVariable<double> >( char_name ));
    CCVariable<double>& raw_coal    = *(tsk_info->get_uintah_field<CCVariable<double> >( rc_name ));

    CoalHelper& coal_helper = CoalHelper::self();
    CoalHelper::CoalDBInfo& coal_db = coal_helper.get_coal_db();

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      raw_coal(i,j,k) = std::min(std::max(0.0, raw_coal(i,j,k)), coal_db.init_rawcoal[ei]);
      coal_char(i,j,k) = std::max(coal_char(i,j,k), -1.*raw_coal(i,j,k));
    });
  }
}
