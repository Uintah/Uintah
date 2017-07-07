#include <CCA/Components/Arches/ParticleModels/TotNumDensity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
TotNumDensity::TotNumDensity( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::problemSetup( ProblemSpecP& db ){

  bool doing_dqmom = ParticleTools::check_for_particle_method(db,ParticleTools::DQMOM);
  bool doing_cqmom = ParticleTools::check_for_particle_method(db,ParticleTools::CQMOM);

  if ( doing_dqmom ){
    _Nenv = ParticleTools::get_num_env( db, ParticleTools::DQMOM );
  } else if ( doing_cqmom ){
    _Nenv = ParticleTools::get_num_env( db, ParticleTools::CQMOM );
  } else {
    throw ProblemSetupException(
      "Error: This method only working for DQMOM/CQMOM.",__FILE__,__LINE__);
  }

}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::create_local_labels(){

  register_new_variable<CCVariable<double> >( _task_name );

}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool pack_tasks ){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){
    const std::string weight_name  = ParticleTools::append_env( "w", ienv);
    register_variable(
      weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& num_den = *(tsk_info->get_uintah_field<CCVariable<double> >( _task_name ));
  num_den.initialize(0.0);

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){


    const std::string weight_name = ParticleTools::append_env( "w", ienv);
    constCCVariable<double>& weight =
      tsk_info->get_const_uintah_field_add<constCCVariable<double> >( weight_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){

      num_den(i,j,k) += weight(i,j,k);

    });
  }
}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){
    const std::string weight_name  = ParticleTools::append_env( "w", ienv);
    register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::NEWDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
void
TotNumDensity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& num_den = *(tsk_info->get_uintah_field<CCVariable<double> >( _task_name ));
  num_den.initialize(0.0);

  for ( int ienv = 0; ienv < _Nenv; ienv++ ){


    const std::string weight_name = ParticleTools::append_env( "w", ienv);
    constCCVariable<double>& weight =
      tsk_info->get_const_uintah_field_add<constCCVariable<double> >( weight_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){

      num_den(i,j,k) += weight(i,j,k);

    });
  }
}
} //namespace Uintah
