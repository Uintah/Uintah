#include <CCA/Components/Arches/ParticleModels/TotNumDensity.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

TotNumDensity::TotNumDensity( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

TotNumDensity::~TotNumDensity(){
}

void
TotNumDensity::problemSetup( ProblemSpecP& db ){

  //only working for DQMOM at the moment

  bool doing_dqmom = ParticleTools::check_for_particle_method(db,ParticleTools::DQMOM);
  bool doing_cqmom = ParticleTools::check_for_particle_method(db,ParticleTools::CQMOM);

  if ( doing_dqmom ){
    _Nenv = ParticleTools::get_num_env( db, ParticleTools::DQMOM );
  } else if ( doing_cqmom ){
    _Nenv = ParticleTools::get_num_env( db, ParticleTools::CQMOM );
  } else {
    throw ProblemSetupException("Error: This method only working for DQMOM/CQMOM.",__FILE__,__LINE__);
  }

}

void
TotNumDensity::create_local_labels(){

  register_new_variable<CCVariable<double> >( _task_name );

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
TotNumDensity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int i = 0; i < _Nenv; i++ ){
    const std::string weight_name  = ParticleTools::append_env( "w", i);
    register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }

}

void
TotNumDensity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  SVolFP num_den = tsk_info->get_so_field<SVolF>( _task_name );
  SpatialOps::SpatFldPtr<SVolF> sum = SpatialFieldStore::get<SVolF>(*num_den);

  *sum <<= 0.0;

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string weight_name  = ParticleTools::append_env( "w", i);
    SVolFP weight   = tsk_info->get_const_so_field<SVolF>( weight_name );

    *sum <<= *sum + *weight;

  }

  *num_den <<= *sum;

}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
TotNumDensity::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  register_variable( _task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int i = 0; i < _Nenv; i++ ){
    const std::string weight_name  = ParticleTools::append_env( "w", i);
    register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry );
  }

}

void
TotNumDensity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

  SVolFP num_den = tsk_info->get_so_field<SVolF>( _task_name );
  SpatialOps::SpatFldPtr<SVolF> sum = SpatialFieldStore::get<SVolF>(*num_den);

  *sum <<= 0.0;

  for ( int i = 0; i < _Nenv; i++ ){

    const std::string weight_name  = ParticleTools::append_env( "w", i);
    SVolFP weight   = tsk_info->get_const_so_field<SVolF>( weight_name );

    *sum <<= *sum + *weight;

  }

  *num_den <<= *sum;

}
