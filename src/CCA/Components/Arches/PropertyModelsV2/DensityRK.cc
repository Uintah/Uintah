#include <CCA/Components/Arches/PropertyModelsV2/DensityRK.h>
#include <CCA/Components/Arches/KokkosTools.h>
#include <CCA/Components/Arches/UPSHelper.h>


namespace Uintah{

//--------------------------------------------------------------------------------------------------
DensityRK::DensityRK( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
DensityRK::~DensityRK(){
}

//--------------------------------------------------------------------------------------------------
void
DensityRK::problemSetup( ProblemSpecP& db ){

  using namespace ArchesCore;
  m_label_density = parse_ups_for_role( DENSITY, db, "density" );
  m_label_densityRK = m_label_density + "_rk" ;
  ProblemSpecP db_root = db->getRootNode();

  db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->getAttribute("order", _time_order);

  if ( _time_order == 1 ){

    _alpha.resize(1);
    _beta.resize(1);
    _time_factor.resize(1);

    _alpha[0] = 0.0;

    _beta[0]  = 1.0;

    _time_factor[0] = 1.0;

  } else if ( _time_order == 2 ) {

    _alpha.resize(2);
    _beta.resize(2);
    _time_factor.resize(2);

    _alpha[0]= 0.0;
    _alpha[1]= 0.5;

    _beta[0]  = 1.0;
    _beta[1]  = 0.5;

    _time_factor[0] = 1.0;
    _time_factor[1] = 1.0;

  } else if ( _time_order == 3 ) {

    _alpha.resize(3);
    _beta.resize(3);
    _time_factor.resize(3);

    _alpha[0] = 0.0;
    _alpha[1] = 0.75;
    _alpha[2] = 1.0/3.0;

    _beta[0]  = 1.0;
    _beta[1]  = 0.25;
    _beta[2]  = 2.0/3.0;

    _time_factor[0] = 1.0;
    _time_factor[1] = 0.5;
    _time_factor[2] = 1.0;

  } else {
    throw InvalidValue("Error: <TimeIntegrator> must have value: 1, 2, or 3 (representing the order).",__FILE__, __LINE__);
  }

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_densityRK );

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityRK , ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::NEWDW, variable_registry);

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& rhoRK = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_densityRK );
  constCCVariable<double>& rho = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_label_density );

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    rhoRK(i,j,k)   = rho(i,j,k);
  });

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){


  register_variable( m_label_density , ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );

  register_variable( m_label_densityRK, ArchesFieldContainer::COMPUTES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
DensityRK::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& old_rho = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_label_density,ArchesFieldContainer::OLDDW);
  CCVariable<double>& rho = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_density );
  CCVariable<double>& rhoRK = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_densityRK );

  const int time_substep = tsk_info->get_time_substep();

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

    rhoRK(i,j,k) = _alpha[time_substep] * old_rho(i,j,k) + _beta[time_substep] * rho(i,j,k);

    rho(i,j,k)  = rhoRK(i,j,k); // I am copy density guess in density

  });
}

} //namespace Uintah
