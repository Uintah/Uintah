#include <CCA/Components/Arches/PropertyModelsV2/DensityStar.h>
#include <CCA/Components/Arches/KokkosTools.h>
#include <CCA/Components/Arches/UPSHelper.h>


namespace Uintah{

//--------------------------------------------------------------------------------------------------
DensityStar::DensityStar( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
DensityStar::~DensityStar(){
}

//--------------------------------------------------------------------------------------------------
void
DensityStar::problemSetup( ProblemSpecP& db ){

  using namespace ArchesCore;
  m_label_density = parse_ups_for_role( DENSITY, db, "density" );
  m_label_densityStar = m_label_density + "_star" ;
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
DensityStar::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_densityStar );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::NEWDW, variable_registry);

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& rhoStar = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_densityStar );
  constCCVariable<double>& rho = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_label_density );

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    rhoStar(i,j,k)   = rho(i,j,k);
  });

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_label_densityStar , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::OLDDW, variable_registry);

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& rhoStar = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_densityStar );
  constCCVariable<double>& old_rhoStar = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_label_densityStar );
  rhoStar.copyData(old_rhoStar);

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( "x-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "y-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "z-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_label_density , ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );

  register_variable( m_label_densityStar, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");

  constCCVariable<double>& old_rho = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_label_density,ArchesFieldContainer::OLDDW);
  CCVariable<double>& rho = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_density );
  CCVariable<double>& rhoStar = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_densityStar );

  const int time_substep = tsk_info->get_time_substep();
  const double dt = tsk_info->get_dt();

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double vol       = DX.x()*DX.y()*DX.z();

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

    rhoStar(i,j,k)   = rho(i,j,k) - ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                                      area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) )+
                                      area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) )) * dt / vol;

    rhoStar(i,j,k) = _alpha[time_substep] * old_rho(i,j,k) + _beta[time_substep] * rhoStar(i,j,k);

    rho(i,j,k)  = rhoStar(i,j,k); // I am copy density guess in density

  });
}

} //namespace Uintah
