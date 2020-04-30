#include <CCA/Components/Arches/PropertyModelsV2/DensityStar.h>
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
  m_label_density = parse_ups_for_role( DENSITY_ROLE, db, "density" );
  m_label_densityStar = m_label_density + "_star" ;

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

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& rhoStar = tsk_info->get_field<CCVariable<double> >( m_label_densityStar );
  rhoStar.initialize(0.0);

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

  register_variable( m_label_densityStar , ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  //register_variable( m_label_densityStar , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::OLDDW, variable_registry, m_task_name );
  register_variable( m_label_density , ArchesFieldContainer::REQUIRES,0, ArchesFieldContainer::OLDDW, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& rhoStar = tsk_info->get_field<CCVariable<double> >( m_label_densityStar );
  constCCVariable<double>& old_rho = tsk_info->get_field<constCCVariable<double> >( m_label_density );
  rhoStar.copyData(old_rho);

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( ArchesCore::default_uMom_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( ArchesCore::default_vMom_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( ArchesCore::default_wMom_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_label_density , ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

  register_variable( m_label_densityStar, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& xmom = tsk_info->get_field<constSFCXVariable<double> >(ArchesCore::default_uMom_name);
  constSFCYVariable<double>& ymom = tsk_info->get_field<constSFCYVariable<double> >(ArchesCore::default_vMom_name);
  constSFCZVariable<double>& zmom = tsk_info->get_field<constSFCZVariable<double> >(ArchesCore::default_wMom_name);

  CCVariable<double>& rho = tsk_info->get_field<CCVariable<double> >( m_label_density );
  CCVariable<double>& rhoStar = tsk_info->get_field<CCVariable<double> >( m_label_densityStar );

  const double dt = tsk_info->get_dt();

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double vol       = DX.x()*DX.y()*DX.z();

  double check_guess_density = 0;
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){

    rhoStar(i,j,k)   = rho(i,j,k) - ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                                      area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) )+
                                      area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) )) * dt / vol;
    if (rhoStar(i,j,k) < 0) {
      check_guess_density = 1;
    }
  });

  if (check_guess_density > 0){
    std::cout << "NOTICE: Negative density guess(es) occurred. Reverting to old density."<< std::endl ;
  } else {
    Uintah::parallel_for( range, [&](int i, int j, int k){
      rho(i,j,k)  = rhoStar(i,j,k); // I am copy density guess in density
    });
  }

}

} //namespace Uintah
