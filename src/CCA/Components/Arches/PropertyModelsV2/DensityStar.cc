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

  register_variable( m_label_densityStar, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
DensityStar::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");

  CCVariable<double>& rho = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_density );
  CCVariable<double>& rhoStar = tsk_info->get_uintah_field_add<CCVariable<double> >( m_label_densityStar );

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
