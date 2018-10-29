#include <CCA/Components/Arches/PropertyModelsV2/ContinuityPredictor.h>
#include <CCA/Components/Arches/KokkosTools.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
ContinuityPredictor::ContinuityPredictor( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
ContinuityPredictor::~ContinuityPredictor(){
}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::problemSetup( ProblemSpecP& db ){

  m_label_balance = "continuity_balance";

  if (db->findBlock("KMomentum")->findBlock("use_drhodt")){

    db->findBlock("KMomentum")->findBlock("use_drhodt")->getAttribute("label",m_label_drhodt);

  } else {

    m_label_drhodt = "drhodt";

  }

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_label_balance );

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_label_balance , ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& Balance = *(tsk_info->get_uintah_field<CCVariable<double> >( m_label_balance ));
  Balance.initialize(0.0);

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const bool packed_tasks ){

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( "x-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "y-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( "z-mom", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  register_variable( m_label_drhodt , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  register_variable( m_label_balance , ArchesFieldContainer::COMPUTES, variable_registry, time_substep );


}

//--------------------------------------------------------------------------------------------------
void
ContinuityPredictor::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");

  constCCVariable<double>& drho_dt = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_label_drhodt ));
  CCVariable<double>& Balance = *(tsk_info->get_uintah_field<CCVariable<double> >( m_label_balance ));
  Balance.initialize(0.0);
  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double vol     = DX.x()*DX.y()*DX.z();

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    Balance(i,j,k) = vol*drho_dt(i,j,k) + ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                                            area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) )+
                                            area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) ));
  });
}
} //namespace Uintah
