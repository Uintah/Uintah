#include <CCA/Components/Arches/ChemMixV2/ColdFlowProperties.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
ColdFlowProperties::ColdFlowProperties( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
ColdFlowProperties::~ColdFlowProperties(){

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::problemSetup( ProblemSpecP& db ){

  for ( ProblemSpecP db_prop = db->findBlock("property");
	db_prop.get_rep() != nullptr;
        db_prop = db_prop->findNextBlock("property") ){

    std::string label;
    bool inverted = false;
    //double value0;
    //double value1;
    double m_rho0;
    double m_rho1;
    
    db_prop->getAttribute("label", label);
    //db_prop->getAttribute("rho0", value0);
    //db_prop->getAttribute("rho1", value1);
    db->getWithDefault( "rho0", m_rho0, 20.0);
    db->getWithDefault( "rho1", m_rho1, 1.0);
    
    inverted = db_prop->findBlock("volumetric");

//    SpeciesInfo info{ value0, value1, inverted };
    SpeciesInfo info{ m_rho0, m_rho1, inverted };

    m_name_to_value.insert( std::make_pair( label, info ));

  }

  db->findBlock("mixture_fraction")->getAttribute("label", m_mixfrac_label );

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::create_local_labels(){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_new_variable<CCVariable<double> >( i->first );
  }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::register_initialize( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  eval(patch, tsk_info );
//  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
//    CCVariable<double>& var = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
//    var.initialize(0.0);
//  }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( i->first, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    constCCVariable<double>& old_var = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( i->first );
    CCVariable<double>& var = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );

    var.copyData(old_var);
  }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

  // for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
  //   register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  // }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  
  // for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
  //   CCVariable<double>& var = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
  //   var.initialize(i->second);
  // }

}

void ColdFlowProperties::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

void ColdFlowProperties::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& f =
    tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_mixfrac_label );

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){

    CCVariable<double>& prop = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
    const SpeciesInfo info = i->second;

    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

      Uintah::parallel_for( range, [&]( int i, int j, int k ){
        prop(i,j,k) = 1./(f(i,j,k) / info.rho1 + ( 1. - f(i,j,k) ) / info.rho0);
    });
  }
}

void ColdFlowProperties::register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::MODIFIES, variable_registry, time_substep  );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry, time_substep  );

}

void ColdFlowProperties::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double> f = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_mixfrac_label );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    //Get the iterator
    Uintah::Iterator cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
    //std::string facename = i_bc->second.name;

    IntVector iDir = patch->faceDirection( i_bc->second.face );

    for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){

      CCVariable<double>& prop = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
      const SpeciesInfo info = i->second;

      for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){

        IntVector c = *cell_iter;
        IntVector cp = *cell_iter - iDir;

        //const double f_interp = 0.5 *( f[c] + f[cp] );

        //const double value = ( info.volumetric ) ?
        //                     1./(f_interp / info.rho1 + ( 1. - f_interp ) / info.rho0) :
        //                     f_interp * info.rho1 + ( 1. - f_interp ) * info.rho0;

        //prop[c] = 2. * value - prop[cp];
        prop[c] = 1./(f[c]/info.rho1 + (1.-f[c])/info.rho0);

      }
    }
  }
}
