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
    double value0;
    double value1;

    db_prop->getAttribute("label", label);
    db_prop->getAttribute("stream_0", value0);
    db_prop->getAttribute("stream_1", value1);
    inverted = db_prop->findBlock("volumetric");

    SpeciesInfo info{ value0, value1, inverted };

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

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    CCVariable<double>& var = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
    var.initialize(0.0);
  }

  get_properties( patch, tsk_info );

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

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::COMPUTES, variable_registry );
  }

}

//--------------------------------------------------------------------------------------------------
void ColdFlowProperties::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  get_properties( patch, tsk_info );

}

void ColdFlowProperties::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::MODIFIES, variable_registry );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry );

}

void ColdFlowProperties::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  get_properties( patch, tsk_info );

}

void ColdFlowProperties::register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){
    register_variable( i->first, ArchesFieldContainer::MODIFIES, variable_registry );
  }

  register_variable( m_mixfrac_label, ArchesFieldContainer::REQUIRES, 0,
                     ArchesFieldContainer::NEWDW, variable_registry );

}

void ColdFlowProperties::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double> f = tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_mixfrac_label );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());
    if ( !on_this_patch ) continue;

    //Get the iterator
    Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
    std::string facename = i_bc->second.name;

    IntVector iDir = patch->faceDirection( i_bc->second.face );

    for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){

      CCVariable<double>& prop = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
      const SpeciesInfo info = i->second;

      parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (int i,int j,int k) {

        int ip = i-iDir[0];
        int jp = j-iDir[1];
        int kp = k-iDir[2];

        const double f_interp = 0.5 *( f(i,j,k) + f(ip,jp,kp) );

        const double value = ( info.volumetric ) ?
                             1./(f_interp / info.stream_1 + ( 1. - f_interp ) / info.stream_2) :
                             f_interp * info.stream_1 + ( 1. - f_interp ) * info.stream_2;

        prop(i,j,k) = 2. * value - prop(ip,jp,kp);

      });
    }
  }
}

void ColdFlowProperties::get_properties( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& f =
    tsk_info->get_const_uintah_field_add<constCCVariable<double> >( m_mixfrac_label );

  for ( auto i = m_name_to_value.begin(); i != m_name_to_value.end(); i++ ){

    CCVariable<double>& prop = tsk_info->get_uintah_field_add<CCVariable<double> >( i->first );
    const SpeciesInfo info = i->second;

    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

    Uintah::parallel_for( range, [&]( int i, int j, int k ){

      const double value = ( info.volumetric ) ?
                           1./(f(i,j,k) / info.stream_1 + ( 1. - f(i,j,k) ) / info.stream_2) :
                           f(i,j,k) * info.stream_1 + ( 1. - f(i,j,k) ) * info.stream_2;
      prop(i,j,k) = value;

    });

  }
}
