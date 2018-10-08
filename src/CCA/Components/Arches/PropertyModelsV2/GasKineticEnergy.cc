#include <CCA/Components/Arches/PropertyModelsV2/GasKineticEnergy.h>
#include <CCA/Components/Arches/KokkosTools.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
GasKineticEnergy::GasKineticEnergy( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
GasKineticEnergy::~GasKineticEnergy(){
}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::problemSetup( ProblemSpecP& db ){

  m_u_vel_name = parse_ups_for_role( Uintah::ArchesCore::CCUVELOCITY, db, "CCUVelocity" );
  m_v_vel_name = parse_ups_for_role( Uintah::ArchesCore::CCVVELOCITY, db, "CCVVelocity" );
  m_w_vel_name = parse_ups_for_role( Uintah::ArchesCore::CCWVELOCITY, db, "CCWVelocity" );
  m_kinetic_energy = "gas_kinetic_energy";
  m_max_ke = 1e9 ;
}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_kinetic_energy );

}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
                                       variable_registry, const bool packed_tasks ){

  register_variable( m_kinetic_energy , ArchesFieldContainer::COMPUTES, variable_registry , m_task_name );

}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& ke = *(tsk_info->get_uintah_field<CCVariable<double> >( m_kinetic_energy ));
  ke.initialize(0.0);

}



//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
                                          variable_registry, const int time_substep,
                                          const bool packed_tasks ){

  register_variable( m_u_vel_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_v_vel_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_w_vel_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name );

  register_variable( m_kinetic_energy , ArchesFieldContainer::COMPUTES, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
GasKineticEnergy::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  // cc gas velocities 
  constCCVariable<double>& u = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_w_vel_name);
  constCCVariable<double>& v = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_w_vel_name);
  constCCVariable<double>& w = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_w_vel_name);

  CCVariable<double>& ke = *(tsk_info->get_uintah_field<CCVariable<double> >( m_kinetic_energy ));
  ke.initialize(0.0);
  double ke_p = 0;
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    ke(i,j,k) = 0.5*(u(i,j,k)*u(i,j,k) + v(i,j,k)*v(i,j,k) +w(i,j,k)*w(i,j,k)); 
    ke_p += ke(i,j,k);
  });
  // check if ke is diverging in this patch 
  if ( ke_p > m_max_ke )
    throw InvalidValue("Error: KE is diverging.",__FILE__,__LINE__);
}
} //namespace Uintah
