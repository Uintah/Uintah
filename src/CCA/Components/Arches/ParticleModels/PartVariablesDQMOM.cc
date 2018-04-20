#include <CCA/Components/Arches/ParticleModels/PartVariablesDQMOM.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
PartVariablesDQMOM::PartVariablesDQMOM( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
void
PartVariablesDQMOM::problemSetup( ProblemSpecP& db ){

  m_Nenv = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );

  m_length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
  m_number_density_name   = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TOTNUM_DENSITY);
  m_surfAreaF_root = "surfaceAreaFraction";
}

//--------------------------------------------------------------------------------------------------
void
PartVariablesDQMOM::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_number_density_name);

  for ( int ienv = 0; ienv < m_Nenv; ienv++ ){
    const std::string surfAreaF_name = ArchesCore::append_env( m_surfAreaF_root, ienv);
    register_new_variable<CCVariable<double> >( surfAreaF_name );
  }
}

//--------------------------------------------------------------------------------------------------
void
PartVariablesDQMOM::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool pack_tasks ){

  register_variable( m_number_density_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int ienv = 0; ienv < m_Nenv; ienv++ ){
    const std::string weight_name = ArchesCore::append_env( "w", ienv);
    const std::string length_name = ArchesCore::append_env( m_length_root, ienv);
    const std::string surfAreaF_name = ArchesCore::append_env( m_surfAreaF_root, ienv);

    register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::NEWDW, variable_registry );

    register_variable( length_name, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::NEWDW, variable_registry );

    register_variable( surfAreaF_name, ArchesFieldContainer::COMPUTES, variable_registry );

  }


}

//--------------------------------------------------------------------------------------------------
void
PartVariablesDQMOM::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& num_den  = tsk_info->get_uintah_field_add<CCVariable<double> >( m_number_density_name );
  CCVariable<double>& AreaSumF = tsk_info->get_uintah_field_add< CCVariable<double> >("AreaSum",0);// temporal variable

  AreaSumF.initialize(0.0);
  num_den.initialize(0.0);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  for (int ienv = 0; ienv < m_Nenv; ienv++) {
    const std::string weight_name = ArchesCore::append_env( "w", ienv);
    const std::string length_name = ArchesCore::append_env( m_length_root, ienv);

    constCCVariable<double>& weight =
      tsk_info->get_const_uintah_field_add<constCCVariable<double> >( weight_name );

    constCCVariable<double>& length =
      tsk_info->get_const_uintah_field_add< constCCVariable<double> >(length_name);

    Uintah::parallel_for(range,  [&]( int i,  int j, int k){
      AreaSumF(i,j,k) += weight(i,j,k)*length(i,j,k)*length(i,j,k); // [#/m]
      num_den(i,j,k)  += weight(i,j,k);
    }); //end cell loop
  }

  for ( int ienv = 0; ienv < m_Nenv; ienv++ ){

    const std::string weight_name    = ArchesCore::append_env( "w", ienv);
    const std::string length_name    = ArchesCore::append_env( m_length_root, ienv);
    const std::string surfAreaF_name = ArchesCore::append_env( m_surfAreaF_root, ienv);

    constCCVariable<double>& weight =
      tsk_info->get_const_uintah_field_add<constCCVariable<double> >( weight_name );
    constCCVariable<double>& length =
      tsk_info->get_const_uintah_field_add< constCCVariable<double> >(length_name);

    CCVariable<double>& surfaceAreaFraction
      = tsk_info->get_uintah_field_add<CCVariable<double> >( surfAreaF_name );

    Uintah::parallel_for( range, [&](int i, int j, int k){
     surfaceAreaFraction(i,j,k) =  weight(i,j,k)*length(i,j,k)*length(i,j,k)/AreaSumF(i,j,k);
    });
  }
}

//--------------------------------------------------------------------------------------------------
void
PartVariablesDQMOM::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( m_number_density_name, ArchesFieldContainer::COMPUTES, variable_registry );

  for ( int ienv = 0; ienv < m_Nenv; ienv++ ){
    const std::string weight_name = ArchesCore::append_env( "w", ienv);
    const std::string length_name = ArchesCore::append_env( m_length_root, ienv);
    const std::string surfAreaF_name = ArchesCore::append_env( m_surfAreaF_root, ienv);

    register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::LATEST, variable_registry, time_substep);

    register_variable( length_name, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::LATEST, variable_registry, time_substep);

     register_variable( surfAreaF_name, ArchesFieldContainer::COMPUTES, variable_registry, time_substep);

  }

}

//--------------------------------------------------------------------------------------------------
void
PartVariablesDQMOM::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& num_den  = tsk_info->get_uintah_field_add<CCVariable<double> >( m_number_density_name );
  CCVariable<double>& AreaSumF = tsk_info->get_uintah_field_add< CCVariable<double> >("AreaSum",0);// temporal variable

  AreaSumF.initialize(0.0);
  num_den.initialize(0.0);

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

  for (int ienv = 0; ienv < m_Nenv; ienv++) {
    const std::string weight_name = ArchesCore::append_env( "w", ienv);
    const std::string length_name = ArchesCore::append_env( m_length_root, ienv);

    constCCVariable<double>& weight =
      tsk_info->get_const_uintah_field_add<constCCVariable<double> >( weight_name );

    constCCVariable<double>& length =
      tsk_info->get_const_uintah_field_add< constCCVariable<double> >(length_name);

    Uintah::parallel_for(range,  [&]( int i,  int j, int k){
      AreaSumF(i,j,k) +=  weight(i,j,k)*length(i,j,k)*length(i,j,k); // [#/m]
      num_den(i,j,k)  += weight(i,j,k);
    }); //end cell loop
  }

  for ( int ienv = 0; ienv < m_Nenv; ienv++ ){

    const std::string weight_name    = ArchesCore::append_env( "w", ienv);
    const std::string length_name    = ArchesCore::append_env( m_length_root, ienv);
    const std::string surfAreaF_name = ArchesCore::append_env( m_surfAreaF_root, ienv);

    constCCVariable<double>& weight =
      tsk_info->get_const_uintah_field_add<constCCVariable<double> >( weight_name );
    constCCVariable<double>& length =
      tsk_info->get_const_uintah_field_add< constCCVariable<double> >(length_name);

    CCVariable<double>& surfaceAreaFraction
      = tsk_info->get_uintah_field_add<CCVariable<double> >( surfAreaF_name );

    Uintah::parallel_for( range, [&](int i, int j, int k){
     surfaceAreaFraction(i,j,k) =  weight(i,j,k)*length(i,j,k)*length(i,j,k)/AreaSumF(i,j,k);
    });
  }
}
} //namespace Uintah
