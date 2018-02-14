#include <CCA/Components/Arches/ParticleModels/DQMOMNoInversion.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
DQMOMNoInversion::DQMOMNoInversion( std::string task_name, int matl_index, const int N ) :
                  TaskInterface( task_name, matl_index ), m_N(N){
}

//--------------------------------------------------------------------------------------------------
DQMOMNoInversion::~DQMOMNoInversion(){}

//--------------------------------------------------------------------------------------------------
void DQMOMNoInversion::problemSetup( ProblemSpecP& db ){

  //m_ic_names = ParticleTools::getICNames( db );

}

//--------------------------------------------------------------------------------------------------
void DQMOMNoInversion::create_local_labels(){

  for ( int i = 0; i < m_N; i++ ){

    std::stringstream sQN;
    sQN << i;

    for ( auto j = m_ic_names.begin(); j != m_ic_names.end(); j++ ){

      std::string source_name = *j + "_qn"+sQN.str()+"_src";
      m_ic_qn_srcnames.push_back(source_name);

      register_new_variable<CCVariable<double> >( source_name );

    }
  }

}

//--------------------------------------------------------------------------------------------------
void DQMOMNoInversion::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks){

  for  ( auto i = m_ic_qn_srcnames.begin(); i != m_ic_qn_srcnames.end(); i++ ){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );

  }
}

//--------------------------------------------------------------------------------------------------
void DQMOMNoInversion::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for  ( auto i = m_ic_qn_srcnames.begin(); i != m_ic_qn_srcnames.end(); i++ ){
    CCVariable<double>& var = tsk_info->get_uintah_field_add<CCVariable<double> >( *i );
    var.initialize(0.0);
  }

}

//--------------------------------------------------------------------------------------------------
void DQMOMNoInversion::register_timestep_init(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks){

  for  ( auto i = m_ic_qn_srcnames.begin(); i != m_ic_qn_srcnames.end(); i++ ){

    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,
                       variable_registry );

  }
}

//--------------------------------------------------------------------------------------------------
void DQMOMNoInversion::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for  ( auto i = m_ic_qn_srcnames.begin(); i != m_ic_qn_srcnames.end(); i++ ){

    CCVariable<double>& var = tsk_info->get_uintah_field_add<CCVariable<double> >(*i);
    constCCVariable<double>& old_var = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(*i);

    var.copy(old_var);

  }
}
