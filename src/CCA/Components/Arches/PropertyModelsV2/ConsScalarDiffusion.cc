#include <CCA/Components/Arches/PropertyModelsV2/ConsScalarDiffusion.h>
#include <CCA/Components/Arches/UPSHelper.h>

using namespace Uintah;
using namespace ArchesCore;

//--------------------------------------------------------------------------------------------------
ConsScalarDiffusion::ConsScalarDiffusion( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ){}

//--------------------------------------------------------------------------------------------------
ConsScalarDiffusion::~ConsScalarDiffusion(){}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::problemSetup( ProblemSpecP& db ){

  m_density_name        = parse_ups_for_role( DENSITY, db, "density" );
  m_turb_viscosity_name = "turb_viscosity";
  m_gamma_name          = m_task_name;
  db->require("D_mol", m_Diffusivity);
  db->getWithDefault("turbulentPrandtlNumber", m_Pr, 0.4);
}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_gamma_name);

}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_gamma_name,          AFC::COMPUTES, variable_registry, m_task_name );
  //register_variable( m_turb_viscosity_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry ,m_task_name );
  //register_variable( m_density_name,        AFC::REQUIRES, 0, AFC::NEWDW, variable_registry ,m_task_name );

}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){


  CCVariable<double>& gamma = tsk_info->get_uintah_field_add<CCVariable<double> >(m_gamma_name);
  //constCCVariable<double>& mu_t    = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_turb_viscosity_name);
  //constCCVariable<double>& density = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_density_name);

  gamma.initialize(0.0);

  //Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  //Uintah::parallel_for( range, [&](int i, int j, int k){
   //gamma(i,j,k) = density(i,j,k)*m_Diffusivity + mu_t(i,j,k)/m_Pr;
  //});
}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::register_timestep_init( AVarInfo& variable_registry , const bool pack_tasks){


}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){


}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_gamma_name,          AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_turb_viscosity_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep ,m_task_name );
  register_variable( m_density_name,        AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep ,m_task_name );

}

//--------------------------------------------------------------------------------------------------
void ConsScalarDiffusion::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& gamma = tsk_info->get_uintah_field_add<CCVariable<double> >(m_gamma_name);
  constCCVariable<double>& mu_t    = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_turb_viscosity_name);
  constCCVariable<double>& density = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_density_name);

  gamma.initialize(0.0);

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
   gamma(i,j,k) = density(i,j,k)*m_Diffusivity + mu_t(i,j,k)/m_Pr;
  });

}

