#include <CCA/Components/Arches/ParticleModels/Burnout.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Burnout::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Burnout::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &Burnout::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Burnout::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Burnout::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Burnout::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &Burnout::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Burnout::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Burnout::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Burnout::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &Burnout::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Burnout::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Burnout::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace Burnout::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
Burnout::problemSetup( ProblemSpecP& db ){

  const ProblemSpecP db_root = db->getRootNode();

  // get weight information
  std::string m_weight_name = "w";
  for ( int qn = 0; qn < _Nenv; qn++ ){
    m_weight_names.push_back(ArchesCore::append_qn_env( m_weight_name, qn));
    m_weight_scaling_constants.push_back(ArchesCore::get_scaling_constant(db, m_weight_name, qn));
  }
  // get rc information
  std::string m_rc_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
  for ( int qn = 0; qn < _Nenv; qn++ ){
    m_rc_names.push_back(ArchesCore::append_qn_env( m_rc_name, qn));
    m_rc_scaling_constants.push_back(ArchesCore::get_scaling_constant(db, m_rc_name, qn));
  }
  // get char information
  std::string m_char_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
  for ( int qn = 0; qn < _Nenv; qn++ ){
    m_char_names.push_back(ArchesCore::append_qn_env( m_char_name, qn));
    m_char_scaling_constants.push_back(ArchesCore::get_scaling_constant(db, m_char_name, qn));
  }

  ArchesCore::GridVarMap<CCVariable<double> > varMap;
  m_vol_fraction_name = varMap.vol_frac_name;

}

//--------------------------------------------------------------------------------------------------
void
Burnout::create_local_labels(){

  register_new_variable<CCVariable<double> >( m_task_name );
  register_new_variable<CCVariable<double> >( m_numerator_name );
  register_new_variable<CCVariable<double> >( m_denominator_name );

}

//--------------------------------------------------------------------------------------------------
void
Burnout::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void Burnout::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& burnout = tsk_info->get_field<CCVariable<double> >( m_task_name );
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    burnout(i,j,k) = 0.0;
  });

}

//--------------------------------------------------------------------------------------------------
void
Burnout::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
Burnout::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  CCVariable<double>& burnout = tsk_info->get_field<CCVariable<double> >( m_task_name );
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    burnout(i,j,k) = 0.0;
  });

}

//--------------------------------------------------------------------------------------------------
void
Burnout::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  for ( int qn = 0; qn < _Nenv; qn++ ){
    register_variable( m_weight_names[qn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( m_rc_names[qn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( m_char_names[qn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }
  register_variable( m_task_name, ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( m_vol_fraction_name, ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( m_denominator_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_numerator_name, ArchesFieldContainer::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void Burnout::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
  CCVariable<double>& numerator_sum = tsk_info->get_field< CCVariable<double> >(m_numerator_name);
  CCVariable<double>& denominator_sum = tsk_info->get_field< CCVariable<double> >(m_denominator_name);
  numerator_sum.initialize(0.0);
  denominator_sum.initialize(0.0);
  CCVariable<double>& burnout = tsk_info->get_field<CCVariable<double> >( m_task_name );
  constCCVariable<double>& vol_frac = tsk_info->get_field<constCCVariable<double> >(m_vol_fraction_name);

  for ( int qn = 0; qn < _Nenv; qn++ ){
    const double rc_scaling_constant = m_rc_scaling_constants[qn];
    const double weight_scaling_constant = m_weight_scaling_constants[qn];
    const double char_scaling_constant = m_char_scaling_constants[qn];
    constCCVariable<double>& weight = tsk_info->get_field<constCCVariable<double> >(m_weight_names[qn]);
    constCCVariable<double>& rcmass = tsk_info->get_field<constCCVariable<double> >(m_rc_names[qn]);
    constCCVariable<double>& charmass = tsk_info->get_field<constCCVariable<double> >(m_char_names[qn]);
    Uintah::parallel_for( range, [&](int i, int j, int k){
      numerator_sum(i,j,k) += vol_frac(i,j,k)*weight_scaling_constant*(rc_scaling_constant*rcmass(i,j,k) +
                                                                   char_scaling_constant*charmass(i,j,k));
      // Current organic mass in the cell for environment qn. [kg/m^3]

      denominator_sum(i,j,k) += vol_frac(i,j,k)*weight_scaling_constant*rc_scaling_constant*weight(i,j,k);
      // Had no reactions occured this would be the current organic mass in the cell for environment qn. [kg/m^3]
      // NOTE: this model assumes all of the initial mass is in the form raw coal.

    }); // parallel for
  } // environment loop
  Uintah::parallel_for( range, [&](int i, int j, int k){

    burnout(i,j,k) = vol_frac(i,j,k)*(numerator_sum(i,j,k) / (denominator_sum(i,j,k) + 1e-100));

  });
}
} //namespace Uintah
