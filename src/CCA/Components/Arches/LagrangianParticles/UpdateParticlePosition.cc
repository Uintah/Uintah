#include <CCA/Components/Arches/LagrangianParticles/UpdateParticlePosition.h>

namespace Uintah{

//--------------------------------------------------------------------------------------------------
UpdateParticlePosition::UpdateParticlePosition( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
UpdateParticlePosition::~UpdateParticlePosition(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace UpdateParticlePosition::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace UpdateParticlePosition::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &UpdateParticlePosition::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &UpdateParticlePosition::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &UpdateParticlePosition::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace UpdateParticlePosition::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &UpdateParticlePosition::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &UpdateParticlePosition::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &UpdateParticlePosition::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace UpdateParticlePosition::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace UpdateParticlePosition::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
UpdateParticlePosition::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_ppos = db->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_name);
  db_ppos->getAttribute("y",_py_name);
  db_ppos->getAttribute("z",_pz_name);

  ProblemSpecP db_vel = db->findBlock("ParticleVelocity");
  db_vel->getAttribute("u",_u_name);
  db_vel->getAttribute("v",_v_name);
  db_vel->getAttribute("w",_w_name);

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticlePosition::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks)
{
  register_variable( _px_name, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  register_variable( _py_name, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  register_variable( _pz_name, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void UpdateParticlePosition::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  ParticleTuple px_tup = tsk_info->get_uintah_particle_field( _px_name );
  ParticleTuple py_tup = tsk_info->get_uintah_particle_field( _py_name );
  ParticleTuple pz_tup = tsk_info->get_uintah_particle_field( _pz_name );

  ParticleVariable<double>& px = *(std::get<0>(px_tup));
  ParticleVariable<double>& py = *(std::get<0>(py_tup));
  ParticleVariable<double>& pz = *(std::get<0>(pz_tup));

  ParticleSubset* p_subset = std::get<1>(px_tup);
  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;
    px[i] = 0.0;
    py[i] = 0.0;
    pz[i] = 0.0;
  }

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticlePosition::register_timestep_init(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void UpdateParticlePosition::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

//--------------------------------------------------------------------------------------------------
void
UpdateParticlePosition::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep , const bool packed_tasks){

  register_variable( _px_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry, m_task_name );
  register_variable( _py_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry, m_task_name );
  register_variable( _pz_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry, m_task_name );

  register_variable( _px_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, m_task_name );
  register_variable( _py_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, m_task_name );
  register_variable( _pz_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, m_task_name );

  register_variable( _u_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, m_task_name );
  register_variable( _v_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, m_task_name );
  register_variable( _w_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void UpdateParticlePosition::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  ParticleTuple px_tup = tsk_info->get_uintah_particle_field( _px_name );
  ParticleTuple py_tup = tsk_info->get_uintah_particle_field( _py_name );
  ParticleTuple pz_tup = tsk_info->get_uintah_particle_field( _pz_name );

  ParticleVariable<double>& px = *(std::get<0>(px_tup));
  ParticleVariable<double>& py = *(std::get<0>(py_tup));
  ParticleVariable<double>& pz = *(std::get<0>(pz_tup));
  ParticleSubset* p_subset = std::get<1>(px_tup);

  ConstParticleTuple pu_tup = tsk_info->get_const_uintah_particle_field(_u_name);
  ConstParticleTuple pv_tup = tsk_info->get_const_uintah_particle_field(_v_name);
  ConstParticleTuple pw_tup = tsk_info->get_const_uintah_particle_field(_w_name);

  constParticleVariable<double>& pu = *(std::get<0>(pu_tup));
  constParticleVariable<double>& pv = *(std::get<0>(pv_tup));
  constParticleVariable<double>& pw = *(std::get<0>(pw_tup));

  ConstParticleTuple old_px_tup = tsk_info->get_const_uintah_particle_field(_px_name);
  ConstParticleTuple old_py_tup = tsk_info->get_const_uintah_particle_field(_py_name);
  ConstParticleTuple old_pz_tup = tsk_info->get_const_uintah_particle_field(_pz_name);

  constParticleVariable<double>& old_px = *(std::get<0>(old_px_tup));
  constParticleVariable<double>& old_py = *(std::get<0>(old_py_tup));
  constParticleVariable<double>& old_pz = *(std::get<0>(old_pz_tup));

  const double dt = tsk_info->get_dt();

  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;

    px[i] = old_px[i] + dt * pu[i];
    py[i] = old_py[i] + dt * pv[i];
    pz[i] = old_pz[i] + dt * pw[i];

  }
}
} //namespace Uintah
