#include <CCA/Components/Arches/Utility/RandParticleLoc.h>

namespace Uintah {

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RandParticleLoc::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RandParticleLoc::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &RandParticleLoc::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &RandParticleLoc::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &RandParticleLoc::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RandParticleLoc::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RandParticleLoc::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace RandParticleLoc::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
RandParticleLoc::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_ppos = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("LagrangianParticles")->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_name);
  db_ppos->getAttribute("y",_py_name);
  db_ppos->getAttribute("z",_pz_name);

}

//--------------------------------------------------------------------------------------------------
void
RandParticleLoc::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( _px_name, ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _py_name, ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _pz_name, ArchesFieldContainer::MODIFIES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void RandParticleLoc::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  typedef std::tuple<ParticleVariable<double>*, ParticleSubset*> PVarTuple;

  PVarTuple px = tsk_info->get_uintah_particle_field(_px_name);
  PVarTuple py = tsk_info->get_uintah_particle_field(_py_name);
  PVarTuple pz = tsk_info->get_uintah_particle_field(_pz_name);

  //poor man's random particle initialization
  ParticleVariable<double>& varx = *(std::get<0>(px));
  ParticleVariable<double>& vary = *(std::get<0>(py));
  ParticleVariable<double>& varz = *(std::get<0>(pz));

  for ( auto iter = (std::get<1>(px))->begin(); iter != (std::get<1>(px))->end(); iter++ ){

    particleIndex i = *iter;
    varx[i] = ((double)std::rand()/RAND_MAX);

  }

  for ( auto iter = (std::get<1>(py))->begin(); iter != (std::get<1>(py))->end(); iter++ ){

    particleIndex i = *iter;
    vary[i] = ((double)std::rand()/RAND_MAX);

  }

  for ( auto iter = (std::get<1>(pz))->begin(); iter != (std::get<1>(pz))->end(); iter++ ){

    particleIndex i = *iter;
    varz[i] = ((double)std::rand()/RAND_MAX);

  }
}
} //namespace Uintah
