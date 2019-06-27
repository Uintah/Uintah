#include <CCA/Components/Arches/Utility/InitLagrangianParticleSize.h>

namespace Uintah{

InitLagrangianParticleSize::InitLagrangianParticleSize( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

InitLagrangianParticleSize::~InitLagrangianParticleSize(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleSize::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleSize::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &InitLagrangianParticleSize::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &InitLagrangianParticleSize::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &InitLagrangianParticleSize::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleSize::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleSize::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleSize::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

void
InitLagrangianParticleSize::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_lp = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("LagrangianParticles");

  ProblemSpecP db_ppos = db_lp->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_label);
  db_ppos->getAttribute("y",_py_label);
  db_ppos->getAttribute("z",_pz_label);

  db_lp->findBlock("ParticleVelocity")->getAttribute("u", _pu_label );
  db_lp->findBlock("ParticleVelocity")->getAttribute("v", _pv_label );
  db_lp->findBlock("ParticleVelocity")->getAttribute("w", _pw_label );

  ProblemSpecP db_size = db_lp->findBlock("ParticleSize");
  db_size->getAttribute("label",_size_label);

  _init_type = "NA";
  db->findBlock("size_init")->getAttribute("type", _init_type );

  if ( _init_type == "fixed"){
    db->require("fixed_diameter",_fixed_d);
  } else if ( _init_type == "random"){
    db->require("max_diameter", _max_d);
  } else {
    throw InvalidValue("Error: Unrecognized lagrangian particle velocity initializiation.",__FILE__,__LINE__);
  }

}

void
InitLagrangianParticleSize::create_local_labels(){

  register_new_variable< ParticleVariable<double> >( _size_label );

}


//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
InitLagrangianParticleSize::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( _size_label, ArchesFieldContainer::MODIFIES, variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void InitLagrangianParticleSize::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  ParticleTuple dp_t = tsk_info->get_uintah_particle_field(_size_label);
  ParticleVariable<double>& dp = *(std::get<0>(dp_t));
  ParticleSubset* p_subset = std::get<1>(dp_t);

  if ( _init_type == "fixed"){
    for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
      particleIndex i = *iter;
      dp[i] = _fixed_d;
    }
  } else if ( _init_type == "random"){
    for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
      particleIndex i = *iter;
      dp[i] = ((double)std::rand()/RAND_MAX)*_max_d;
    }
  }

}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
InitLagrangianParticleSize::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
}

template <typename ExecSpace, typename MemSpace> void
InitLagrangianParticleSize::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj){}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task.
void
InitLagrangianParticleSize::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){
}

//This is the work for the task.  First, get the variables. Second, do the work!
template <typename ExecSpace, typename MemSpace>
void InitLagrangianParticleSize::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

} //namespace Uintah
