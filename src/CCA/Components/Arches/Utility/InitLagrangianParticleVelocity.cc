#include <CCA/Components/Arches/Utility/InitLagrangianParticleVelocity.h>

namespace Uintah{
InitLagrangianParticleVelocity::InitLagrangianParticleVelocity( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

InitLagrangianParticleVelocity::~InitLagrangianParticleVelocity(){
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleVelocity::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleVelocity::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &InitLagrangianParticleVelocity::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &InitLagrangianParticleVelocity::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &InitLagrangianParticleVelocity::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleVelocity::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleVelocity::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace InitLagrangianParticleVelocity::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
InitLagrangianParticleVelocity::problemSetup( ProblemSpecP& db ){

  _init_type = "NA";
  db->findBlock("velocity_init")->getAttribute("type", _init_type );

  if ( _init_type == "as_gas_velocity"){
    //nothing to do here.
  } else {
    throw InvalidValue("Error: Unrecognized lagrangian particle velocity initializiation.",__FILE__,__LINE__);
  }

  ProblemSpecP db_lp = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("LagrangianParticles");

  //also need coordinate information for interpolation
  ProblemSpecP db_ppos = db_lp->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_label);
  db_ppos->getAttribute("y",_py_label);
  db_ppos->getAttribute("z",_pz_label);

  db_lp->findBlock("ParticleVelocity")->getAttribute("u", _pu_label );
  db_lp->findBlock("ParticleVelocity")->getAttribute("v", _pv_label );
  db_lp->findBlock("ParticleVelocity")->getAttribute("w", _pw_label );

  ProblemSpecP db_size = db_lp->findBlock("ParticleSize");
  db_size->getAttribute("label",_size_label);

}

void
InitLagrangianParticleVelocity::create_local_labels(){

  register_new_variable<ParticleVariable<double> >( _pu_label );
  register_new_variable<ParticleVariable<double> >( _pv_label );
  register_new_variable<ParticleVariable<double> >( _pw_label );

}


//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
InitLagrangianParticleVelocity::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  register_variable( _pu_label  , ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _pv_label  , ArchesFieldContainer::MODIFIES, variable_registry );
  register_variable( _pw_label  , ArchesFieldContainer::MODIFIES, variable_registry );

  register_variable( _px_label , ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::NEWDW , variable_registry );
  register_variable( _py_label , ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::NEWDW , variable_registry );
  register_variable( _pz_label , ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::NEWDW , variable_registry );

  register_variable( _size_label , ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::NEWDW , variable_registry );

  //gas velocity
  register_variable( "uVelocitySPBC", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "vVelocitySPBC", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "wVelocitySPBC", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );

}

template <typename ExecSpace, typename MemSpace>
void InitLagrangianParticleVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  ParticleTuple pu_tup = tsk_info->get_uintah_particle_field(_pu_label);
  ParticleTuple pv_tup = tsk_info->get_uintah_particle_field(_pv_label);
  ParticleTuple pw_tup = tsk_info->get_uintah_particle_field(_pw_label);

  ParticleVariable<double>& pu = *(std::get<0>(pu_tup));
  ParticleVariable<double>& pv = *(std::get<0>(pv_tup));
  ParticleVariable<double>& pw = *(std::get<0>(pw_tup));
  ParticleSubset* p_subset = std::get<1>(pu_tup);

  ConstParticleTuple px_tup = tsk_info->get_const_uintah_particle_field(_px_label);
  ConstParticleTuple py_tup = tsk_info->get_const_uintah_particle_field(_py_label);
  ConstParticleTuple pz_tup = tsk_info->get_const_uintah_particle_field(_pz_label);

  constParticleVariable<double>& px = *(std::get<0>(px_tup));
  constParticleVariable<double>& py = *(std::get<0>(py_tup));
  constParticleVariable<double>& pz = *(std::get<0>(pz_tup));

  // ConstParticleTuple psize_tup = tsk_info->get_const_uintah_particle_field(_size_label);
  // constParticleVariable<double>& psize = *(std::get<0>(psize_tup));

  constSFCXVariable<double>& ug = tsk_info->get_field<constSFCXVariable<double> >("uVelocitySPBC");
  constSFCYVariable<double>& vg = tsk_info->get_field<constSFCYVariable<double> >("vVelocitySPBC");
  constSFCZVariable<double>& wg = tsk_info->get_field<constSFCZVariable<double> >("wVelocitySPBC");

  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;

    const double x = px[i];
    const double y = py[i];
    const double z = pz[i];

    Uintah::Point my_loc(x,y,z);

    //nearest neighbor
    IntVector idx = patch->getCellIndex(my_loc);

    const double u = ug[idx];
    const double v = vg[idx];
    const double w = wg[idx];

    pu[i] = u;
    pv[i] = v;
    pw[i] = w;

  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
InitLagrangianParticleVelocity::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
}

template <typename ExecSpace, typename MemSpace> void
InitLagrangianParticleVelocity::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task.
void
InitLagrangianParticleVelocity::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){
}

//This is the work for the task.  First, get the variables. Second, do the work!
template <typename ExecSpace, typename MemSpace>
void InitLagrangianParticleVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

} // namespace Uintah
