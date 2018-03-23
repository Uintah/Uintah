#include <CCA/Components/Arches/LagrangianParticles/UpdateParticleSize.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>

namespace Uintah{

UpdateParticleSize::UpdateParticleSize( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

UpdateParticleSize::~UpdateParticleSize(){
}

void
UpdateParticleSize::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_ppos = db->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_name);
  db_ppos->getAttribute("y",_py_name);
  db_ppos->getAttribute("z",_pz_name);

  ProblemSpecP db_vel = db->findBlock("ParticleVelocity");
  db_vel->getAttribute("u",_u_name);
  db_vel->getAttribute("v",_v_name);
  db_vel->getAttribute("w",_w_name);

  //parse and add the size variable here
  ProblemSpecP db_size = db->findBlock("ParticleSize");
  db_size->getAttribute("label",_size_name);
  Uintah::ArchesParticlesHelper::mark_for_relocation(_size_name);
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_size_name);

  //potentially remove later when Tony updates the particle helper class
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_px_name);
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_py_name);
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_pz_name);

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleSize::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks)
{
  register_variable( _size_name, ArchesFieldContainer::COMPUTES, variable_registry );
}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleSize::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  ParticleTuple pd_t = tsk_info->get_uintah_particle_field( _size_name );
  ParticleVariable<double>& pd = *(std::get<0>(pd_t));
  ParticleSubset* p_subset = std::get<1>(pd_t);

  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;
    pd[i] = 0.0;
  }

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleSize::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( _size_name, ArchesFieldContainer::COMPUTES, variable_registry );

  register_variable( _size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,
      variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work!
void
UpdateParticleSize::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  ParticleTuple pd_t = tsk_info->get_uintah_particle_field( _size_name );
  ParticleVariable<double>& pd = *(std::get<0>(pd_t));
  ParticleSubset* p_subset = std::get<1>(pd_t);

  ConstParticleTuple old_pd_t = tsk_info->get_const_uintah_particle_field( _size_name );
  constParticleVariable<double>& old_pd = *(std::get<0>(old_pd_t));

  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;
    pd[i] = old_pd[i];
  }

}
} //namespace Uintah
