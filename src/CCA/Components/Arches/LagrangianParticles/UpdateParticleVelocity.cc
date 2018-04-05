#include <CCA/Components/Arches/LagrangianParticles/UpdateParticleVelocity.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>

//--------------------------------------------------------------------------------------------------
namespace Uintah{
UpdateParticleVelocity::UpdateParticleVelocity( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
UpdateParticleVelocity::~UpdateParticleVelocity(){
}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleVelocity::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_ppos = db->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_name);
  db_ppos->getAttribute("y",_py_name);
  db_ppos->getAttribute("z",_pz_name);

  ProblemSpecP db_vel = db->findBlock("ParticleVelocity");
  db_vel->getAttribute("u",_u_name);
  db_vel->getAttribute("v",_v_name);
  db_vel->getAttribute("w",_w_name);

  Uintah::ArchesParticlesHelper::mark_for_relocation(_u_name);
  Uintah::ArchesParticlesHelper::mark_for_relocation(_v_name);
  Uintah::ArchesParticlesHelper::mark_for_relocation(_w_name);
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_u_name);
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_v_name);
  Uintah::ArchesParticlesHelper::needs_boundary_condition(_w_name);

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleVelocity::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const bool packed_tasks)
{
  register_variable( _u_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _v_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _w_name, ArchesFieldContainer::COMPUTES, variable_registry );
}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  ParticleTuple pu_t = tsk_info->get_uintah_particle_field( _u_name );
  ParticleTuple pv_t = tsk_info->get_uintah_particle_field( _v_name );
  ParticleTuple pw_t = tsk_info->get_uintah_particle_field( _w_name );

  ParticleVariable<double>& pu = *(std::get<0>(pu_t));
  ParticleVariable<double>& pv = *(std::get<0>(pv_t));
  ParticleVariable<double>& pw = *(std::get<0>(pw_t));
  ParticleSubset* p_subset = std::get<1>(pu_t);

  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;
    pu[i] = 0.0;
    pv[i] = 0.0;
    pw[i] = 0.0;
  }

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleVelocity::register_timestep_eval( 
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, 
    const int time_substep , const bool packed_tasks){

  register_variable( _u_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( _v_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( _w_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );

  register_variable( _u_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );
  register_variable( _v_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );
  register_variable( _w_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
UpdateParticleVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  ParticleTuple pu_t = tsk_info->get_uintah_particle_field( _u_name );
  ParticleTuple pv_t = tsk_info->get_uintah_particle_field( _v_name );
  ParticleTuple pw_t = tsk_info->get_uintah_particle_field( _w_name );

  ParticleVariable<double>& pu = *(std::get<0>(pu_t));
  ParticleVariable<double>& pv = *(std::get<0>(pv_t));
  ParticleVariable<double>& pw = *(std::get<0>(pw_t));
  ParticleSubset* p_subset = std::get<1>(pu_t);

  ConstParticleTuple old_pu_t = tsk_info->get_const_uintah_particle_field( _u_name );
  ConstParticleTuple old_pv_t = tsk_info->get_const_uintah_particle_field( _v_name );
  ConstParticleTuple old_pw_t = tsk_info->get_const_uintah_particle_field( _w_name );

  constParticleVariable<double>& old_pu = *(std::get<0>(old_pu_t));
  constParticleVariable<double>& old_pv = *(std::get<0>(old_pv_t));
  constParticleVariable<double>& old_pw = *(std::get<0>(old_pw_t));

  //const double dt = tsk_info->get_dt();

  //no RHS currently
  for (auto iter = p_subset->begin(); iter != p_subset->end(); iter++){
    particleIndex i = *iter;

    pu[i] = old_pu[i];
    pv[i] = old_pv[i];
    pw[i] = old_pw[i];

  }
}
} //namespace Uintah
