#include <CCA/Components/Arches/LagrangianParticles/UpdateParticleVelocity.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *; 

UpdateParticleVelocity::UpdateParticleVelocity( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

UpdateParticleVelocity::~UpdateParticleVelocity(){ 
}

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

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
UpdateParticleVelocity::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

}

void 
UpdateParticleVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 


}


//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
UpdateParticleVelocity::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 
}

void 
UpdateParticleVelocity::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 
}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
UpdateParticleVelocity::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  register_variable( _u_name, PARTICLE, COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( _v_name, PARTICLE, COMPUTES, 0, NEWDW,  variable_registry );
  register_variable( _w_name, PARTICLE, COMPUTES, 0, NEWDW,  variable_registry );

  register_variable( _u_name, PARTICLE, REQUIRES, 0, OLDDW,  variable_registry );
  register_variable( _v_name, PARTICLE, REQUIRES, 0, OLDDW,  variable_registry );
  register_variable( _w_name, PARTICLE, REQUIRES, 0, OLDDW,  variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
UpdateParticleVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SpatFldPtr<Particle::ParticleField> Pptr; 

  Pptr pu = tsk_info->get_particle_field( _u_name ); 
  Pptr pv = tsk_info->get_particle_field( _v_name ); 
  Pptr pw = tsk_info->get_particle_field( _w_name ); 

  Pptr old_pu = tsk_info->get_const_particle_field( _u_name ); 
  Pptr old_pv = tsk_info->get_const_particle_field( _v_name ); 
  Pptr old_pw = tsk_info->get_const_particle_field( _w_name ); 

  const double dt = tsk_info->get_dt(); 

  //no RHS currently
  *pu <<= *old_pu; 
  *pv <<= *old_pv; 
  *pw <<= *old_pw; 

}
