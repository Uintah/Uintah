#include <CCA/Components/Arches/LagrangianParticles/UpdateParticleSize.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *; 

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

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
UpdateParticleSize::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

}

void 
UpdateParticleSize::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
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
UpdateParticleSize::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 
}

void 
UpdateParticleSize::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 
}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
UpdateParticleSize::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  register_variable( _size_name, PARTICLE, COMPUTES, 0, NEWDW,  variable_registry );

  register_variable( _size_name, PARTICLE, REQUIRES, 0, OLDDW,  variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
UpdateParticleSize::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SpatFldPtr<Particle::ParticleField> Pptr; 

  Pptr pd = tsk_info->get_particle_field( _size_name ); 

  Pptr old_pd = tsk_info->get_const_particle_field( _size_name ); 

  *pd <<= *old_pd; 

}
