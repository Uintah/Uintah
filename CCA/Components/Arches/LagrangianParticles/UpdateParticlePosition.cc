#include <CCA/Components/Arches/LagrangianParticles/UpdateParticlePosition.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *;

UpdateParticlePosition::UpdateParticlePosition( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

UpdateParticlePosition::~UpdateParticlePosition(){
}

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

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
UpdateParticlePosition::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

}

void
UpdateParticlePosition::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
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
UpdateParticlePosition::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
UpdateParticlePosition::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){ 
}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
UpdateParticlePosition::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  register_variable( _px_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( _py_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( _pz_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );

  register_variable( _px_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );
  register_variable( _py_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );
  register_variable( _pz_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );

  register_variable( _u_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );
  register_variable( _v_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );
  register_variable( _w_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry );

}

//This is the work for the task.  First, get the variables. Second, do the work!
void
UpdateParticlePosition::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;
  typedef SpatialOps::SpatFldPtr<Particle::ParticleField> Pptr;

  Pptr px = tsk_info->get_particle_field( _px_name );
  Pptr py = tsk_info->get_particle_field( _py_name );
  Pptr pz = tsk_info->get_particle_field( _pz_name );

  const Pptr pu = tsk_info->get_const_particle_field(_u_name);
  const Pptr pv = tsk_info->get_const_particle_field(_v_name);
  const Pptr pw = tsk_info->get_const_particle_field(_w_name);

  const Pptr old_px = tsk_info->get_const_particle_field(_px_name);
  const Pptr old_py = tsk_info->get_const_particle_field(_py_name);
  const Pptr old_pz = tsk_info->get_const_particle_field(_pz_name);

  const double dt = tsk_info->get_dt();

  *px <<= *old_px + dt * *pu;
  *py <<= *old_py + dt * *pv;
  *pz <<= *old_pz + dt * *pw;

}
