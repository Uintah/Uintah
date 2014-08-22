#include <CCA/Components/Arches/Utility/InitLagrangianParticleVelocity.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *; 

InitLagrangianParticleVelocity::InitLagrangianParticleVelocity( std::string task_name, int matl_index ) : 
TaskInterface( task_name, matl_index ) { 
}

InitLagrangianParticleVelocity::~InitLagrangianParticleVelocity(){ 
}

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

  register_new_variable( _pu_label, PARTICLE ); 
  register_new_variable( _pv_label, PARTICLE ); 
  register_new_variable( _pw_label, PARTICLE ); 

}


//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
InitLagrangianParticleVelocity::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  register_variable( _pu_label  , PARTICLE , COMPUTES , 0 , NEWDW , variable_registry );
  register_variable( _pv_label  , PARTICLE , COMPUTES , 0 , NEWDW , variable_registry );
  register_variable( _pw_label  , PARTICLE , COMPUTES , 0 , NEWDW , variable_registry );

  register_variable( _px_label , PARTICLE , REQUIRES, 0 , NEWDW , variable_registry );
  register_variable( _py_label , PARTICLE , REQUIRES, 0 , NEWDW , variable_registry );
  register_variable( _pz_label , PARTICLE , REQUIRES, 0 , NEWDW , variable_registry );

  register_variable( _size_label , PARTICLE , REQUIRES, 0 , NEWDW , variable_registry );

  //gas velocity
  register_variable( "uVelocitySPBC", FACEX, REQUIRES, 1, NEWDW, variable_registry ); 
  register_variable( "vVelocitySPBC", FACEY, REQUIRES, 1, NEWDW, variable_registry ); 
  register_variable( "wVelocitySPBC", FACEZ, REQUIRES, 1, NEWDW, variable_registry ); 

}

void 
InitLagrangianParticleVelocity::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){ 


  using namespace SpatialOps;
  using SpatialOps::operator *; 

  typedef SpatialOps::SpatFldPtr<Particle::ParticleField> Pptr; 
  typedef SpatialOps::SpatFldPtr<SpatialOps::XVolField> XVolPtr; 
  typedef SpatialOps::SpatFldPtr<SpatialOps::YVolField> YVolPtr; 
  typedef SpatialOps::SpatFldPtr<SpatialOps::ZVolField> ZVolPtr; 

  Pptr up = tsk_info->get_particle_field(_pu_label); 
  Pptr vp = tsk_info->get_particle_field(_pv_label); 
  Pptr wp = tsk_info->get_particle_field(_pw_label); 

  const Pptr px = tsk_info->get_const_particle_field(_px_label); 
  const Pptr py = tsk_info->get_const_particle_field(_py_label); 
  const Pptr pz = tsk_info->get_const_particle_field(_pz_label); 

  const Pptr psize = tsk_info->get_const_particle_field(_size_label); 

  XVolPtr ug = tsk_info->get_const_so_field<SpatialOps::XVolField>("uVelocitySPBC"); 
  YVolPtr vg = tsk_info->get_const_so_field<SpatialOps::YVolField>("vVelocitySPBC"); 
  ZVolPtr wg = tsk_info->get_const_so_field<SpatialOps::ZVolField>("wVelocitySPBC"); 

  typedef SpatialOps::Particle::CellToParticle<SpatialOps::XVolField> GXtoPT;
  typedef SpatialOps::Particle::CellToParticle<SpatialOps::YVolField> GYtoPT;
  typedef SpatialOps::Particle::CellToParticle<SpatialOps::ZVolField> GZtoPT;

  GXtoPT* gx_to_pt = opr.retrieve_operator<GXtoPT>(); 
  GYtoPT* gy_to_pt = opr.retrieve_operator<GYtoPT>(); 
  GZtoPT* gz_to_pt = opr.retrieve_operator<GZtoPT>(); 

  //gx_to_pt->set_coordinate_information( &*px, &*py, &*pz, &*size ); 
  gx_to_pt->set_coordinate_information( px.operator->(), py.operator->(), pz.operator->(), psize.operator->() ); 
  gx_to_pt->apply_to_field(*ug, *up); 

  gy_to_pt->set_coordinate_information( px.operator->(), py.operator->(), pz.operator->(), psize.operator->() ); 
  gy_to_pt->apply_to_field(*vg, *vp); 

  gz_to_pt->set_coordinate_information( px.operator->(), py.operator->(), pz.operator->(), psize.operator->() ); 
  gz_to_pt->apply_to_field(*wg, *wp); 

}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void 
InitLagrangianParticleVelocity::register_timestep_init( std::vector<VariableInformation>& variable_registry ){ 
}

void 
InitLagrangianParticleVelocity::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                          SpatialOps::OperatorDatabase& opr ){ 

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task. 
void 
InitLagrangianParticleVelocity::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 
}

//This is the work for the task.  First, get the variables. Second, do the work! 
void 
InitLagrangianParticleVelocity::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *; 

}
