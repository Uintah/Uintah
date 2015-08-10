#include <CCA/Components/Arches/Utility/RandParticleLoc.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *;

RandParticleLoc::RandParticleLoc( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

RandParticleLoc::~RandParticleLoc(){
}

void
RandParticleLoc::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_ppos = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("LagrangianParticles")->findBlock("ParticlePosition");
  db_ppos->getAttribute("x",_px_name);
  db_ppos->getAttribute("y",_py_name);
  db_ppos->getAttribute("z",_pz_name);

}

void
RandParticleLoc::create_local_labels(){
}


//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
RandParticleLoc::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable( _px_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( _py_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );
  register_variable( _pz_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW,  variable_registry );

}

void
RandParticleLoc::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;


  typedef SpatialOps::SpatFldPtr<ParticleField> Pptr;

  Pptr px = tsk_info->get_particle_field(_px_name);
  Pptr py = tsk_info->get_particle_field(_py_name);
  Pptr pz = tsk_info->get_particle_field(_pz_name);

  //this is a poor man's random particle initialization...

  ParticleField& varx = *px;

  ParticleField::iterator iterx = varx.begin();
  const ParticleField::iterator iterx_end = varx.end();
  for( ; iterx != iterx_end; ++iterx ){
    *iterx = ((double)std::rand()/RAND_MAX);
  }

  ParticleField& vary = *py;

  ParticleField::iterator itery = vary.begin();
  const ParticleField::iterator itery_end = vary.end();
  for( ; itery != itery_end; ++itery ){
    *itery = ((double)std::rand()/RAND_MAX);
  }

  ParticleField& varz = *pz;

  ParticleField::iterator iterz = varz.begin();
  const ParticleField::iterator iterz_end = varz.end();
  for( ; iterz != iterz_end; ++iterz ){
    *iterz = ((double)std::rand()/RAND_MAX);
  }
}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
RandParticleLoc::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
RandParticleLoc::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task.
void
RandParticleLoc::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

  //FUNCITON CALL     STRING NAME(VL)     DEPENDENCY    GHOST DW     VR
  //register_variable( "a_sample_variable", ArchesFieldContainer::COMPUTES,       0, ArchesFieldContainer::NEWDW,  variable_registry, time_substep );

}

//This is the work for the task.  First, get the variables. Second, do the work!
void
RandParticleLoc::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                  SpatialOps::OperatorDatabase& opr ){

  using namespace SpatialOps;
  using SpatialOps::operator *;

  //SurfZP const w      = tsk_info->get_so_field<SurfZ>("wVelocitySPBC" );


}
