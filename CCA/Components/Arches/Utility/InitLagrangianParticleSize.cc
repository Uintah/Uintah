#include <CCA/Components/Arches/Utility/InitLagrangianParticleSize.h>
#include <CCA/Components/Arches/Operators/Operators.h>

#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


using namespace Uintah;
using namespace SpatialOps;
using SpatialOps::operator *;

InitLagrangianParticleSize::InitLagrangianParticleSize( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

InitLagrangianParticleSize::~InitLagrangianParticleSize(){
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

  register_new_variable_new< ParticleVariable<double> >( _size_label );

}


//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
InitLagrangianParticleSize::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  register_variable_new( _size_label, ArchesFieldContainer::COMPUTES , 0 , ArchesFieldContainer::NEWDW , variable_registry );

}

void
InitLagrangianParticleSize::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){


  using namespace SpatialOps;
  using SpatialOps::operator *;

  typedef SpatialOps::SpatFldPtr<Particle::ParticleField> Pptr;

  Pptr dp = tsk_info->get_particle_field(_size_label);

  if ( _init_type == "fixed"){
    *dp <<= _fixed_d;
  } else if ( _init_type == "random"){
    ParticleField& dd = *dp;
    for ( ParticleField::iterator iter = dd.begin(); iter != dd.end(); iter++ ){
      *iter = ((double)std::rand()/RAND_MAX)*_max_d;
    }
  }

}

//
//------------------------------------------------
//------------- TIMESTEP INIT --------------------
//------------------------------------------------
//
void
InitLagrangianParticleSize::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
}

void
InitLagrangianParticleSize::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

}


//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

//Register all variables both local and those needed from elsewhere that are required for this task.
void
InitLagrangianParticleSize::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
}

//This is the work for the task.  First, get the variables. Second, do the work!
void
InitLagrangianParticleSize::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                  SpatialOps::OperatorDatabase& opr ){

}
