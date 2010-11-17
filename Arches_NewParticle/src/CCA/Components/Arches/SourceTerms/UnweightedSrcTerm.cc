#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/UnweightedSrcTerm.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

UnweightedSrcTerm::UnweightedSrcTerm( std::string srcName, 
                                      SimulationStateP& sharedState,
                                      vector<std::string> reqLabelNames, 
                                      ArchesLabel* fieldLabels ) 
: SourceTermBase(srcName, sharedState, reqLabelNames, fieldLabels )
{
  _label_sched_init = false; 
  _src_label = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 
}

UnweightedSrcTerm::~UnweightedSrcTerm()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
UnweightedSrcTerm::problemSetup(const ProblemSpecP& inputdb)
{

  //ProblemSpecP db = inputdb; 

  _source_type = CC_SRC; 
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
UnweightedSrcTerm::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "UnweightedSrcTerm::computeSource";
  Task* tsk = scinew Task(taskname, this, &UnweightedSrcTerm::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;
  }
  
  if( timeSubStep == 0 ) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }

  const VarLabel* d_areaFractionLabel = VarLabel::find( "areaFraction" );
  tsk->requires(Task::OldDW, d_areaFractionLabel, Ghost::AroundCells, 1);

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  CoalModelFactory& coalFactory = CoalModelFactory::self();

  for( vector<std::string>::iterator iter = _required_labels.begin();
       iter != _required_labels.end(); iter++) {
    DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>( &dqmomFactory.retrieve_scalar_eqn( *iter ) );
    tsk->requires( Task::OldDW, dqmom_eqn->getTransportEqnLabel(), Ghost::None, 0 );

    int quadNode = dqmom_eqn->getQuadNode();

    // require particle velocity
    if( coalFactory.useParticleVelocityModel() ) {
      d_particle_velocity_label = coalFactory.getParticleVelocityLabel( quadNode );
      tsk->requires( Task::NewDW, d_particle_velocity_label, Ghost::AroundCells, 1 );
    } else {
      d_particle_velocity_label = _fieldLabels->d_newCCVelocityLabel;
      tsk->requires( Task::OldDW, d_particle_velocity_label, Ghost::AroundCells, 1 );
    }
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
UnweightedSrcTerm::computeSource( const ProcessorGroup* pc, 
                                  const PatchSubset* patches, 
                                  const MaterialSubset* matls, 
                                  DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw, 
                                  int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> constSrc; 
    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( constSrc, _src_label, matlIndex, patch );
      constSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( constSrc, _src_label, matlIndex, patch ); 
    }

    constCCVariable<Vector> areaFraction;
    const VarLabel* d_areaFractionLabel = VarLabel::find( "areaFraction" );
    old_dw->get(areaFraction, d_areaFractionLabel, matlIndex, patch, Ghost::AroundCells, 1);

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
    CoalModelFactory& coalFactory  = CoalModelFactory::self();

    constCCVariable<double> unwa;
    constCCVariable<Vector> partVel;

    // _required_labels only has 1 element
    for( vector<string>::iterator iter = _required_labels.begin(); iter != _required_labels.end(); ++iter ) {
      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>( &dqmomFactory.retrieve_scalar_eqn(*iter) );
      old_dw->get( unwa, dqmom_eqn->getTransportEqnLabel(), matlIndex, patch, Ghost::None, 0 );

      int quadNode = dqmom_eqn->getQuadNode();
      if( coalFactory.useParticleVelocityModel() ) {
        d_particle_velocity_label = coalFactory.getParticleVelocityLabel( quadNode );
        new_dw->get(partVel, d_particle_velocity_label, matlIndex, patch, Ghost::AroundCells, 1 );
      } else {
        d_particle_velocity_label = _fieldLabels->d_newCCVelocityLabel;
        old_dw->get(partVel, d_particle_velocity_label, matlIndex, patch, Ghost::AroundCells, 1 );
      }
    }      

    Vector Dx = patch->dCell();

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector cxm = c - IntVector(1,0,0);
      IntVector cxp = c + IntVector(1,0,0);
      IntVector cym = c - IntVector(0,1,0);
      IntVector cyp = c + IntVector(0,1,0);
      IntVector czm = c - IntVector(0,0,1);
      IntVector czp = c + IntVector(0,0,1);
 
      constSrc[c] += unwa[c]*( (areaFraction[cxp].x()*(partVel[cxp].x()+partVel[c].x())-areaFraction[c].x()*(partVel[c].x()+partVel[cxm].x()))/(2*Dx.x()) +
                               (areaFraction[cyp].y()*(partVel[cyp].y()+partVel[c].y())-areaFraction[c].y()*(partVel[c].y()+partVel[cym].y()))/(2*Dx.y()) +
                               (areaFraction[czp].z()*(partVel[czp].z()+partVel[c].z())-areaFraction[c].z()*(partVel[c].z()+partVel[czm].z()))/(2*Dx.z()) );
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
UnweightedSrcTerm::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "UnweightedSrcTerm::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &UnweightedSrcTerm::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
UnweightedSrcTerm::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
    }
  }
}

