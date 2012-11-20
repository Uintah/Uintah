#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/UnweightedSrcTerm.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

UnweightedSrcTerm::UnweightedSrcTerm( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _label_sched_init = false; 
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

UnweightedSrcTerm::~UnweightedSrcTerm()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
UnweightedSrcTerm::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  _source_grid_type = CC_SRC; 
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
UnweightedSrcTerm::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "UnweightedSrcTerm::eval";
  Task* tsk = scinew Task(taskname, this, &UnweightedSrcTerm::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }

  const VarLabel* d_areaFractionLabel = VarLabel::find( "areaFraction" );
  tsk->requires(Task::OldDW, d_areaFractionLabel, Ghost::AroundCells, 1);

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 

    std::string label_name = (*iter);
    EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( label_name );

    const VarLabel* unwaLabel = eqn.getTransportEqnLabel();
    tsk->requires( Task::OldDW, unwaLabel, Ghost::None, 0 );

    //DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();

    //for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin();
    //     ieqn != dqmom_eqns.end(); ieqn++){
          
    DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(&eqn);
    int d_quadNode = dqmom_eqn->getQuadNode(); 

    // require particle velocity
    string partVel_name = "vel_qn";
    std::string node;
    std::stringstream out;
    out << d_quadNode;
    node = out.str();
    partVel_name += node;

    const VarLabel* partVelLabel = VarLabel::find( partVel_name );
    tsk->requires( Task::NewDW, partVelLabel, Ghost::AroundCells, 1 );
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
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( constSrc, _src_label, matlIndex, patch ); 
      constSrc.initialize(0.0);
    } else {
      new_dw->allocateAndPut( constSrc, _src_label, matlIndex, patch );
      constSrc.initialize(0.0);
    } 

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
    constCCVariable<double> unwa;
    constCCVariable<Vector> partVel;
    constCCVariable<Vector> areaFraction;
    std::string label_name;

    const VarLabel* d_areaFractionLabel = VarLabel::find( "areaFraction" );
    old_dw->get(areaFraction, d_areaFractionLabel, matlIndex, patch, Ghost::AroundCells, 1);

    for (vector<std::string>::iterator iter = _required_labels.begin(); 
         iter != _required_labels.end(); iter++) { 
   
      label_name = (*iter);
      EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( label_name );

      const VarLabel* unwaLabel = eqn.getTransportEqnLabel();
      old_dw->get( unwa, unwaLabel, matlIndex, patch, Ghost::None, 0 );

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(&eqn);
      int d_quadNode = dqmom_eqn->getQuadNode();

      //ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
      //old_dw->get(partVel, iter->second, matlIndex, patch, Ghost::None, 0 );
 
      string partVel_name = "vel_qn";
      std::string node;
      std::stringstream out;
      out << d_quadNode;
      node = out.str();
      partVel_name += node;

      const VarLabel* partVelLabel = VarLabel::find( partVel_name );
      new_dw->get(partVel, partVelLabel, matlIndex, patch, Ghost::AroundCells, 1 );
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

