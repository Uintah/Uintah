#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/ConstantSourceTerm.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

ConstantSourceTerm::ConstantSourceTerm( std::string srcName, SimulationStateP& sharedState,
                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{
  _src_label = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 
}

ConstantSourceTerm::~ConstantSourceTerm()
{}


//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ConstantSourceTerm::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault("constant",d_constant, 0.); 

}


//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
ConstantSourceTerm::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ConstantSourceTerm::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &ConstantSourceTerm::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
ConstantSourceTerm::dummyInit( const ProcessorGroup* pc, 
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


//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
ConstantSourceTerm::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ConstantSourceTerm::computeSource";
  Task* tsk = scinew Task(taskname, this, &ConstantSourceTerm::computeSource, timeSubStep);

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

  //for (vector<std::string>::iterator iter = _required_labels.begin(); 
  //     iter != _required_labels.end(); iter++) { 
  //  // require any variables needed to compute the source
  //  //tsk->requires( Task::OldDW, .... ); 
  //}

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ConstantSourceTerm::computeSource( const ProcessorGroup* pc, 
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
    } else {
      new_dw->getModifiable( constSrc, _src_label, matlIndex, patch ); 
    }

    //for (vector<std::string>::iterator iter = _required_labels.begin(); 
    //     iter != _required_labels.end(); iter++) { 
    //  //CCVariable<double> temp; 
    //  //old_dw->get( *iter.... ); 
    //}

    //constSrc.initialize(d_constant);
    proc0cout << "Initializing source term to " << d_constant << endl;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      constSrc[c] = d_constant; 
    }
  }
}

