#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/TabRxnRate.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

TabRxnRate::TabRxnRate( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _label_sched_init = false; 
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

TabRxnRate::~TabRxnRate()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
TabRxnRate::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->require("rxn_rate",_rxn_rate); 

  _source_grid_type = CC_SRC; 

	_table_lookup_species.push_back(_rxn_rate); 
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
TabRxnRate::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "TabRxnRate::eval";
  Task* tsk = scinew Task(taskname, this, &TabRxnRate::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }

  const VarLabel* the_label = VarLabel::find(_rxn_rate); 
  tsk->requires( Task::OldDW, the_label, Ghost::None, 0 ); 


  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
TabRxnRate::computeSource( const ProcessorGroup* pc, 
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

    CCVariable<double> rateSrc; 
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( rateSrc, _src_label, matlIndex, patch ); 
      rateSrc.initialize(0.0);
    } else {
      new_dw->allocateAndPut( rateSrc, _src_label, matlIndex, patch );
    } 

    constCCVariable<double> rxn_rate; 
    const VarLabel* the_label = VarLabel::find(_rxn_rate); 
    old_dw->get( rxn_rate, the_label, matlIndex, patch, Ghost::None, 0 ); 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      rateSrc[c] = rxn_rate[c]; 
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
TabRxnRate::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "TabRxnRate::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &TabRxnRate::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
TabRxnRate::dummyInit( const ProcessorGroup* pc, 
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

