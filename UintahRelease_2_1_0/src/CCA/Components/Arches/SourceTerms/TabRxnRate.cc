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

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( _rxn_rate );

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
TabRxnRate::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "TabRxnRate::eval";
  Task* tsk = scinew Task(taskname, this, &TabRxnRate::computeSource, timeSubStep);

  if (timeSubStep == 0) {

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
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
TabRxnRate::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "TabRxnRate::initialize";

  Task* tsk = scinew Task(taskname, this, &TabRxnRate::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
TabRxnRate::initialize( const ProcessorGroup* pc,
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
