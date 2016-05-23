#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/ConstSrcTerm.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

ConstSrcTerm::ConstSrcTerm( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

ConstSrcTerm::~ConstSrcTerm()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ConstSrcTerm::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault("constant",_constant, 0.); 

  //multiply the constant source term by a variable
  if ( db->findBlock("multiply_by_variable") ){
    db->findBlock("multiply_by_variable")->require("variable_string_name",_mult_var_string);
    db->findBlock("multiply_by_variable")->getWithDefault("NewDW_only",_NewDW_only,false);
    _mult_by_variable = true; 
  } else { 
    _mult_by_variable = false; 
  }

  _source_grid_type = CC_SRC; 
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
ConstSrcTerm::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ConstSrcTerm::eval";
  Task* tsk = scinew Task(taskname, this, &ConstSrcTerm::computeSource, timeSubStep);

  if (timeSubStep == 0) { 

    tsk->computes(_src_label);

  } else {
    tsk->modifies(_src_label); 

  }

  if ( _mult_by_variable ){
    if ( _NewDW_only ){
      tsk->requires( Task::NewDW, VarLabel::find(_mult_var_string), Ghost::None, 0 ); 
    } else {
      if (timeSubStep == 0) { 
        tsk->requires( Task::OldDW, VarLabel::find(_mult_var_string), Ghost::None, 0 ); 
      } else {
        tsk->requires( Task::NewDW, VarLabel::find(_mult_var_string), Ghost::None, 0 ); 
      }
    }
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ConstSrcTerm::computeSource( const ProcessorGroup* pc, 
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
    constCCVariable<double> mult_variable; 

    if ( timeSubStep ==0 ){  // double check this for me jeremy
      new_dw->allocateAndPut( constSrc, _src_label, matlIndex, patch );
      constSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( constSrc, _src_label, matlIndex, patch ); 
      constSrc.initialize(0.0);
    } 

    if ( _mult_by_variable ){ 
      if ( _NewDW_only ){
        new_dw->get( mult_variable, VarLabel::find(_mult_var_string), matlIndex, patch, Ghost::None, 0 );
      } else {
        if (timeSubStep == 0) { 
          old_dw->get( mult_variable, VarLabel::find(_mult_var_string), matlIndex, patch, Ghost::None, 0 );
        } else {
          new_dw->get( mult_variable, VarLabel::find(_mult_var_string), matlIndex, patch, Ghost::None, 0 );
        }
      }
    }


    if ( _mult_by_variable ) { 
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter; 
        constSrc[c] += mult_variable[c] * _constant; 
      }
    } else { 
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter; 
        constSrc[c] += _constant; 
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
ConstSrcTerm::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ConstSrcTerm::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ConstSrcTerm::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
ConstSrcTerm::initialize( const ProcessorGroup* pc, 
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

