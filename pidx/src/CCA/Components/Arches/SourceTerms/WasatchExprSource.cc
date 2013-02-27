#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/WasatchExprSource.h>

using namespace std;
using namespace Uintah; 

WasatchExprSource::WasatchExprSource( std::string src_name, SimulationStateP& shared_state,
                                     vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _label_sched_init = false; 
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

WasatchExprSource::~WasatchExprSource()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
WasatchExprSource::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 
  
  db->require("expr",_was_expr); 
  
  _source_grid_type = CC_SRC; 
  
	_wasatch_expr_names.push_back(_was_expr); 
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
WasatchExprSource::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "WasatchExprSource::eval";
  Task* tsk = scinew Task(taskname, this, &WasatchExprSource::computeSource, timeSubStep);
  
  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;
    
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }
  
  const VarLabel* wasatch_label = VarLabel::find(_was_expr); 
  tsk->requires( Task::OldDW, wasatch_label, Ghost::None, 0 ); 
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
WasatchExprSource::computeSource( const ProcessorGroup* pc, 
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
    
    constCCVariable<double> was_expr; 
    const VarLabel* wasatch_label = VarLabel::find(_was_expr); 
    
    /* This check must occur since wasatch expressions are not in old_dw 
     on the dummy time step until after the MPMArches calculation. 
     This has no effect on arches-wasatch only simulations.
     NOTE: Remove this check if MPMArches-Wasatch dummy Init changed*/
    if ( old_dw->exists(wasatch_label, matlIndex, patch ) ){ 
      old_dw->get( was_expr, wasatch_label, matlIndex, patch, Ghost::AroundCells, 0 ); 
      rateSrc.copyData(was_expr);
    } else {
      rateSrc.initialize(0.0);
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
WasatchExprSource::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "WasatchExprSource::dummyInit"; 
  
  Task* tsk = scinew Task(taskname, this, &WasatchExprSource::dummyInit);
  
  tsk->computes(_src_label);
  
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }
  
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials() );
}
//---------------------------------------------------------------------------
// Method: Dummy initialization
//---------------------------------------------------------------------------
void 
WasatchExprSource::dummyInit( const ProcessorGroup* pc, 
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
  }
}
