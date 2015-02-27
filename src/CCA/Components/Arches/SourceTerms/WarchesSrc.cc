#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/WarchesSrc.h>
#ifdef WASATCH_IN_ARCHES
//-- SpatialOps includes --//
#include <spatialops/Nebo.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#endif

//===========================================================================

using namespace std;
using namespace Uintah; 

WarchesSrc::WarchesSrc( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

WarchesSrc::~WarchesSrc()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
WarchesSrc::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault("constant",_constant, 0.); 

  //multiply the source term by a density
  if ( db->findBlock("density_weighted") ){

    _density_weight = true; 

  } else { 

    _density_weight = false; 
  
  }

  _source_grid_type = CC_SRC; 
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
WarchesSrc::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "WarchesSrc::eval";
  Task* tsk = scinew Task(taskname, this, &WarchesSrc::computeSource, timeSubStep);

  Ghost::GhostType  gType;
  int nGhosts;
  
#ifdef WASATCH_IN_ARCHES
  gType = Ghost::AroundCells;
  nGhosts = Wasatch::get_n_ghost<SVolField>();
#else
  gType = Ghost::None;
  nGhosts = 0;
#endif

  if (timeSubStep == 0) {

    tsk->computes(_src_label);

  } else {
#ifdef WASATCH_IN_ARCHES
    
    tsk->modifiesWithScratchGhost( _src_label,
                                  level->eachPatch()->getUnion(), Uintah::Task::ThisLevel,
                                  _shared_state->allArchesMaterials()->getUnion(), Uintah::Task::NormalDomain,
                                  gType, nGhosts);
#else
    tsk->modifies(_src_label);
#endif

  }

  if ( _density_weight ){ 
    tsk->requires( Task::NewDW, VarLabel::find("density"), Ghost::None, 0 ); 
  }

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE SOURCe
    //tsk->requires( Task::OldDW, .... ); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
WarchesSrc::computeSource( const ProcessorGroup* pc, 
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
    constCCVariable<double> density; 
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( constSrc, _src_label, matlIndex, patch ); 
      constSrc.initialize(0.0);
    } else {
      new_dw->allocateAndPut( constSrc, _src_label, matlIndex, patch );
      constSrc.initialize(0.0);
    } 
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
WarchesSrc::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "WarchesSrc::initialize"; 

  Task* tsk = scinew Task(taskname, this, &WarchesSrc::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
WarchesSrc::initialize( const ProcessorGroup* pc, 
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

