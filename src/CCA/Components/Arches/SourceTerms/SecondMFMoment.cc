#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/SecondMFMoment.h>

using namespace std;
using namespace Uintah; 

SecondMFMoment::SecondMFMoment( std::string src_name, MaterialManagerP& materialManager,
                                     vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, materialManager, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

SecondMFMoment::~SecondMFMoment()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
SecondMFMoment::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 
  
//  db->require("density",_density); 
  db->getWithDefault("density_label", _density_name, "density");  
  db->require("scalar_dissipation_label", _scalarDissipation_name);
  
  _source_grid_type = CC_SRC; 
  
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
SecondMFMoment::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "SecondMFMoment::eval";
  Task* tsk = scinew Task(taskname, this, &SecondMFMoment::computeSource, timeSubStep);
  
  if (timeSubStep == 0) {

    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }

  densityLabel = VarLabel::find(_density_name);
  scalarDissLabel = VarLabel::find(_scalarDissipation_name);
  
  if (timeSubStep == 0) {
    tsk->requires( Task::OldDW, densityLabel, Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, scalarDissLabel, Ghost::None, 0 );
  } else {
    tsk->requires( Task::NewDW, densityLabel, Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, scalarDissLabel, Ghost::None, 0 );
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ) ); 
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
SecondMFMoment::computeSource( const ProcessorGroup* pc, 
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
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    
    CCVariable<double> rateSrc; 
    constCCVariable<double> den;    // mixture density
    constCCVariable<double> chi;    // scalar diss

    if (timeSubStep == 0) {
      new_dw->allocateAndPut( rateSrc, _src_label, matlIndex, patch );
      old_dw->get( den, densityLabel, matlIndex, patch, Ghost::None , 0);
      old_dw->get( chi, scalarDissLabel, matlIndex, patch, Ghost::None , 0);
      rateSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( rateSrc, _src_label, matlIndex, patch );
      new_dw->get( den, densityLabel, matlIndex, patch, Ghost::None , 0);
      new_dw->get( chi, scalarDissLabel, matlIndex, patch, Ghost::None , 0);
    }
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      rateSrc[c] = - den[c] * chi[c];
    }
    
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
SecondMFMoment::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "SecondMFMoment::initialize"; 
  
  Task* tsk = scinew Task(taskname, this, &SecondMFMoment::initialize);
  
  tsk->computes(_src_label);
  
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }
  
  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ) );
}
//---------------------------------------------------------------------------
// Method: initialization
//---------------------------------------------------------------------------
void 
SecondMFMoment::initialize( const ProcessorGroup* pc, 
                                          const PatchSubset* patches, 
                                          const MaterialSubset* matls, 
                                          DataWarehouse* old_dw, 
                                          DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    CCVariable<double> src;
    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 
    src.initialize(0.0); 
  }
}
